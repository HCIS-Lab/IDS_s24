import os
import ujson
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from tqdm import tqdm
import sys
import cv2
import gzip
import laspy
import io
import utils
from skimage.transform import rotate
from copy import deepcopy




class CARLA_Data(Dataset):  # pylint: disable=locally-disabled, invalid-name
	"""
			Custom dataset that dynamically loads a CARLA dataset from disk.
			"""
	def __init__(self,
							root,
							config):
	
		self.config = config
		self.clear_frame = 0

		self.boxes = []
		self.future_boxes = []
		self.topdowns = []

		total_routes = 0
		perfect_routes = 0
		crashed_routes = 0

		for sub_root in tqdm(root, file=sys.stdout):

			# list subdirectories in root
			routes = next(os.walk(sub_root))[1]

			for route in routes:
				route_dir = sub_root + '/' + route

				perfect_routes += 1
				num_seq = len(os.listdir(route_dir + '/boxes'))

				for seq in range(config.skip_first, num_seq - self.config.pred_len - self.config.seq_len):
					if seq % config.train_sampling_rate != 0:
							continue
					# load input seq and pred seq jointly
					box = []
					future_box = []
					topdown = []
									
					topdown.append(route_dir  + "/birdview" + ("/%04d.png" % (seq)))
					forecast_step = int(config.forecast_time / (config.data_save_freq / config.carla_fps))
					box.append(route_dir + '/boxes' + (f'/{(seq):04}.json.gz'))
					future_box.append(route_dir + '/boxes' + (f'/{(seq + forecast_step):04}.json.gz'))

					self.boxes.append(box)
					self.future_boxes.append(future_box)
					self.topdowns.append(topdown)


		# There is a complex "memory leak"/performance issue when using Python
		# objects like lists in a Dataloader that is loaded with
		# multiprocessing, num_workers > 0
		# A summary of that ongoing discussion can be found here
		# https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
		# A workaround is to store the string lists as numpy byte objects
		# because they only have 1 refcount.
		self.boxes = np.array(self.boxes).astype(np.string_)
		self.future_boxes = np.array(self.future_boxes).astype(np.string_)
		self.topdowns = np.array(self.topdowns).astype(np.string_)


	def __len__(self):
		"""Returns the length of the dataset. """
		return self.boxes.shape[0]

	def __getitem__(self, index):
		data = {}

		boxes = self.boxes[index]
		future_boxes = self.future_boxes[index]
		topdowns = self.topdowns[index]
	
		# Load bounding boxes
		with gzip.open(str(boxes[0], encoding='utf-8'), 'rt', encoding='utf-8') as f2:
			boxes = ujson.load(f2)
		with gzip.open(str(future_boxes[0], encoding='utf-8'), 'rt', encoding='utf-8') as f2:
			future_boxes = ujson.load(f2)
	

		bounding_boxes, quant_future_bounding_boxes, future_bounding_boxes = self.parse_bounding_boxes(boxes, future_boxes)

		# Pad bounding boxes to a fixed number
		bounding_boxes = np.array(bounding_boxes)
		bounding_boxes_padded = np.zeros((self.config.max_num_bbs, 8), dtype=np.float32)

		future_bounding_boxes = np.array(future_bounding_boxes)
		future_bounding_boxes_padded = np.zeros((self.config.max_num_bbs, 8), dtype=np.float32)

		quant_future_bounding_boxes = np.array(quant_future_bounding_boxes)
		quant_future_bounding_boxes_padded = np.ones((self.config.max_num_bbs, 8), dtype=np.int32) * self.config.ignore_index

		if bounding_boxes.shape[0] > 0:
			if bounding_boxes.shape[0] <= self.config.max_num_bbs:
				bounding_boxes_padded[:bounding_boxes.shape[0], :] = bounding_boxes
				future_bounding_boxes_padded[:future_bounding_boxes.shape[0], :] = future_bounding_boxes
				quant_future_bounding_boxes_padded[:quant_future_bounding_boxes.shape[0], :] = quant_future_bounding_boxes
			else:
				bounding_boxes_padded[:self.config.max_num_bbs, :] = bounding_boxes[:self.config.max_num_bbs]
				future_bounding_boxes_padded[:self.config.max_num_bbs, :] = future_bounding_boxes[:self.config.max_num_bbs]
				quant_future_bounding_boxes_padded[:self.config.max_num_bbs, :] = quant_future_bounding_boxes[:self.config.max_num_bbs]

		data['bounding_boxes'] = bounding_boxes_padded
		data['quant_future_bounding_boxes'] = quant_future_bounding_boxes_padded
		data["future_bounding_boxes"] = future_bounding_boxes_padded
  
		topdown = cv2.imread(str(topdowns[0], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
		topdown = cv2.cvtColor(topdown, cv2.COLOR_BGR2RGB)
		topdown = topdown[72:72+256, 72:72+256]
		data['topdown'] = topdown

		return data

	def parse_bounding_boxes(self, boxes, future_boxes):
		# Find ego matrix of the current time step, i.e. the coordinate frame we want to use:
		ego_matrix = None
		ego_yaw = None
		# ego_car always exists
		for ego_candiate in boxes:
			if ego_candiate['class'] == 'ego_car':
				ego_matrix = np.array(ego_candiate['matrix'])
				ego_yaw = utils.extract_yaw_from_matrix(ego_matrix)
				break

		bboxes = []
		quant_future_bboxes = []
		future_bounding_boxes = []
		for current_box in boxes:
			# Ego car is always at the origin. We don't predict it.
			if current_box['class'] == 'ego_car':
				continue

			bbox, height = self.get_bbox_label(current_box)

			if current_box['class'] == 'traffic_light':
				continue

			if current_box['class'] == 'stop_sign':
				continue

			# Filter bb that are outside of the detection range.
			if bbox[0] <= self.config.min_x \
					or bbox[0] >= self.config.max_x \
					or bbox[1] <= self.config.min_y \
					or bbox[1] >= self.config.max_y \
					or height  <= self.config.min_z \
					or height  >= self.config.max_z:
				continue

			# Load bounding boxes to forecast
			exists = False
			for future_box in future_boxes:
				# We only forecast boxes visible in the current frame
				if future_box['id'] == current_box['id'] and future_box['class'] in ('car', 'walker'):
					# Found a valid box
					# Get values in current coordinate system
					future_box_matrix = np.array(future_box['matrix'])
					relative_pos = utils.get_relative_pos(ego_matrix, future_box_matrix)
					# Update position into current coordinate system
					future_box['position'] = [relative_pos[0], relative_pos[1], relative_pos[2]]
					future_yaw = utils.extract_yaw_from_matrix(future_box_matrix)
					relative_yaw = utils.normalize_angle(future_yaw - ego_yaw)
					future_box['yaw'] = relative_yaw

					future_bounding_box, _ = self.get_bbox_label(future_box)
					future_bounding_boxes.append(deepcopy(future_bounding_box))
					quantized_future_box = self.quantize_box(future_bounding_box)
					quant_future_bboxes.append(quantized_future_box)
					exists = True
					break

			if not exists:
				# Bounding box has no future counterpart. Add a dummy with ignore index
				quant_future_bboxes.append(
					np.array([
							self.config.ignore_index, self.config.ignore_index, self.config.ignore_index,
							self.config.ignore_index, self.config.ignore_index, self.config.ignore_index,
							self.config.ignore_index, self.config.ignore_index
					]))
				future_bounding_boxes.append(
					np.zeros((8), dtype=np.float32))	

			bboxes.append(bbox)

		return bboxes, quant_future_bboxes, future_bounding_boxes

	def get_bbox_label(self, bbox_dict):
		x = bbox_dict['position'][0]
		y = bbox_dict['position'][1]
  
		# center_x, center_y, w, h, yaw
		bbox = np.array([x, y, bbox_dict['extent'][0], bbox_dict['extent'][1], 0, 0, 0, 0])
		bbox[4] = utils.normalize_angle(bbox_dict['yaw'])

		if bbox_dict['class'] == 'car':
			bbox[5] = bbox_dict['speed']
			bbox[6] = bbox_dict['brake']
			bbox[7] = 0
		elif bbox_dict['class'] == 'walker':
			bbox[5] = bbox_dict['speed']
			bbox[7] = 1
		elif bbox_dict['class'] == 'traffic_light':
			bbox[7] = 2
		elif bbox_dict['class'] == 'stop_sign':
			bbox[7] = 3
		return bbox, bbox_dict['position'][2]

	def quantize_box(self, boxes):
		"""Quantizes a bounding box into bins and writes the index into the array a classification label"""
		# range of xy is [-32, 32]
		# range of yaw is [-pi, pi]
		# range of speed is [0, 60]
		# range of extent is [0, 30]
		# Normalize all values between 0 and 1
		boxes[0] = (boxes[0] + self.config.max_x) / (self.config.max_x - self.config.min_x)
		boxes[1] = (boxes[1] + self.config.max_y) / (self.config.max_y - self.config.min_y)

		# quantize extent
		boxes[2] = boxes[2] / 30
		boxes[3] = boxes[3] / 30

		# quantize yaw
		boxes[4] = (boxes[4] + np.pi) / (2 * np.pi)

		# quantize speed, future_bounding_boxes max speed to m/s
		boxes[5] = boxes[5] / (self.config.max_speed_pred / 3.6)

		# 6 Brake is already in 0, 1
		# Clip values that are outside the range we classify
		boxes[:7] = np.clip(boxes[:7], 0, 1)

		size_pos = pow(2, self.config.model_precision_pos)
		size_speed = pow(2, self.config.model_precision_speed)
		size_angle = pow(2, self.config.model_precision_angle)

		boxes[[0, 1, 2, 3]] = (boxes[[0, 1, 2, 3]] * (size_pos - 1)).round()
		boxes[4] = (boxes[4] * (size_angle - 1)).round()
		boxes[5] = (boxes[5] * (size_speed - 1)).round()
		boxes[6] = boxes[6].round()

		return boxes.astype(np.int32)