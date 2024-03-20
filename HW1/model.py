"""
Planning TransFormer implementation.
"""

import logging
import numpy as np
import torch
from torch import nn
from einops import rearrange
from PIL import Image
from pathlib import Path
import utils 
import cv2
from copy import deepcopy

from transformers import (
		AutoConfig,
		AutoModel,
)

logger = logging.getLogger(__name__)


class Learn_Forecast(nn.Module):
	"""
	Neural network that takes in bounding boxes and outputs waypoints for driving.
	"""

	def __init__(self, config):
		super().__init__()
		self.config = config

		precisions = [
				self.config.model_precision_pos, self.config.model_precision_pos, self.config.model_precision_pos,
				self.config.model_precision_pos, self.config.model_precision_angle, self.config.model_precision_speed,
				self.config.model_precision_brake
		]

		self.vocab_size = [2**i for i in precisions]

		auto_config = AutoConfig.from_pretrained(self.config.model_hf_checkpoint)
		n_embd = auto_config.hidden_size
		self.bert = AutoModel.from_config(config=auto_config)
		self.bert.pooler.requires_grad_(False)
		self.bert.embeddings.word_embeddings.weight.requires_grad_(False)

		# token embedding
		self.tok_emb = nn.Linear(self.config.model_num_attributes, n_embd)
		# object type embedding
		self.obj_token = nn.ParameterList(
				[nn.Parameter(torch.randn(1, self.config.model_num_attributes)) for _ in range(self.config.model_object_types)])
		self.obj_emb = nn.ModuleList(
				[nn.Linear(self.config.model_num_attributes, n_embd) for _ in range(self.config.model_object_types)])
		self.drop = nn.Dropout(self.config.model_embd_pdrop)

		# decoder head forecasting
		# one head for each attribute type -> we have different precision per attribute
		self.heads = nn.ModuleList([nn.Linear(n_embd, self.vocab_size[i]) for i in range(self.config.model_num_attributes)])

		self.apply(self._init_weights)

		self.loss_forecast = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

		logger.info('number of parameters: %e', sum(p.numel() for p in self.parameters()))

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
			if isinstance(module, nn.Linear) and module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def create_optimizer_groups(self, weight_decay):
		"""
		This long function is unfortunately doing something very simple and is
		being very defensive:
		We are separating out all parameters of the model into two buckets:
		those that will experience
		weight decay for regularization and those that won't
		(biases, and layernorm/embedding weights).
		We are then returning the optimizer groups.
		"""

		# separate out all parameters to those that will and won't experience
		# regularizing weight decay
		decay = set()
		no_decay = set()
		whitelist_weight_modules = torch.nn.Linear
		blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
		for mn, m in self.named_modules():
			for pn, _ in m.named_parameters():
				fpn = f'{mn}.{pn}' if mn else pn  # full param name

				if pn.endswith('bias'):
					# all biases will not be decayed
					no_decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
					# weights of whitelist modules will be weight decayed
					decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
					# weights of blacklist modules will NOT be weight decayed
					no_decay.add(fpn)
				elif pn.endswith('_ih') or pn.endswith('_hh'):
					# all recurrent weights will not be decayed
					no_decay.add(fpn)
				elif pn.endswith('_emb') or '_token' in pn:
					no_decay.add(fpn)
				elif 'bias_ih_l0' in pn or 'bias_hh_l0' in pn:
					no_decay.add(fpn)
				elif 'weight_ih_l0' in pn or 'weight_hh_l0' in pn:
					decay.add(fpn)

		# validate that we considered every parameter
		param_dict = dict(self.named_parameters())
		inter_params = decay & no_decay
		union_params = decay | no_decay
		assert (len(inter_params) == 0), f'parameters {str(inter_params)} made it into both decay/no_decay sets!'
		assert (
				len(param_dict.keys() - union_params) == 0
		), f'parameters {str(param_dict.keys() - union_params)} were not ' \
			 f'separated into either decay/no_decay set!'

		# create the pytorch optimizer object
		optim_groups = [
				{
						'params': [param_dict[pn] for pn in sorted(list(decay))],
						'weight_decay': weight_decay,
				},
				{
						'params': [param_dict[pn] for pn in sorted(list(no_decay))],
						'weight_decay': 0.0,
				},
		]
		return optim_groups

	def forward(self, bounding_boxes):

		num_boxes = bounding_boxes.shape[1]

		input_batch_type = bounding_boxes[:, :, 7]  # class of bounding box
		input_batch_data = bounding_boxes[:, :, :7]

		# create masks by object type
		car_mask = (input_batch_type == 0).unsqueeze(-1)
		walker_mask = (input_batch_type == 1).unsqueeze(-1)
		# CLS token will be other
		masks = [car_mask, walker_mask]

		# get size of input
		# batch size, number of objects, number of attributes
		(batch, objects, _) = input_batch_data.shape

		# embed tokens object wise (one object -> one token embedding)
		input_batch_data = rearrange(input_batch_data, 'b objects attributes -> (b objects) attributes')
		embedding = self.tok_emb(input_batch_data)
		embedding = rearrange(embedding, '(b o) features -> b o features', b=batch, o=objects)

		# create object type embedding
		obj_embeddings = [
				self.obj_emb[i](self.obj_token[i])  # pylint: disable=locally-disabled, unsubscriptable-object
				for i in range(self.config.model_object_types)
		]  # list of a tensors of size 1 x features

		# add object type embedding to embedding (mask needed to only add to the correct tokens)
		embedding = [(embedding + obj_embeddings[i]) * masks[i] for i in range(self.config.model_object_types)]
		embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)

		# embedding dropout
		x = self.drop(embedding)

		# Transformer Encoder; use embedding for hugging face model and get output states and attention map
		output = self.bert(**{'inputs_embeds': x}, output_attentions=True)

		tf_features = output.last_hidden_state
		preidction_features = tf_features[:, 0:num_boxes + 1, :]

		# forecasting encoding
		# vocab_size (vocab_size differs for each attribute)
		# we forcast only for vehicles and pedestrians, (forecasts for other classes are ignore in the loss)
		box_pred_logits = []
		for i in range(self.config.model_num_attributes):
			head_output = self.heads[i](tf_features)

			box_pred_logits.append(head_output)

		return box_pred_logits

	def compute_loss(self, pred_future_bounding_box, future_bounding_box_label):
		# Put boxes onto batch dimension to parallelize
		pred_future_bounding_box = [
				rearrange(box, 'b o vocab_size -> (b o) vocab_size') for box in pred_future_bounding_box
		]
		future_bounding_box_label = rearrange(future_bounding_box_label, 'b o vocab_size -> (b o) vocab_size')

		# Compute mean cross entropy loss
		loss_forecast = 0
		for i in range(len(pred_future_bounding_box)):
			loss_forecast += self.loss_forecast(pred_future_bounding_box[i], future_bounding_box_label[:, i])

		loss_forecast = loss_forecast / len(pred_future_bounding_box)

		return loss_forecast