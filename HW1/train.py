'''
Training script for training transFuser and related models.
Usage:
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=1234576890 --rdzv_backend=c10d
train.py --logdir /path/to/logdir --root_dir /path/to/dataset_root/ --id exp_000 --cpu_cores 8
'''

import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record
import torch.multiprocessing as mp

from config import GlobalConfig
from data import CARLA_Data
from model import Learn_Forecast
import utils


import pathlib
import random
import pickle

from collections import defaultdict

# On some systems it is necessary to increase the limit on open file descriptors.
try:
	import resource
	rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
	resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
except (ModuleNotFoundError, ImportError) as e:
	print(e)


@record  # Records error and tracebacks in case of failure
def main():
	torch.cuda.empty_cache()

	# Loads the default values for the argparse so we have only one default
	config = GlobalConfig()

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=str, default=config.id, help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=config.epochs, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=config.lr, help='Learning rate.')
	parser.add_argument('--batch_size',
											type=int,
											default=config.batch_size,
											help='Batch size for one GPU. When training with multiple GPUs the effective'
											' batch size will be batch_size*num_gpus')
	parser.add_argument('--logdir', type=str, required=True, help='Directory to log data and models to.')
	parser.add_argument('--load_file',
											type=str,
											default=config.load_file,
											help='Model to load for initialization.'
											'Expects the full path with ending /path/to/model.pth '
											'Optimizer files are expected to exist in the same directory')
	parser.add_argument('--root_dir', type=str, required=True, help='Root directory of your training data')
	parser.add_argument('--continue_epoch',
											type=int,
											default=int(config.continue_epoch),
											help='Whether to continue the training from the loaded epoch or from 0.')
	parser.add_argument('--cpu_cores',
											type=int,
											required=True,
											help='How many cpu cores are available on the machine.'
											'The code will spawn a thread for each cpu.')
	parser.add_argument('--validate',
											type=int,
											default=int(0),
											help='run validation without training')
	parser.add_argument('--model',
											type=str,
											default="constant",
											help='select bounding boxes forecasting model between learn & constant')
	parser.add_argument('--forecast_time',
											type=float,
											default=config.forecast_time,
											help='forecast time')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	device = torch.device(f'cuda')

	ngpus_per_node = torch.cuda.device_count()
	ncpus_per_node = args.cpu_cores
	num_workers = int(ncpus_per_node / ngpus_per_node)
	torch.cuda.device(device)
	# We want the highest performance
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.allow_tf32 = True

	# Configure config. Converts all arguments into config attributes
	config.initialize(**vars(args))

	config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0))

	# Data, configures config. Create before the model
	train_set = CARLA_Data(root=config.train_data,
												 config=config)
 
	val_set = CARLA_Data(root=config.val_data, config=config)

	start_epoch = 0  # Epoch to continue training from
	# Create model and optimizers
	if args.model == "constant":
		model = utils.ConstantModel(config.forecast_time)
		optimizer = None
		scheduler = None
		scaler = None
	else:
		model = Learn_Forecast(config)
		model.cuda(device=device)

		if not args.load_file is None:
			# Load checkpoint
			print('=============load=================')
			# Add +1 because the epoch before that was already trained
			load_name = str(pathlib.Path(args.load_file).stem)
			if args.continue_epoch:
				start_epoch = int(''.join(filter(str.isdigit, load_name))) + 1
			model.load_state_dict(torch.load(args.load_file, map_location=device), strict=False)
  
		model_parameters = filter(lambda p: p.requires_grad, model.parameters())
		num_params = sum(np.prod(p.size()) for p in model_parameters)
		print('Total trainable parameters: ', num_params)

		params = model.parameters()

		optimizer = optim.AdamW(params, lr=args.lr, amsgrad=True)
		if not args.load_file is None and args.continue_epoch:
			optimizer.load_state_dict(torch.load(args.load_file.replace('model_', 'optimizer_'), map_location=device))
		if config.use_cosine_schedule:
			scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
																																			T_0=config.cosine_t0,
																																			T_mult=config.cosine_t_mult,
																																			verbose=False)
		else:
			milestones = [config.schedule_reduce_epoch_01, config.schedule_reduce_epoch_02]
			scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
																											milestones,
																											gamma=config.multi_step_lr_decay,
																											verbose=True)

		scaler = torch.cuda.amp.GradScaler(enabled=False)
		if not args.load_file is None:
			if args.continue_epoch:
				scheduler.load_state_dict(torch.load(args.load_file.replace('model_', 'scheduler_'), map_location=device))
				scaler.load_state_dict(torch.load(args.load_file.replace('model_', 'scaler_'), map_location=device))


	g_cuda = torch.Generator(device='cpu')
	g_cuda.manual_seed(torch.initial_seed())

	sampler_train = torch.utils.data.RandomSampler(train_set)
	sampler_val = torch.utils.data.SequentialSampler(val_set)
 
	dataloader_train = DataLoader(train_set,
																sampler=sampler_train,
																batch_size=args.batch_size,
																worker_init_fn=seed_worker,
																generator=g_cuda,
																num_workers=num_workers,
																pin_memory=False,
																drop_last=True)
	dataloader_val = DataLoader(val_set,
															sampler=sampler_val,
															batch_size=1,
															worker_init_fn=seed_worker,
															generator=g_cuda,
															num_workers=num_workers,
															pin_memory=False,
															drop_last=True)

	# Create logdir
	print('Created dir:', args.logdir)
	os.makedirs(args.logdir, exist_ok=True)

	# We only need one process to log the losses
	writer = SummaryWriter(log_dir=args.logdir)
	# Log args
	with open(os.path.join(args.logdir, 'args.txt'), 'w', encoding='utf-8') as f:
		json.dump(args.__dict__, f, indent=2)

	with open(os.path.join(args.logdir, 'config.pickle'), 'wb') as f2:
		pickle.dump(config, f2, protocol=4)


	trainer = Engine(model=model,
									 optimizer=optimizer,
									 dataloader_train=dataloader_train,
									 dataloader_val=dataloader_val,
									 args=args,
									 config=config,
									 writer=writer,
									 device=device,
									 cur_epoch=start_epoch,
									 scheduler=scheduler,
									 scaler=scaler)
	
	if args.model == "learn" and not bool(args.validate):
		for epoch in range(trainer.cur_epoch, args.epochs):
			# Update the seed depending on the epoch so that the distributed
			# sampler will use different shuffles across different epochs
			# sampler_train.set_epoch(epoch)

			trainer.train()
			torch.cuda.empty_cache()

			if not config.use_cosine_schedule:
				scheduler.step()

			trainer.save()

			trainer.cur_epoch += 1

	trainer.validate()


class Engine(object):
	"""
		Engine that runs training.
		"""

	def __init__(self,
							 model,
							 optimizer,
							 dataloader_train,
							 dataloader_val,
							 args,
							 config,
							 writer,
							 device,
							 scheduler,
							 scaler,
							 rank=0,
							 world_size=1,
							 cur_epoch=0):

		self.cur_epoch = cur_epoch
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10
		self.model = model
		self.optimizer = optimizer
		self.dataloader_train = dataloader_train
		self.dataloader_val = dataloader_val
		self.args = args
		self.config = config
		self.writer = writer
		self.device = device
		self.rank = rank
		self.world_size = world_size
		self.step = 0
		self.vis_save_path = self.args.logdir + r'/visualizations'
		self.scheduler = scheduler
		self.iters_per_epoch = len(self.dataloader_train)
		self.scaler = scaler
	
		if self.config.debug:
			pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)


	def load_data_compute_loss(self, data, validation=False):
		# Validation = True will compute additional metrics not used for optimization
		# Load data used in both methods
		bounding_box_label = data['bounding_boxes'].to(self.device, dtype=torch.float32)
		quant_future_bounding_box_label = data['quant_future_bounding_boxes'].to(self.device, dtype=torch.long)
		future_bounding_boxes = data['future_bounding_boxes']
		topdown = data['topdown']

		# Load model specific data and execute model
		pred_future_bounding_box = self.model.forward(bounding_boxes=bounding_box_label)

		if validation:
			# Debug visualizations
			with torch.no_grad():
				metric, valid = utils.visualize(
											self.config,
											self.args,
											self.vis_save_path,
											self.step,
											topdown,
											future_bounding_boxes,
											pred_future_bounding_box
									)
				return metric, valid
    
		loss = 0.0
		if self.args.model == "learn":
			loss = self.model.compute_loss(pred_future_bounding_box=pred_future_bounding_box,
																		future_bounding_box_label=quant_future_bounding_box_label)
		
		return loss, None

	def train(self):
		self.model.train()

		num_batches = 0
		detailed_losses_epoch = defaultdict(float)
		self.optimizer.zero_grad(set_to_none=False)

		# Train loop
		for i, data in enumerate(tqdm(self.dataloader_train, disable=self.rank != 0)):
			with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
				loss, _ = self.load_data_compute_loss(data, validation=False)
				detailed_losses_epoch['loss_forecast'] += loss

			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()
			self.optimizer.zero_grad(set_to_none=True)

			num_batches += 1

			if self.config.use_cosine_schedule:
				self.scheduler.step(self.cur_epoch + i / self.iters_per_epoch)

		self.optimizer.zero_grad(set_to_none=True)
		torch.cuda.empty_cache()

		self.log_losses(detailed_losses_epoch, num_batches, '')

	@torch.inference_mode()
	def validate(self):
		if self.args.model == "learn":
			self.model.eval()

		num_batches = 0
		detailed_val_losses_epoch = defaultdict(float)

		# Evaluation loop loop
		for data in tqdm(self.dataloader_val, disable=self.rank != 0):
			metric, valid = self.load_data_compute_loss(data, validation=True)
			self.step += 1
			for key, value in metric.items():
				detailed_val_losses_epoch[key] += float(value)

			if valid:
				num_batches += 1

			del metric
			del valid

		self.log_losses(detailed_val_losses_epoch, num_batches, 'val_')

	def log_losses(self, detailed_losses_epoch, num_batches, prefix=''):
		# Collecting the losses from all GPUs has led to issues.
		# I simply log the loss from GPU 0 for now they should be similar.
		if self.rank == 0:
			for key, value in detailed_losses_epoch.items():
				self.writer.add_scalar(prefix + key, value / num_batches, self.cur_epoch)
				print(f"{key}: {value / num_batches}")

	def save(self):

		model_file = os.path.join(self.args.logdir, f'model_{self.cur_epoch:04d}.pth')
		optimizer_file = os.path.join(self.args.logdir, f'optimizer_{self.cur_epoch:04d}.pth')
		scaler_file = os.path.join(self.args.logdir, f'scaler_{self.cur_epoch:04d}.pth')
		scheduler_file = os.path.join(self.args.logdir, f'scheduler_{self.cur_epoch:04d}.pth')

		# The parallel weights are named differently with the module.
		# We remove that, so that we can load the model with the same code.
		torch.save(self.model.state_dict(), model_file)

		torch.save(self.optimizer.state_dict(), optimizer_file)
		torch.save(self.scaler.state_dict(), scaler_file)
		torch.save(self.scheduler.state_dict(), scheduler_file)

		# Remove last epochs files to avoid accumulating storage
		if self.cur_epoch > 0:
			last_model_file = os.path.join(self.args.logdir, f'model_{self.cur_epoch - 1:04d}.pth')
			last_optimizer_file = os.path.join(self.args.logdir, f'optimizer_{self.cur_epoch - 1:04d}.pth')
			last_scaler_file = os.path.join(self.args.logdir, f'scaler_{self.cur_epoch - 1:04d}.pth')
			last_scheduler_file = os.path.join(self.args.logdir, f'scheduler_{self.cur_epoch - 1:04d}.pth')
			if os.path.isfile(last_model_file):
				os.remove(last_model_file)
			if os.path.isfile(last_optimizer_file):
				os.remove(last_optimizer_file)
			if os.path.isfile(last_scaler_file):
				os.remove(last_scaler_file)
			if os.path.isfile(last_scheduler_file):
				os.remove(last_scheduler_file)


# We need to seed the workers individually otherwise random processes in the
# dataloader return the same values across workers!
def seed_worker(worker_id):  # pylint: disable=locally-disabled, unused-argument
	# Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
	worker_seed = (torch.initial_seed()) % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


if __name__ == '__main__':
	# Select how the threads in the data loader are spawned
	available_start_methods = mp.get_all_start_methods()
	if 'fork' in available_start_methods:
		mp.set_start_method('fork')
	# Available on all OS.
	elif 'spawn' in available_start_methods:
		mp.set_start_method('spawn')
	elif 'forkserver' in available_start_methods:
		mp.set_start_method('forkserver')
	print('Start method of multiprocessing:', mp.get_start_method())

	main()
