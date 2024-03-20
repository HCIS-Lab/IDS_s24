"""
Config class that contains all the hyperparameters needed to build any model.
"""

import os
import re


class GlobalConfig:
  """
  Config class that contains all the hyperparameters needed to build any model.
  """

  def __init__(self):
    """ base architecture configurations """
    # Set to one for backwards compatibility. Released dataset was collected with 5
    # -----------------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------------
    self.data_save_freq = 5
    self.carla_fps = 20  # Simulator Frames per second
    self.seq_len = 1  # input timesteps
    # Number of initial frames to skip during data loading
    self.skip_first = int(2.5 * self.carla_fps) // self.data_save_freq
    self.pred_len = int(2.0 * self.carla_fps) // self.data_save_freq  # number of future waypoints predicted
    # Number of LiDAR hits a bounding box needs for it to be a valid label
    self.num_lidar_hits_for_detection = 7

    # How many pixels make up 1 meter.
    # 1 / pixels_per_meter = size of pixel in meters
    self.pixels_per_meter = 4.0
    # Max and minimum LiDAR ranges used for voxelization
    self.min_x = -32
    self.max_x = 32
    self.min_y = -32
    self.max_y = 32
    self.min_z = -4
    self.max_z = 4
    self.min_z_projection = -10
    self.max_z_projection = 14

    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    self.local_rank = -999
    self.id = '___'  # Unique experiment identifier.
    self.epochs = 41  # Number of epochs to train
    self.batch_size = 12  # Batch size used during training
    self.logdir = './'  # Directory to log data to.
    self.load_file = None  # File to continue training from
    # When to reduce the learning rate for the first and second  time
    self.schedule_reduce_epoch_01 = 30
    self.schedule_reduce_epoch_02 = 40
    self.parallel_training = 1  # Whether training was done in parallel
    self.val_every = 2  # Validation frequency in epochs
    self.sync_batch_norm = 0  # Whether batch norm was synchronized between GPUs
    # Whether zero_redundancy_optimizer was used during training
    self.zero_redundancy_optimizer = 1
    self.use_disk_cache = 0  # Whether disc cache was used during training
    self.detect_boxes = 0  # Whether to use the bounding box auxiliary task
    self.train_sampling_rate = 1  # We train on every n th sample on the route
    # Number of route points we use for prediction in TF or input in model
    self.num_route_points = 20
    self.augment_percentage = 0.5  # Probability of the augmented sample being used.
    self.learn_origin = 1  # Whether to learn the origin of the waypoints or use 0 / 0
    self.augment = 1  # Whether to use rotation and translation augmentation
    # If this is true we convert the batch norms, to synced bach norms.
    self.sync_batch_norm = False
    # At which interval to save debug files to disk during training
    self.train_debug_save_freq = 1
    # Label smoothing applied to the cross entropy losses
    self.label_smoothing_alpha = 0.1
    # Optimization
    self.lr = 0.0003  # learning rate
    # Learning rate decay, applied when using multi-step scheduler
    self.multi_step_lr_decay = 0.1
    # Whether to use a cosine schedule instead of the linear one.
    self.use_cosine_schedule = False
    # Epoch of the first restart
    self.cosine_t0 = 1
    # Multiplier applied to t0 after every restart
    self.cosine_t_mult = 2
    
    self.root_dir = './'
    self.train_towns = []
    self.val_towns = []
    self.train_data = []
    self.val_data = []
    self.num_repetitions = 1  # How many repetitions of the dataset we train with.
    self.continue_epoch = False  # Whether to continue the training from the loaded epoch or from 0.

    self.smooth_route = True  # Whether to smooth the route points with a spline.
    self.ignore_index = -999  # Index to ignore for future bounding box prediction task.
    self.use_speed_weights = True  # Whether to weight target speed classes
    self.use_optim_groups = False  # Whether to use optimizer groups to exclude some parameters from weight decay
    self.weight_decay = 0.01  # Weight decay coefficient used during training
    self.use_model_labels = False  # Whether to use the relabeling from model or the original labels
    self.use_label_smoothing = False  # Whether to use label smoothing in the classification losses


    self.max_throttle = 0.75  # upper limit on throttle signal value in dataset
    self.brake_speed = 0.4  # desired speed below which brake is triggered
    # ratio of speed to desired speed at which brake is triggered
    self.brake_ratio = 1.1
    self.clip_delta = 0.25  # maximum change in speed input to logitudinal controller
    self.clip_throttle = 0.75  # Maximum throttle allowed by the controller

    # Whether the model in and outputs will be visualized and saved into SAVE_PATH
    self.debug = True

    # -----------------------------------------------------------------------------
    # Logger
    # -----------------------------------------------------------------------------
    self.logging_freq = 10  # Log every 10 th frame
    self.logger_region_of_interest = 30.0  # Meters around the car that will be logged.
    self.route_points = 10  # Number of route points to render in logger
    # Minimum distance to the next waypoint in the logger
    self.log_route_planner_min_distance = 4.0


    # -----------------------------------------------------------------------------
    # Agent file
    # -----------------------------------------------------------------------------
    self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
    # Iou threshold used for non-maximum suppression on the Bounding Box
    # predictions for the ensembles
    self.iou_treshold_nms = 0.2
    self.route_planner_min_distance = 7.5
    self.route_planner_max_distance = 50.0
    # Min distance to the waypoint in the dense rout that the expert is trying to follow
    self.dense_route_planner_min_distance = 3.5
    self.dense_route_planner_max_distance = 50.0
    self.action_repeat = 1  # Number of times we repeat the networks action.
    # Number of frames after which the creep controller starts triggering. 1100 is larger than wait time at red light.
    self.stuck_threshold = 1100 / self.action_repeat
    self.creep_duration = 20 / self.action_repeat  # Number of frames we will creep forward
    self.creep_throttle = 0.4
    # CARLA needs some time to initialize in which the cars actions are blocked.
    # Number tuned empirically
    self.inital_frames_delay = 2.0 / self.carla_frame_rate

    # Extent of the ego vehicles bounding box
    self.ego_extent_x = 2.4508416652679443
    self.ego_extent_y = 1.0641621351242065
    self.ego_extent_z = 0.7553732395172119
    
    self.max_num_bbs = 30

    # Size of the safety box
    self.safety_box_z_min = 0.5
    self.safety_box_z_max = 1.5

    self.safety_box_y_min = -self.ego_extent_y * 0.8
    self.safety_box_y_max = self.ego_extent_y * 0.8

    self.safety_box_x_min = self.ego_extent_x
    self.safety_box_x_max = self.ego_extent_x + 2.5

    # Probability 0 - 1. If the confidence in the brake action is higher than this
    # value brake is chosen as the action.
    self.brake_uncertainty_threshold = 0.5
    self.checkpoint_buffer_len = 10  # Number of time steps that we use for route consistency

    # -----------------------------------------------------------------------------
    # model
    # -----------------------------------------------------------------------------
    self.model_precision_pos = 8  # 8: 0.25 meters
    self.model_precision_angle = 7  # 7: 2.8125 degrees
    self.model_precision_speed = 4  # 4: 1,875 km/h
    self.model_precision_brake = 1  # 1: true, false
    self.model_object_types = 2  # vehicle, pedestrian
    self.model_num_attributes = 7  # x,y, extent x, extent y,yaw,speed, brake, (class)
    # Options: prajjwal1/bert-tiny, prajjwal1/bert-mini, prajjwal1/bert-small, prajjwal1/bert-medium
    self.model_hf_checkpoint = 'prajjwal1/bert-medium'
    self.model_embd_pdrop = 0.1
    self.model_pretraining = None
    self.model_pretraining_path = None
    self.model_multitask = False
    self.max_speed_pred = 60.0  # Maximum speed we classify when forcasting cars.
    self.forecast_time = 2  # Number of seconds we forcast into the future

  def initialize(self, root_dir='', setting='all', **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

    self.root_dir = root_dir

    val_town = 'Town10HD'

    self.train_towns = os.listdir(self.root_dir)  # Scenario Folders
    self.val_towns = self.train_towns
    self.train_data, self.val_data = [], []
    for town in self.train_towns:
      root_files = os.listdir(os.path.join(self.root_dir, town))  # Town folders
      for file in root_files:
        # Only load as many repetitions as specified
        repetition = int(re.search('Repetition(\\d+)', file).group(1))
        if repetition >= self.num_repetitions:
          continue
        # We don't train on two towns and reserve them for validation
        if (file.find(val_town) != -1):
          continue
        if not os.path.isfile(os.path.join(self.root_dir, file)):
          self.train_data.append(os.path.join(self.root_dir, town, file))
    for town in self.val_towns:
      root_files = os.listdir(os.path.join(self.root_dir, town))
      for file in root_files:
        repetition = int(re.search('Repetition(\\d+)', file).group(1))
        if repetition >= self.num_repetitions:
          continue
        # Only use withheld towns for validation
        if (file.find(val_town) == -1):
          continue
        if not os.path.isfile(os.path.join(self.root_dir, file)):
          self.val_data.append(os.path.join(self.root_dir, town, file))