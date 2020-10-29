# W = 142
#W = 240
#H = 320
W = 60
H = 80
batch_size = 16
epochs = 50
workers = 4
lr = 1e-3
use_wandb = True
gpu_limit = 0.9
DATA_PATH = 'surface_80_60'
# DATA_PATH = 'surface_320_240'
CONFIG_PATH = 'config.yaml'

# checkpoint settings
monitor = 'miou'
mode = 'max'

n_classes = 2
colors = [
    [0, 0, 0], # background
    [255, 0, 255], # crosswalk
    [0, 255, 0] # sidewalk + guide_block
]
