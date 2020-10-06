# W = 142
W = 240
H = 320
batch_size = 16
epochs = 100
workers = 4
lr = 1e-4
use_wandb = True
gpu_limit = 0.6
DATA_PATH = 'surface_320_240'
CONFIG_PATH = 'config.yaml'
TEST_MODE = False

# checkpoint settings
monitor = 'loss'
mode = 'max'

n_classes = 2
colors = [
    [0, 0, 0], # background
    [255, 0, 255], # crosswalk
#    [0, 255, 0] # sidewalk + guide_block
]
