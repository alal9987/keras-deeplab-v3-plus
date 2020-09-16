W = 142
H = 80
n_classes = 2
batch_size = 32
epochs = 100
workers = 4
lr = 1e-3
use_wandb = True
DATA_PATH = 'dataset'
CONFIG_PATH = 'config.yaml'

# checkpoint settings
monitor = 'loss'
mode = 'max'


colors = [
    [0, 0, 0], # background
    [255, 0, 255] # crosswalk
#    [0, 255, 0] # sidewalk + guide_block
]
n_classes = 3
colors = [
    [0, 0, 0], # background
    [255, 0, 255], # crosswalk
    [0, 255, 0] # sidewalk + guide_block
]
