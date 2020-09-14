W = 142
H = 80
n_classes = 2
batch_size = 2
epochs = 2
workers = 2
use_wandb = False
DATA_PATH = 'D:/surface_80/'
CONFIG_PATH = 'config.yaml'

# checkpoint settings
monitor = 'loss'
mode = 'max'


colors = [
    [0, 0, 0], # background
    [255, 0, 255] # crosswalk
#    [0, 255, 0] # sidewalk + guide_block
]