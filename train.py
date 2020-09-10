import argparse, sys, os
from datetime import datetime
from model import Deeplabv3
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, \
    EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from generator import BatchGenerator
import wandb, yaml
import numpy as np


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
monitor = 'Jaccard'
mode = 'max'

def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def get_callbacks(model_path: str):
    tensor_board = TensorBoard(log_dir=os.path.join('.', 'trainings', exp_name),
                               write_graph=True, update_freq='epoch',
                               write_images=True, embeddings_freq=10)

    checkpoint = ModelCheckpoint(model_path, verbose=1,
                                 monitor = 'val_{}'.format(monitor),
                                 save_best_only=True, mode=mode)

    early_stopping = EarlyStopping(monitor = 'val_{}'.format(monitor),
                                   patience=100, verbose=1, mode = mode)
    csv_logger = CSVLogger(
        os.path.join('.', 'trainings', exp_name, 'training_log.csv'))

    callbacks = [tensor_board, checkpoint, early_stopping, csv_logger]

    if use_wandb:
        wb = wandb.keras.WandbCallback(monitor='val_{}'.format(monitor),
                                       save_model=True)
        callbacks.append(wb)

    return callbacks


def main():
    # configurations
    global exp_name
    exp_name = datetime.strftime(datetime.now(), '%y%m%d-%H%M%S')
    opt = {
        'width': W,
        'height': H, 
        'n_classes': n_classes,
        'batch_size': batch_size,
        'epochs': epochs,
        'workers': workers,
        'wandb': use_wandb,
        'monitor': monitor,
        'mode': mode
    }

    if use_wandb:
        wandb.init(project="seg_keras", name=exp_name, config=opt, #TODO: opt
                   sync_tensorboard=True)

    # Setup model directory
    if not os.path.exists("trainings"):
        os.makedirs("trainings")
    if not os.path.exists(os.path.join('.', 'trainings', exp_name)):
        os.makedirs(os.path.join('.', 'trainings', exp_name))

    config_file_dst = os.path.join('.', 'trainings', exp_name,
                                   os.path.basename(CONFIG_PATH))
    with open(config_file_dst, 'w') as f:
        yaml.dump(opt, f, default_flow_style=False, default_style='')

    if use_wandb:
        wandb.save(config_file_dst)

    # Build data generators
    train_gen = BatchGenerator(DATA_PATH, batch_size, mode='train',
                               n_classes=2)
    valid_gen = BatchGenerator(DATA_PATH, batch_size, mode='valid',
                               n_classes=2)

    # Initialize a model
    losses = {}
    metrics = {}

    model_path = os.path.join('.', 'trainings', exp_name, exp_name + '.h5')
    model = Deeplabv3(weights=None, input_shape=(W, H, 3), classes=n_classes)
    model.compile(optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
                  loss = losses, metrics = metrics)
    #model.summary()

    print('***', len(train_gen), len(valid_gen))
    # training
    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_gen),
                        epochs = epochs, verbose=1, 
                        callbacks = get_callbacks(model_path), 
                        validation_data=valid_gen, 
                        validation_steps=len(valid_gen),
                        max_queue_size=10,
                        workers=workers, use_multiprocessing=True)
    
    # save trflite model
    new_path = os.path.join('.', 'trainings', exp_name, exp_name + '.tflite')
    convert_to_tflite(model_path, new_path)
    if use_wandb:
        wandb.save(os.path.join('trainings', exp_name))
    

if __name__ == '__main__':
    main()