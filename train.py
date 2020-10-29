import argparse, sys, os
from datetime import datetime
from model import Deeplabv3
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, \
    EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import MeanIoU, categorical_accuracy
from generator import BatchGenerator
import wandb, yaml
import numpy as np
from metrics import Jaccard, MIOU
from convert_to_tflite import convert_to_tflite
import settings


def limit_keras_gpu_usage(fraction: float):
    assert 0. < fraction <= 1.
    tf_config = tf1.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = fraction


def get_callbacks(model_path: str):
    tensor_board = TensorBoard(log_dir=os.path.join('.', 'trainings', exp_name),
                               write_graph=True, update_freq='epoch',
                               write_images=True, embeddings_freq=10)

    checkpoint = ModelCheckpoint(model_path, verbose=1,
                                 monitor = 'val_{}'.format(settings.monitor),
                                 save_best_only=True, mode=settings.mode)

    early_stopping = EarlyStopping(monitor = 'val_{}'.format(settings.monitor),
                                   patience=100, verbose=1, mode = settings.mode)
    csv_logger = CSVLogger(
        os.path.join('.', 'trainings', exp_name, 'training_log.csv'))

    callbacks = [tensor_board, checkpoint, early_stopping, csv_logger]

    if settings.use_wandb:
        wb = wandb.keras.WandbCallback(monitor='val_{}'.format(settings.monitor),
                                       save_model=True)
        callbacks.append(wb)

    return callbacks


def main():
    # GPU
    limit_keras_gpu_usage(settings.gpu_limit)
    # configurations
    global exp_name
    exp_name = datetime.strftime(datetime.now(), '%y%m%d-%H%M%S')
    opt = {
        'width': settings.W,
        'height': settings.H, 
        'n_classes': settings.n_classes,
        'batch_size': settings.batch_size,
        'epochs': settings.epochs,
        'workers': settings.workers,
        'wandb': settings.use_wandb,
        'monitor': settings.monitor,
        'mode': settings.mode
    }

    if settings.use_wandb:
        wandb.init(project="seg_keras", name=exp_name, config=opt, #TODO: opt
                   sync_tensorboard=True)

    # Setup model directory
    if not os.path.exists("trainings"):
        os.makedirs("trainings")
    if not os.path.exists(os.path.join('.', 'trainings', exp_name)):
        os.makedirs(os.path.join('.', 'trainings', exp_name))

    config_file_dst = os.path.join('.', 'trainings', exp_name,
                                   os.path.basename(settings.CONFIG_PATH))
    with open(config_file_dst, 'w') as f:
        yaml.dump(opt, f, default_flow_style=False, default_style='')

    if settings.use_wandb:
        wandb.save(config_file_dst)

    # Build data generators
    train_gen = BatchGenerator(settings.DATA_PATH, settings.batch_size, mode='train',
                               n_classes=settings.n_classes)
    valid_gen = BatchGenerator(settings.DATA_PATH, settings.batch_size, mode='valid',
                               n_classes=settings.n_classes)

    # Initialize a model
    cce = categorical_crossentropy
    metrics = [MIOU(settings.n_classes), categorical_accuracy]

    model_path = os.path.join('.', 'trainings', exp_name, exp_name + '.h5')
    model = Deeplabv3(weights=None, input_shape=(settings.H, settings.W, 3),
                      classes=settings.n_classes, activation='softmax',
                      backbone='mobilenetv2')
    model.summary()
    model.compile(optimizer=Adam(lr=settings.lr, epsilon=1e-8, decay=1e-6),
                  sample_weight_mode = "temporal", loss = cce, metrics = metrics)
    #model.summary()

    # training
    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_gen),
                        epochs = settings.epochs, verbose=1, 
                        callbacks = get_callbacks(model_path), 
                        validation_data=valid_gen, 
                        validation_steps=len(valid_gen),
                        max_queue_size=10,
                        workers=settings.workers, use_multiprocessing=False)
    
    # save trflite model
    new_path = os.path.join('.', 'trainings', exp_name, exp_name + '.tflite')
    convert_to_tflite(model_path, new_path)
    if settings.use_wandb:
        wandb.save(os.path.join('trainings', exp_name))
    

if __name__ == '__main__':
    main()
