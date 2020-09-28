import os
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
from metrics import Jaccard, MIOU


def convert_to_tflite(model_path: str, new_path: str) -> str:
    model = load_model(model_path, custom_objects={
        'Jaccard': Jaccard, 'tf': tf, 'MIOU': MIOU, 'relu6': tf.nn.relu6,
    })

    converted = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    open(new_path, 'wb').write(converted)
    return new_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()
    new_path = args.model_path[:-3] + '.tflite'
    convert_to_tflite(args.model_path, new_path)


if __name__ == '__main__':
    main()

