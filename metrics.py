import numpy as np
import settings
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.compat.v1 import to_int32
tf.config.experimental_run_functions_eagerly(True)

_IS_TF_2 = True


def Jaccard(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    #print('***', K.shape(y_true))
    #print(pred_pixels[0,0,0])
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:,:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = to_int32(true_labels & pred_labels)
        union = to_int32(true_labels | pred_labels)
        legal_batches = K.sum(to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        if _IS_TF_2:
            iou.append(K.mean(ious[legal_batches]))
        else:
            iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou) if _IS_TF_2 else ~tf.debugging.is_nan(iou)
    iou = iou[legal_labels] if _IS_TF_2 else tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)