import numpy as np
import settings
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.compat.v1 import to_int32

_IS_TF_2 = True


def generate_matrix(gt_image, pre_image):
    mask = (gt_image >= 0) & (gt_image < settings.n_classes)
    label = settings.n_classes * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=settings.n_classes**2)
    confusion_matrix = count.reshape(settings.n_classes, settings.n_classes)
    return confusion_matrix


def acc(y_true, y_pred):
    confusion_matrix = generate_matrix(y_true, y_pred)
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return acc


def acc_class(y_true, y_pred):
    confusion_matrix = generate_matrix(y_true, y_pred)
    acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    acc = np.nanmean(acc)
    return acc


def Intersection_over_Union(y_true, y_pred):
    confusion_matrix = generate_matrix(y_true, y_pred)
    IoU = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
        np.diag(confusion_matrix))
    return IoU


def mIoU(y_true, y_pred):
    IoU = Intersection_over_Union(y_true, y_pred)
    MIoU = np.nanmean(IoU)
    return MIoU


def Jaccard(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:,:,0], i)
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


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)