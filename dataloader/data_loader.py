import os

from dataloader.data_process import clip

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from itertools import count

def BCDD():
    t1 = plt.imread("./BCDD/prechange.bmp")
    t2 = plt.imread("./BCDD/postchange.bmp")
    change_mask = plt.imread("../datasets/BCDD/gt.bmp")
    t1, t2, change_mask = t1[:, :, :], t2[:, :, :], change_mask[:, :]
    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)
    t1, t2 = clip(t1), clip(t2)

    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    t1, t2 = tf.expand_dims(t1, axis=0), tf.expand_dims(t2, axis=0)
    return t1, t2, change_mask

def Beijing():
    t1 = plt.imread("./Beijing/prechange.jpg")
    t2 = plt.imread("./Beijing/postchange.jpg")
    change_mask = plt.imread("./Beijing/gt.jpg")
    t1, t2, change_mask = t1[:, :, :], t2[:, :, :], change_mask[:, :]
    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)

    t1, t2 = clip(t1), clip(t2)

    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def NJUST():
    t1 = plt.imread("./NJUST/prechange.bmp")
    t2 = plt.imread("./NJUST/postchange.bmp")
    change_mask = plt.imread("./NJUST/gt.bmp")
    t1, t2, change_mask = t1[:, :, :], t2[:, :, :], change_mask[:, :, 0]
    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)
    t1, t2 = clip(t1), clip(t2)

    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    t1, t2 = tf.expand_dims(t1, axis=0), tf.expand_dims(t2, axis=0)
    return t1, t2, change_mask

def Szada():
    t1 = plt.imread("./Szada/prechange.bmp")
    t2 = plt.imread("./Szada/postchange.bmp")
    change_mask = plt.imread("./Szada/gt.bmp")
    t1, t2, change_mask = t1[:, :, :], t2[:, :, :], change_mask[:, :]
    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)
    t1, t2 = clip(t1), clip(t2)

    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    t1, t2 = tf.expand_dims(t1, axis=0), tf.expand_dims(t2, axis=0)
    return t1, t2, change_mask

DATASETS = {
    "Beijing": Beijing,
    "NJUST": NJUST,
    "Szada": Szada,
    "BCDD": BCDD
}

def data_load(name):
    x_im, y_im, target_cm = DATASETS[name]()
    dataset = [x_im, y_im, target_cm]

    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    x, y = dataset[0], dataset[1]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    channel = x_im.shape[-1]
    return x, y, evaluation_data, channel

def training_data_generator(x, y):

    w, h, c = x.shape[0], x.shape[1], x.shape[2]

    chs = 2 * c
    x_chs = slice(0, c, 1)
    y_chs = slice(c, chs, 1)
    data = tf.concat([x, y], axis=-1)

    def gen():
        for _ in count():
            yield data[:, :, x_chs], data[:, :, y_chs]

    dtypes = (tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([w, h, c]),
        tf.TensorShape([w, h, c]),
    )

    return gen, dtypes, shapes

if __name__=="__main__":
    for dataset in DATASETS:
        print(data_load(dataset))
