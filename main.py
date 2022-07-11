import argparse
import os

import tensorflow as tf

from dataloader.data_process import get_difference_img, dense_gaussian_filtering, threshold_otsu
from configs import get_config
from dataloader.data_loader import data_load, training_data_generator
from decorators import image_to_tensorboard
from network import ImageTranslationNetwork, Graph_Attention_Union
from tqdm import trange


class BGAAE:

    def __init__(self, translation_spec, **configs):

        """
            Build attributes and method.
        """

        self.dcl_alpha = configs.get("dcl_alpha", 10)
        self.sem_beta = configs.get("sem_beta", 1)
        self.l2_lambda = configs.get("l2_lambda", 1e-6)
        self.lr = configs.get("learning_rate", 1e-4)
        self.crop = configs.get("crop", 0.1)
        self._encoder = ImageTranslationNetwork(
            **translation_spec["encoder"], name="encoder", l2_lambda=self.l2_lambda
        )
        self._decoder_x = ImageTranslationNetwork(
            **translation_spec["decoder_x"], name="decoder_x", l2_lambda=self.l2_lambda
        )
        self._decoder_y = ImageTranslationNetwork(
            **translation_spec["decoder_y"], name="decoder_y", l2_lambda=self.l2_lambda
        )
        self._gam = Graph_Attention_Union(**translation_spec["gam"])

        self.evaluation_frequency = tf.constant(
            configs.get("evaluation_frequency", 1), dtype=tf.int64
        )

        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        logdir = configs.get("logdir", None)

        if logdir is not None:
            self.log_path = logdir
            self.tb_writer = tf.summary.create_file_writer(self.log_path)
            self._img_dir = tf.constant(os.path.join(self.log_path, "images"))
        self._save_images = tf.Variable(False, trainable=False)


    @image_to_tensorboard()
    def encoder(self, inputs, training=False):
        return self._encoder(inputs, training)

    @image_to_tensorboard()
    def decoder_x(self, inputs, training=False):
        return self._decoder_x(inputs, training)

    @image_to_tensorboard()
    def decoder_y(self, inputs, training=False):
        return self._decoder_y(inputs, training)

    @image_to_tensorboard("difference_image")
    def get_diff(self, x, y):
        difference_img = self([x, y])
        return dense_gaussian_filtering(x, y, difference_img)

    @image_to_tensorboard("change_map")
    def get_changemap(self, di_img):
        tmp = tf.cast(di_img * 255, tf.int32)
        threshold = threshold_otsu(tmp) / 255
        return di_img >= threshold


    def __call__(self, inputs, training=False):
        x, y = inputs
        tf.debugging.Assert(tf.rank(x) == 4, [x.shape])
        tf.debugging.Assert(tf.rank(y) == 4, [y.shape])
        if training:
            x_code, y_code = self._encoder(x, training), self._encoder(y, training)
            x_tilde, y_tilde = self._decoder_x(x_code, training), self._decoder_y(y_code, training)
            x_hat, y_hat = self._decoder_x(y_code, training), self._decoder_y(x_code, training)
            y_sem, x_sem = self._encoder(x_hat, training), self._encoder(y_hat, training)
            x_gam = self._gam(tf.image.central_crop(x_code, self.crop), tf.image.central_crop(y_code, self.crop))
            y_gam = self._gam(tf.image.central_crop(y_code, self.crop), tf.image.central_crop(x_code, self.crop))

            keep = [x_code, y_code, x_tilde, y_tilde, x_hat, y_hat, x_sem, y_sem, x_gam, y_gam]
        else:
            x_code, y_code = self.encoder(x, name="x_code"), self.encoder(y, name="y_code")
            x_hat, y_hat = self.decoder_x(y_code, name="x_hat"), self.decoder_y(x_code, name="y_hat")
            keep = get_difference_img(x_hat, y_hat)

        return keep

    def train(self, x, y):

        with tf.GradientTape() as tape:
            x_code, y_code, x_tilde, y_tilde, x_hat, y_hat, x_sem, y_sem, x_gam, y_gam = self([x, y], training=True)
            recon_x_loss = self.loss_object(x, x_tilde)
            recon_y_loss = self.loss_object(y, y_tilde)

            dcl_x_loss = self.dcl_alpha * self.loss_object(tf.image.central_crop(x_hat, self.crop), x_gam)
            dcl_y_loss = self.dcl_alpha * self.loss_object(tf.image.central_crop(y_hat, self.crop), y_gam)

            sem_x_loss = self.sem_beta * self.loss_object(x_code, x_sem)
            sem_y_loss = self.sem_beta * self.loss_object(y_code, y_sem)

            total_loss = (
                recon_x_loss
                + recon_y_loss
                + dcl_x_loss
                + dcl_y_loss
                + sem_x_loss
                + sem_y_loss
            )
            target_all = (
                self._encoder.trainable_variables
                + self._decoder_x.trainable_variables
                + self._decoder_y.trainable_variables
            )

            gradients_all = tape.gradient(total_loss, target_all)
            self.optimizer.apply_gradients(zip(gradients_all, target_all))


    def evaluate(self, eva_dataset, filter):
        x, y, gt = eva_dataset
        self._save_images.assign(True)
        if filter:
            self.get_changemap(self.get_diff(x, y))
        else:
            self.get_changemap(self(x, y))

def test(DATASET = "Beijing", CONFIG = None):

    """
        Set up the network and start training.

    """

    print(f"Loading {DATASET} dataset !!!")
    x, y, EVALUATE, Channel = data_load(DATASET)
    C_CODE = CONFIG["C_CODE"]
    filters = CONFIG["filters"]
    TRANSLATION_SPEC = {
        "encoder": {"input_chs": Channel, "filter_spec": [filters, filters, filters, C_CODE]},
        "decoder_x": {"input_chs": C_CODE, "filter_spec": [filters, filters, filters, Channel]},
        "decoder_y": {"input_chs": C_CODE, "filter_spec": [filters, filters, filters, Channel]},
        "gam": {"input_chs": C_CODE, "output_chs": Channel}
    }
    print("Change Detector Init !!!")
    cd = BGAAE(TRANSLATION_SPEC, **CONFIG)
    print("Training !!!")
    for i in trange(CONFIG["epoch"]):
        tr_gen, dtypes, shapes = training_data_generator(
            x[0], y[0]
        )
        TRAIN = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
        TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        for _, batch in zip(range(CONFIG["batches"]), TRAIN.batch(CONFIG["batches"])):
            cd.train(*batch)
        for eva_data in EVALUATE.batch(CONFIG["batches"]):
            cd.evaluate(eva_data, CONFIG["filter"])

    print("finish !!!")
    del cd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset', type=str, default="Beijing")
    args = parser.parse_args()
    DATASET = args.Dataset
    CONFIG = get_config(DATASET)
    test(DATASET=DATASET, CONFIG=CONFIG)
