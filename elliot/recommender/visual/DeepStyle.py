import logging
import os
from abc import ABC

import numpy as np
import tensorflow as tf
import random

from dataset.visual_loader_mixin import VisualLoader
from recommender.traditional.BPRMF import BPRMF

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepStyle(BPRMF, VisualLoader, ABC):

    def __init__(self, data, params):
        """
        Create a DeepStyle instance.
        (see https://dl.acm.org/doi/10.1145/3077136.3080658 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        super(DeepStyle, self).__init__(data, params)

        self.embed_k = self.params.embed_k
        self.learning_rate = self.params.lr
        self.l_e = self.params.l_e

        self.process_visual_features(data)

        # Initialize Model Parameters
        self.L = tf.Variable(
            self.initializer(shape=[self.num_items, self.embed_k]),
            name='L', dtype=tf.float32)
        self.F = tf.Variable(
            self.emb_image,
            name='F', dtype=tf.float32, trainable=False)
        self.E = tf.Variable(
            self.initializer(shape=[self.num_image_feature, self.embed_k]),
            name='E', dtype=tf.float32)  # (items, low_embedding_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def call(self, inputs, training=None, mask=None):
        """
        Generates prediction for passed users and items indices

        Args:
            inputs: user, item (batch)
            training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

        Returns:
            prediction and extracted model parameters
        """
        user, item = inputs
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))

        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        feature_i = tf.squeeze(tf.nn.embedding_lookup(self.F, item))

        l_i = tf.squeeze(tf.nn.embedding_lookup(self.L, item))

        xui = tf.reduce_sum(gamma_u * (tf.matmul(feature_i, self.E) - l_i + gamma_i), 1)

        return xui, gamma_u, gamma_i, feature_i, l_i

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        return tf.matmul(self.Gu, (tf.matmul(self.F, self.E) - self.L + self.Gi), transpose_b=True)

    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        user, pos, neg = batch
        with tf.GradientTape() as t:

            # Clean Inference
            xu_pos, gamma_u, gamma_pos, _, l_pos = \
                self(inputs=(user, pos), training=True)
            xu_neg, _, gamma_neg, _, l_neg = self(inputs=(user, neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(l_pos),
                                                 tf.nn.l2_loss(l_neg)])

            # Loss to be optimized
            loss += reg_loss

        grads = t.gradient(loss, [self.Gu, self.Gi, self.E, self.L])
        self.optimizer.apply_gradients(zip(grads, [self.Gu, self.Gi, self.E, self.L]))

        return loss.numpy()
