import logging
import os
from abc import ABC

import numpy as np
import tensorflow as tf
import concurrent.futures

from dataset.visual_loader_mixin import VisualLoader
from recommender.traditional.BPRMF import BPRMF

np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ACF(BPRMF, VisualLoader, ABC):

    def __init__(self, data, params):
        """
        Create a ACF instance.
        (see https://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-AttentiveCF.pdf
        for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        super(ACF, self).__init__(data, params)

        self.embed_k = self.params.embed_k
        self.learning_rate = self.params.lr

        self.layers_component = self.params.layers_component
        self.layers_item = self.params.layers_item

        self.process_visual_features(data)
        self.emb_image = self.emb_image.reshape([self.feature_shape[0],
                                                 self.feature_shape[1] * self.feature_shape[2],
                                                 self.feature_shape[3]])

        # Initialize Model Parameters
        self.Pi = tf.Variable(self.initializer(shape=[self.num_items, self.embed_k]), name='Pi', dtype=tf.float32)
        self.F = tf.Variable(
            self.emb_image,
            name='F', dtype=tf.float32, trainable=False)
        self.component_weights, self.item_weights = self.build_attention_weights()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def build_attention_weights(self):
        component = dict()
        items = dict()

        for c in range(len(self.layers_component)):
            # the inner layer has all components
            if c == 0:
                component['W_{}_u'.format(c)] = tf.Variable(
                    self.initializer(shape=[self.embed_k, self.layers_component[c]]),
                    name='W_{}_u'.format(c),
                    dtype=tf.float32
                )
                component['W_{}_i'.format(c)] = tf.Variable(
                    self.initializer(shape=[self.feature_shape[-1], self.layers_component[c]]),
                    name='W_{}_i'.format(c),
                    dtype=tf.float32
                )
                component['b_{}'.format(c)] = tf.Variable(
                    self.initializer(shape=[self.layers_component[c]]),
                    name='b_{}'.format(c),
                    dtype=tf.float32
                )
            else:
                component['W_{}'.format(c)] = tf.Variable(
                    self.initializer(shape=[self.layers_component[c], self.layers_component[c - 1]]),
                    name='W_{}_u'.format(c),
                    dtype=tf.float32
                )
                component['b_{}'.format(c)] = tf.Variable(
                    self.initializer(shape=[self.layers_component[c]]),
                    name='b_{}'.format(c),
                    dtype=tf.float32
                )

        for i in range(len(self.layers_item)):
            # the inner layer has all components
            if i == 0:
                items['W_{}_u'.format(i)] = tf.Variable(
                    self.initializer(shape=[self.embed_k, self.layers_item[i]]),
                    name='W_{}_u'.format(i),
                    dtype=tf.float32
                )
                items['W_{}_iv'.format(i)] = tf.Variable(
                    self.initializer(shape=[self.embed_k, self.layers_item[i]]),
                    name='W_{}_iv'.format(i),
                    dtype=tf.float32
                )
                items['W_{}_ip'.format(i)] = tf.Variable(
                    self.initializer(shape=[self.embed_k, self.layers_item[i]]),
                    name='W_{}_ip'.format(i),
                    dtype=tf.float32
                )
                items['W_{}_ix'.format(i)] = tf.Variable(
                    self.initializer(shape=[self.feature_shape[-1], self.layers_item[i]]),
                    name='W_{}_ix'.format(i),
                    dtype=tf.float32
                )
                items['b_{}'.format(i)] = tf.Variable(
                    self.initializer(shape=[self.layers_item[i]]),
                    name='b_{}'.format(i),
                    dtype=tf.float32
                )
            else:
                items['W_{}'.format(i)] = tf.Variable(
                    self.initializer(shape=[self.layers_item[i], self.layers_item[i - 1]]),
                    name='W_{}_u'.format(i),
                    dtype=tf.float32
                )
                items['b_{}'.format(i)] = tf.Variable(
                    self.initializer(shape=[self.layers_item[i]]),
                    name='b_{}'.format(i),
                    dtype=tf.float32
                )
        return component, items

    def calculate_beta_alpha(self, i_p):
        # calculate beta
        u, list_of_pos = i_p['u'], i_p['u_pos']
        g_u = tf.expand_dims(tf.nn.embedding_lookup(self.Gu, u), axis=1)

        f_i = tf.squeeze(tf.nn.embedding_lookup(self.F, list_of_pos))
        b_i_l = tf.squeeze(tf.matmul(self.component_weights['W_{}_u'.format(0)], g_u)) + \
                tf.tensordot(f_i, self.component_weights['W_{}_i'.format(0)], axes=[[2], [0]]) + \
                self.component_weights['b_{}'.format(0)]
        b_i_l = tf.nn.relu(b_i_l)
        for c in range(1, len(self.layers_component)):
            b_i_l = tf.tensordot(b_i_l, self.component_weights['W_{}'.format(c)], axes=[[2], [1]]) + \
                    self.component_weights['b_{}'.format(c)]

        b_i_l = tf.nn.softmax(tf.squeeze(b_i_l, -1), axis=1)
        all_x_l = tf.reduce_sum(tf.multiply(tf.expand_dims(b_i_l, axis=2), f_i), axis=1)

        # calculate alpha
        g_i = tf.nn.embedding_lookup(self.Gi, list_of_pos)
        p_i = tf.nn.embedding_lookup(self.Pi, list_of_pos)
        a_i_l = tf.squeeze(tf.matmul(self.item_weights['W_{}_u'.format(0)], g_u)) + \
                tf.matmul(g_i, self.item_weights['W_{}_iv'.format(0)]) + \
                tf.matmul(p_i, self.item_weights['W_{}_ip'.format(0)]) + \
                tf.matmul(all_x_l, self.item_weights['W_{}_ix'.format(0)]) + \
                self.item_weights['b_{}'.format(0)]
        a_i_l = tf.nn.relu(a_i_l)
        for c in range(1, len(self.layers_item)):
            a_i_l = tf.matmul(a_i_l, self.item_weights['W_{}'.format(c)], transpose_b=True) + \
                    self.item_weights['b_{}'.format(c)]
        a_i_l = tf.nn.softmax(tf.reshape(a_i_l, -1))

        all_a_i_l = tf.reduce_sum(tf.multiply(tf.expand_dims(a_i_l, axis=1), p_i), axis=0)
        g_u_p = tf.squeeze(g_u) + all_a_i_l

        return g_u_p.numpy()

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
        p_i = tf.squeeze(tf.nn.embedding_lookup(self.Pi, item))

        all_pos_u = [{'u': i, 'u_pos': self.data.train_list[i]} for i in user.numpy()[:, 0].tolist()]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            output = executor.map(self.calculate_beta_alpha, all_pos_u)

        gamma_u_p = tf.Variable(np.asarray(list(output)))

        xui = tf.reduce_sum(gamma_u_p * gamma_i, 1)
        return xui, gamma_u, gamma_i, p_i

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        all_pos_u = [{'u': i, 'u_pos': self.data.train_list[i]} for i, _ in enumerate(self.data.train_list)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            output = executor.map(self.calculate_beta_alpha, all_pos_u)

        Gu_p = tf.Variable(np.asarray(list(output)))
        return tf.matmul(Gu_p, self.Gi, transpose_b=True)

    def predict_all_validation(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        all_pos_u = [{'u': i, 'u_pos': self.data.train_list[i] + self.data.validation_list[i]}
                      for i, _ in enumerate(self.data.train_list)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            output = executor.map(self.calculate_beta_alpha, all_pos_u)

        Gu_p = tf.Variable(np.asarray(list(output)))
        return tf.matmul(Gu_p, self.Gi, transpose_b=True)

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
            xu_pos, gamma_u, gamma_pos, p_i_pos = \
                self(inputs=(user, pos), training=True)
            xu_neg, _, gamma_neg, p_i_neg = self(inputs=(user, neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(p_i_pos),
                                                 tf.nn.l2_loss(p_i_neg),
                                                 *[tf.nn.l2_loss(value) for _, value in self.component_weights.items()],
                                                 *[tf.nn.l2_loss(value) for _, value in self.item_weights.items()]]) * 2

            # Loss to be optimized
            loss += reg_loss

        params = [self.Gu,
                  self.Gi,
                  self.Pi,
                  *[value for _, value in self.component_weights.items()],
                  *[value for _, value in self.item_weights.items()]]

        grads = t.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return loss.numpy()
