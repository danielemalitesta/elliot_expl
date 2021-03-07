"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras


class ACF_model(keras.Model):
    def __init__(self, factors=200,
                 layers_component=(64, 1),
                 layers_item=(64, 1),
                 learning_rate=0.001,
                 l_w=0,
                 emb_image=None,
                 sp_i_train={},
                 num_users=100,
                 num_items=100,
                 name="ACF",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._factors = factors
        self.l_w = l_w
        self.emb_image = emb_image
        self.feature_shape = self.emb_image.shape[1:]
        self._learning_rate = learning_rate
        self._num_items = num_items
        self._num_users = num_users

        self._sp_i_train = sp_i_train

        self.layers_component = layers_component
        self.layers_item = layers_item

        self.initializer = tf.initializers.RandomNormal(stddev=0.01)
        self.initializer_attentive = tf.initializers.GlorotUniform()

        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)
        self.Pi = tf.Variable(
            self.initializer(shape=[self._num_items, self._factors]),
            name='Tu', dtype=tf.float32)
        self.Fi = tf.Variable(
            self.emb_image, dtype=tf.float32, trainable=False)

        self.component_weights, self.item_weights = self._build_attention_weights()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

    def _build_attention_weights(self):
        component_dict = dict()
        items_dict = dict()

        for c in range(len(self.layers_component)):
            # the inner layer has all components
            if c == 0:
                component_dict['W_{}_u'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self._factors, self.layers_component[c]]),
                    name='W_{}_u'.format(c),
                    dtype=tf.float32
                )
                component_dict['W_{}_i'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.feature_shape[-1], self.layers_component[c]]),
                    name='W_{}_i'.format(c),
                    dtype=tf.float32
                )
                component_dict['b_{}'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_component[c]]),
                    name='b_{}'.format(c),
                    dtype=tf.float32
                )
            else:
                component_dict['W_{}'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_component[c - 1], self.layers_component[c]]),
                    name='W_{}_u'.format(c),
                    dtype=tf.float32
                )
                component_dict['b_{}'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_component[c]]),
                    name='b_{}'.format(c),
                    dtype=tf.float32
                )

        for i in range(len(self.layers_item)):
            # the inner layer has all components
            if i == 0:
                items_dict['W_{}_u'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self._factors, self.layers_item[i]]),
                    name='W_{}_u'.format(i),
                    dtype=tf.float32
                )
                items_dict['W_{}_iv'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self._factors, self.layers_item[i]]),
                    name='W_{}_iv'.format(i),
                    dtype=tf.float32
                )
                items_dict['W_{}_ip'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self._factors, self.layers_item[i]]),
                    name='W_{}_ip'.format(i),
                    dtype=tf.float32
                )
                items_dict['W_{}_ix'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.feature_shape[-1], self.layers_item[i]]),
                    name='W_{}_ix'.format(i),
                    dtype=tf.float32
                )
                items_dict['b_{}'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_item[i]]),
                    name='b_{}'.format(i),
                    dtype=tf.float32
                )
            else:
                items_dict['W_{}'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_item[i - 1], self.layers_item[i]]),
                    name='W_{}_u'.format(i),
                    dtype=tf.float32
                )
                items_dict['b_{}'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_item[i]]),
                    name='b_{}'.format(i),
                    dtype=tf.float32
                )
        return component_dict, items_dict

    @tf.function
    def _calculate_beta_alpha(self, g_u, g_i, p_i, f_i):
        # calculate beta
        b_i_l = tf.expand_dims(tf.expand_dims(tf.matmul(g_u, self.component_weights['W_{}_u'.format(0)]), 1), 1) + \
                tf.ragged.map_flat_values(tf.matmul, f_i, self.component_weights['W_{}_i'.format(0)]) + \
                self.component_weights['b_{}'.format(0)]
        b_i_l = tf.ragged.map_flat_values(tf.nn.relu, b_i_l)
        for c in range(1, len(self.layers_component)):
            b_i_l = tf.ragged.map_flat_values(tf.matmul, b_i_l, self.component_weights['W_{}'.format(c)]) + \
                    self.component_weights['b_{}'.format(c)]

        b_i_l = tf.ragged.map_flat_values(tf.nn.softmax, b_i_l, 1)
        all_x_l = tf.reduce_sum(tf.multiply(b_i_l, f_i), axis=2)

        # calculate alpha
        a_i_l = tf.expand_dims(tf.matmul(g_u, self.item_weights['W_{}_u'.format(0)]), 1) + \
                tf.ragged.map_flat_values(tf.matmul, g_i, self.item_weights['W_{}_iv'.format(0)]) + \
                tf.ragged.map_flat_values(tf.matmul, p_i, self.item_weights['W_{}_ip'.format(0)]) + \
                tf.ragged.map_flat_values(tf.matmul, all_x_l, self.item_weights['W_{}_ix'.format(0)]) + \
                self.item_weights['b_{}'.format(0)]
        a_i_l = tf.ragged.map_flat_values(tf.nn.relu, a_i_l)
        for c in range(1, len(self.layers_item)):
            a_i_l = tf.ragged.map_flat_values(tf.matmul, a_i_l, self.item_weights['W_{}'.format(c)]) + \
                    self.item_weights['b_{}'.format(c)]
        a_i_l = tf.ragged.map_flat_values(tf.nn.softmax, a_i_l, 1)

        all_a_i_l = tf.reduce_sum(tf.multiply(a_i_l, p_i), axis=1)
        g_u_p = g_u + all_a_i_l

        return g_u_p

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item, user_pos = inputs
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        p_i = tf.squeeze(tf.nn.embedding_lookup(self.Pi, item))

        gamma_i_u_pos = tf.nn.embedding_lookup(self.Gi, tf.where(user_pos > 0)[:, 2])
        p_i_u_pos = tf.nn.embedding_lookup(self.Pi, tf.where(user_pos > 0)[:, 2])
        f_u_i_pos = tf.nn.embedding_lookup(self.Fi, tf.where(user_pos > 0)[:, 2])
        user_pos_indices = tf.where(user_pos > 0)[:, 0]
        gamma_i_u_pos = tf.RaggedTensor.from_value_rowids(gamma_i_u_pos, user_pos_indices)
        p_i_u_pos = tf.RaggedTensor.from_value_rowids(p_i_u_pos, user_pos_indices)
        f_u_i_pos = tf.RaggedTensor.from_value_rowids(f_u_i_pos, user_pos_indices)

        gamma_u_p = self._calculate_beta_alpha(gamma_u, gamma_i_u_pos, p_i_u_pos, f_u_i_pos)
        xui = tf.reduce_sum(gamma_u_p * gamma_i, 1)

        return xui, gamma_u, gamma_i, p_i

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as t:
            user, pos, neg, user_pos = batch
            xu_pos, gamma_u, gamma_pos, p_i_pos = self((user, pos, user_pos), training=True)
            xu_neg, _, gamma_neg, p_i_neg = self((user, neg, user_pos), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(p_i_pos),
                                                 tf.nn.l2_loss(p_i_neg),
                                                 *[tf.nn.l2_loss(value)
                                                   for _, value in self.component_weights.items()],
                                                 *[tf.nn.l2_loss(value)
                                                   for _, value in self.item_weights.items()]])
            # Loss to be optimized
            loss += reg_loss

        params = [self.Gu,
                  self.Gi,
                  self.Pi,
                  *[value for _, value in self.component_weights.items()],
                  *[value for _, value in self.item_weights.items()]]

        grads = t.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return loss

    @tf.function
    def predict(self, start, stop):
        user_pos = self._sp_i_train[start:stop]
        gamma_i_u_pos = tf.nn.embedding_lookup(self.Gi, tf.where(user_pos > 0)[:, 1])
        p_i_u_pos = tf.nn.embedding_lookup(self.Pi, tf.where(user_pos > 0)[:, 1])
        f_u_i_pos = tf.nn.embedding_lookup(self.Fi, tf.where(user_pos > 0)[:, 1])
        user_pos_indices = tf.where(user_pos > 0)[:, 0]
        gamma_i_u_pos = tf.RaggedTensor.from_value_rowids(gamma_i_u_pos, user_pos_indices)
        p_i_u_pos = tf.RaggedTensor.from_value_rowids(p_i_u_pos, user_pos_indices)
        f_u_i_pos = tf.RaggedTensor.from_value_rowids(f_u_i_pos, user_pos_indices)

        gamma_u_p = self._calculate_beta_alpha(self.Gu[start:stop], gamma_i_u_pos, p_i_u_pos, f_u_i_pos)

        return tf.matmul(gamma_u_p, self.Gi, transpose_b=True)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def get_config(self):
        raise NotImplementedError
