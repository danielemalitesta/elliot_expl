"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import numpy as np
import tensorflow as tf
from tensorflow import keras


class FashionExpl_model(keras.Model):
    def __init__(self, factors=200,
                 mlp_color=(64,),
                 mlp_att=(64,),
                 mlp_out=(64,),
                 dropout=0.2,
                 learning_rate=0.001,
                 l_w=0,
                 num_users=100,
                 num_items=100,
                 name="FashionExpl",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._factors = factors
        self._mlp_color = mlp_color
        self._mlp_att = mlp_att
        self._mlp_out = mlp_out
        self.l_w = l_w
        self._learning_rate = learning_rate
        self._num_items = num_items
        self._num_users = num_users
        self._dropout = dropout

        self.initializer = tf.initializers.RandomNormal(mean=0, stddev=0.01)
        self.initializer_attentive = tf.initializers.GlorotUniform()

        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)

        self.color_encoder = keras.Sequential()
        self.shape_encoder = keras.Sequential()
        self.attention_network = dict()
        self.mlp_output = keras.Sequential()

        self.create_color_weights()
        self.create_shape_weights()
        self.create_attention_weights()
        self.create_output_weights()

        self.optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate)

    def create_color_weights(self):
        self.color_encoder.add(keras.layers.Dropout(self._dropout))
        for units in self._mlp_color:
            self.color_encoder.add(keras.layers.Dense(units, activation='relu'))
        self.color_encoder.add(keras.layers.Dense(units=self._factors, use_bias=False))

    def create_shape_weights(self):
        self.shape_encoder.add(keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
        self.shape_encoder.add(keras.layers.MaxPool2D(padding='same'))
        self.shape_encoder.add(keras.layers.GlobalAveragePooling2D())
        self.shape_encoder.add(keras.layers.Dropout(rate=0.5))
        self.shape_encoder.add(keras.layers.Dense(units=self._factors, use_bias=False))

    def create_output_weights(self):
        self.mlp_output.add(keras.layers.Dropout(self._dropout))
        for units in self._mlp_output:
            self.mlp_output.add(keras.layers.Dense(units, activation='relu'))
        self.mlp_output.add(keras.layers.Dense(units=1, use_bias=False))

    def create_attention_weights(self):
        for layer in range(len(self._mlp_att)):
            if layer == 0:
                self.attention_network['W_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self._factors, self._mlp_att[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['b_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self._mlp_att[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )
            else:
                self.attention_network['W_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self._mlp_att[layer - 1], self.attention_layers[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['b_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self._mlp_att[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )

    @tf.function
    def propagate_attention(self, inputs):
        g_u, colors, edges, classes = inputs['gamma_u'], inputs['colors'], inputs['edges'], inputs['classes']
        all_a_i_l = None

        for layer in range(len(self.attention_layers)):
            if layer == 0:
                all_a_i_l = tf.tensordot(
                    tf.expand_dims(g_u, 1) * tf.concat([colors, edges, classes], axis=1),
                    self.attention_network['W_{}'.format(layer + 1)],
                    axes=[[2], [0]]
                ) + self.attention_network['b_{}'.format(layer + 1)]
                all_a_i_l = tf.nn.relu(all_a_i_l)
            else:
                all_a_i_l = tf.tensordot(
                    all_a_i_l,
                    self.attention_network['W_{}'.format(layer + 1)],
                    axes=[[2], [0]]
                ) + self.attention_network['b_{}'.format(layer + 1)]

        all_alpha = tf.nn.softmax(all_a_i_l, axis=1)
        return all_alpha

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item, shapes, colors, classes = inputs

        # USER
        # user collaborative profile
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))

        # ITEM
        # item collaborative profile
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        # item color features
        color_i = tf.expand_dims(self.color_encoder(colors), 1)
        # item edge features
        shapes_i = tf.expand_dims(self.shape_encoder(shapes), 1)

        # attention network
        attention_inputs = {
            'gamma_u': gamma_u,
            'colors': color_i,
            'edges': shapes_i,
            'classes': classes
        }
        all_attention = self.propagate_attention(attention_inputs)
        weighted_features = tf.reduce_sum(tf.multiply(
            all_attention,
            tf.concat([color_i, shapes_i, classes], axis=1)
        ), axis=1)

        # score prediction
        xui = tf.reduce_sum(gamma_u * weighted_features * gamma_i, axis=1)

        return xui, \
               gamma_u, \
               gamma_i, \
               color_i, \
               shapes_i, \
               classes, \
               all_attention

    @tf.function
    def train_step(self, batch):
        user, pos, shapes_pos, colors_pos, classes_pos, neg, shapes_neg, colors_neg, classes_neg = batch
        with tf.GradientTape() as t:
            # Clean Inference
            xu_pos, \
                gamma_u, \
                gamma_i_pos, \
                color_i_pos, \
                shape_i_pos, \
                class_i_pos, \
                attention_pos = self(inputs=(user, pos, shapes_pos, colors_pos, classes_pos), training=True)

            xu_neg, \
                _, \
                gamma_i_neg, \
                color_i_neg, \
                shape_i_neg, \
                class_i_neg, \
                attention_neg = self(inputs=(user, neg, shapes_neg, colors_neg, classes_neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_i_pos), tf.nn.l2_loss(gamma_i_neg),
                                                 tf.nn.l2_loss(color_i_pos), tf.nn.l2_loss(color_i_neg),
                                                 tf.nn.l2_loss(shape_i_pos), tf.nn.l2_loss(shape_i_neg),
                                                 tf.nn.l2_loss(class_i_pos), tf.nn.l2_loss(class_i_neg),
                                                 *[tf.nn.l2_loss(weight) for weight in
                                                   self.color_encoder.trainable_weights],
                                                 *[tf.nn.l2_loss(weight) for weight in
                                                   self.shape_encoder.trainable_weights],
                                                 *[tf.nn.l2_loss(value) for _, value in
                                                   self.attention_network.items()],
                                                 *[tf.nn.l2_loss(weight) for weight in
                                                   self.mlp_output.trainable_weights]])

            # Loss to be optimized
            loss += reg_loss

        params = [
            self.Gu,
            self.Gi,
            *self.color_encoder.trainable_weights,
            *self.shape_encoder.trainable_weights,
            *[value for _, value in self.attention_network.items()],
            *self.mlp_output.trainable_weights
        ]
        grads = t.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return loss

    @tf.function
    def predict_all_batch(self, step, next_image):
        all_predictions = []
        all_attentions = []
        reminder = self.num_items % step

        for u in range(self.num_users):
            current_predictions = []
            current_attentions = []
            gamma_u = tf.repeat(tf.expand_dims(self.Gu[u], 0), repeats=step, axis=0)
            for id_im, im, col, class_ in next_image:
                if self.data.num_items == id_im.numpy()[-1] + 1:
                    gamma_u = gamma_u[:reminder]

                gamma_i = tf.nn.embedding_lookup(self.Gi, id_im)
                edges = tf.expand_dims(self.edges_encoder(im), 1)
                colors = tf.expand_dims(self.color_encoder(col), 1)
                classes = tf.expand_dims(self.class_encoder(class_), 1)
                # attention network
                attention_inputs = {
                    'gamma_u': gamma_u,
                    'colors': colors,
                    'edges': edges,
                    'classes': classes
                }

                all_attention = self.propagate_attention(attention_inputs)
                weighted_features = tf.reduce_sum(tf.multiply(
                    all_attention,
                    tf.concat([colors, edges, classes], axis=1)
                ), axis=1)

                # score prediction
                xui = tf.reduce_sum(gamma_u * weighted_features * gamma_i, axis=1)
                current_predictions += xui.numpy().tolist()
                current_attentions += all_attention.numpy()[:, :, 0].tolist()
            all_predictions.append(current_predictions)
            all_attentions.append(current_attentions)

        return np.array(all_predictions), np.array(all_attentions)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def get_config(self):
        raise NotImplementedError
