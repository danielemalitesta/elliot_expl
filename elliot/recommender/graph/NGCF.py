from recommender.RecommenderModel import *
from recommender.Evaluator import Evaluator
from utils.write import save_obj
from utils.read import find_checkpoint
from config.configs import *

from abc import ABC
from copy import deepcopy
from time import time
import tensorflow as tf
import numpy as np
import random
import logging
import os

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


class NGCF(RecommenderModel, ABC):
    def __init__(self, data, params, *args, **kwargs):
        """
        Create a NGCF instance.
        (see https://arxiv.org/pdf/1905.08108.pdf for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w]: regularization,
                                      lr: learning rate}
        """
        super(NGCF, self).__init__(data, params, *args, **kwargs)

        self.plain_adj, self.norm_adj, self.mean_adj = data.create_adj_mat()

        self.embed_k = self.params.embed_k
        self.learning_rate = self.params.lr
        self.l_w = self.params.l_w
        self.weight_size = self.params.weight_size
        self.n_layers = len(self.weight_size)
        self.node_dropout = self.params.node_dropout
        self.message_dropout = self.params.message_dropout
        self.n_fold = self.params.n_fold

        # Generate a set of adjacency sub-matrix.
        if len(self.node_dropout):
            # node dropout.
            self.A_fold_hat = self._split_A_hat_node_dropout()
        else:
            self.A_fold_hat = self._split_A_hat()

        self.evaluator = Evaluator(self, data, params.k)

        # Initialize Model Parameters
        self._create_weights()

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def _create_weights(self):

        # Gu and Gi are different from the other recommenders, because in this case they are obtained as:
        # Gu = Gu_0 || Gu_1 || ... || Gu_L
        # Gi = Gi_0 || Gi_1 || ... || Gi_L
        self.Gu = tf.Variable(self.initializer([self.num_users + 1, self.embed_k * (self.n_layers + 1)]), name='Gu')
        self.Gi = tf.Variable(self.initializer([self.num_items + 1, self.embed_k * (self.n_layers + 1)]), name='Gi')

        self.Graph = dict()

        self.weight_size_list = [self.embed_k] + self.weight_size

        for k in range(self.n_layers):
            self.Graph['W_gc_%d' % k] = tf.Variable(
                self.initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            self.Graph['b_gc_%d' % k] = tf.Variable(
                self.initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            self.Graph['W_bi_%d' % k] = tf.Variable(
                self.initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            self.Graph['b_bi_%d' % k] = tf.Variable(
                self.initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

    def _propagate_embeddings(self):
        # extract gu_0 and gi_0 to begin embedding updating for L layers
        gu_0 = self.Gu[:, :self.embed_k]
        gi_0 = self.Gi[:, :self.embed_k]

        ego_embeddings = tf.concat([gu_0, gi_0], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(self.A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.Graph['W_gc_%d' % k]) + self.Graph['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.Graph['W_bi_%d' % k]) + self.Graph['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.message_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        gu, gi = tf.split(all_embeddings, [self.num_users + 1, self.num_items + 1], 0)
        self.Gu.assign(gu)
        self.Gi.assign(gi)

    def _split_A_hat(self):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items + 2) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.num_users + self.num_items + 2
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(_convert_sp_mat_to_sp_tensor(self.norm_adj[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items + 2) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.num_users + self.num_items + 2
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = _convert_sp_mat_to_sp_tensor(self.norm_adj[start:end])
            n_nonzero_temp = self.norm_adj[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

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

        xui = tf.reduce_sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        return tf.matmul(self.Gu, self.Gi, transpose_b=True)

    def one_epoch(self, batches):
        """
        Train recommender model for one epoch.
        Args:
            batches: list of batches to train on
        Returns:
            average loss over epoch
        """
        loss = 0
        steps = 0
        for batch in zip(*batches):
            steps += 1
            loss += self.train_step(batch)
        return loss/steps

    def train_step(self, batch):
        """
        Apply a single training step on one batch.

        Args:
            batch: batch used for the current train step

        Returns:
            loss value at the current batch
        """
        user, pos, neg = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            self._propagate_embeddings()
            xu_pos, gamma_u, gamma_pos = self(inputs=(user, pos), training=True)
            xu_neg, _, gamma_neg = self(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg)] +
                                                [tf.nn.l2_loss(value) for _, value in self.Graph.items()])

            # Loss to be optimized
            loss += reg_loss

        grads = tape.gradient(loss, [self.Gu, self.Gi] +
                                    [value for _, value in self.Graph.items()])
        self.optimizer.apply_gradients(zip(grads, [self.Gu, self.Gi] +
                                           [value for _, value in self.Graph.items()]))

        return loss.numpy()

    def train(self):
        if self.restore():
            self.restore_epochs += 1
        else:
            print("Training from scratch...")

        # initialize the max_ndcg to memorize the best result
        max_hr = 0
        best_model = self
        best_epoch = self.restore_epochs
        results = {}

        for self.epoch in range(self.restore_epochs, self.epochs + 1):
            start_ep = time()
            batches = self.data.shuffle(self.batch_size)
            loss = self.one_epoch(batches)
            epoch_text = 'Epoch {0}/{1} \tLoss: {2:.3f}'.format(self.epoch, self.epochs, loss)
            self.evaluator.eval(self.epoch, results, epoch_text, start_ep)

            # print and log the best result (HR@10)
            if max_hr < results[self.epoch]['hr']:
                max_hr = results[self.epoch]['hr']
                best_epoch = self.epoch
                best_model = deepcopy(self)

            if self.epoch % self.verbose == 0 or self.epoch == 1:
                self.saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/' + \
                                     f'weights-{self.epoch}-{self.learning_rate}-{self.__class__.__name__}')

        self.evaluator.store_recommendation(path=f'{results_dir}/{self.params.dataset}/' + \
                                            f'recs-{self.epoch}-{self.learning_rate}-{self.__class__.__name__}.tsv')
        save_obj(results, f'{results_dir}/{self.params.dataset}/results-metrics-{self.learning_rate}')

        # Store the best model
        print("Store Best Model at Epoch {0}".format(best_epoch))
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=best_model)
        saver_ckpt.save(f'{weight_dir}/{self.params.dataset}/' + \
                        f'best-weights-{best_epoch}-{self.learning_rate}-{self.__class__.__name__}')
        best_model.evaluator.store_recommendation(path=f'{results_dir}/{self.params.dataset}/' + \
                                                       f'best-recs-{best_epoch}-{self.learning_rate}-' + \
                                                       f'{self.__class__.__name__}.tsv')

    def restore(self):
        if self.restore_epochs > 1:
            try:
                checkpoint_file = find_checkpoint(weight_dir, self.restore_epochs, self.epochs,
                                                  self.rec)
                self.saver_ckpt.restore(checkpoint_file)
                print("Model correctly Restored at Epoch: {0}".format(self.restore_epochs))
                return True
            except Exception as ex:
                print("Error in model restoring operation! {0}".format(ex))
        else:
            print("Restore Epochs Not Specified")
        return False
