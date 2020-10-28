from recommender.RecommenderModel import *
from abc import ABC
import tensorflow as tf


class NGCF(RecommenderModel, ABC):
    def __init__(self, data, params, *args, **kwargs):
        """
        Create a NGCF instance.
        (see https://arxiv.org/pdf/1905.08108.pdf for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        super(NGCF, self).__init__(data, params, *args, **kwargs)

        self.embed_k = self.params.embed_k
        self.learning_rate = self.params.lr
        self.l_w = self.params.l_w
        self.weight_size = self.params.weight_size
        self.n_layers = len(self.weight_size)
        self.node_dropout = self.params.node_dropout
        self.message_dropout = self.params.message_dropout
        self.n_fold = self.params.n_fold

        # Initialize Model Parameters
        self._create_weights()
        self._create_embeddings()

    def _create_weights(self):
        initializer = tf.initializers.GlorotUniform()

        self.Gu = tf.Variable(initializer([self.num_users, self.embed_k]), name='Gu')
        self.Gi = tf.Variable(initializer([self.num_items, self.embed_k]), name='Gi')

        self.Graph = dict()

        self.weight_size_list = [self.embed_k] + self.weight_size

        for k in range(self.n_layers):
            self.Graph['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            self.Graph['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            self.Graph['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            self.Graph['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            self.Graph['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            self.Graph['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

    def _create_embeddings(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.Gu, self.Gi], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        self.u_g_embeddings, self.i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat
