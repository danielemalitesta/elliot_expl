"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import os
from ast import literal_eval as make_tuple

import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm

from elliot.recommender.base_recommender_model import init_charger

import elliot.dataset.samplers.pairwise_pipeline_features_sampler as pairpfs
import elliot.dataset.samplers.pointwise_pipeline_features_sampler as pointpfs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.custom.FashionExpl.FashionExpl_model import FashionExpl_model
from elliot.utils.write import store_recommendation

np.random.seed(0)
tf.random.set_seed(0)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FashionExpl(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "f", 100, int, None),
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_l_w", "l_w", "lw", 0.000025, None, None),
            ("_l_color", "l_color", "lc", 0.000025, None, None),
            ("_l_shape", "l_shape", "ls", 0.000025, None, None),
            ("_l_att", "l_att", "la", 0.000025, None, None),
            ("_l_out", "l_out", "lo", 0.000025, None, None),
            ("_mlp_color", "mlp_color", "mlpc", "(64,1)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_cnn_channels", "cnn_channels", "cnnch", 32, None, None),
            ("_cnn_kernels", "cnn_kernels", "cnnk", 3, None, None),
            ("_cnn_strides", "cnn_strides", "cnns", 1, None, None),
            ("_mlp_cnn", "mlp_cnn", "mlpc", "(64,1)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_mlp_att", "mlp_att", "mlpa", "(64,1)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_mlp_out", "mlp_out", "mlpo", "(64,1)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "d", 0.2, None, None),
            ("_att_feat_agg", "att_feat_agg", "afa", "multiplication", None, None),
            ("_out_feat_agg", "out_feat_agg", "ofa", "multiplication", None, None),
            ("_sampler_str", "sampler", "s", "pairwise", None, None)
        ]

        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        # dictionary with key (user) and value (attention weights for user's positive items)
        self._attention_dict = dict()

        if self._sampler_str == 'pairwise':
            self._sampler = pairpfs.Sampler(self._data.i_train_dict,
                                            item_indices,
                                            self._data.side_information_data.shapes_src_folder,
                                            self._data.side_information_data.colors_src_folder,
                                            self._data.side_information_data.classes_src_folder,
                                            self._data.output_shape_size,
                                            self._epochs)
        elif self._sampler_str == 'pointwise':
            self._sampler = pointpfs.Sampler(self._data.i_train_dict,
                                             item_indices,
                                             self._data.side_information_data.shapes_src_folder,
                                             self._data.side_information_data.colors_src_folder,
                                             self._data.side_information_data.classes_src_folder,
                                             self._data.output_shape_size,
                                             self._epochs)
        else:
            raise NotImplementedError('This sampler type has not been implemented for this model yet!')

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        # only for evaluation purposes
        self._next_eval_batch = self._sampler.pipeline_eval(self._batch_size)

        self._model = FashionExpl_model(self._factors,
                                        self._mlp_color,
                                        self._mlp_att,
                                        self._mlp_out,
                                        self._mlp_cnn,
                                        self._cnn_channels,
                                        self._cnn_kernels,
                                        self._cnn_strides,
                                        self._att_feat_agg,
                                        self._out_feat_agg,
                                        self._sampler_str,
                                        self._dropout,
                                        self._learning_rate,
                                        self._l_w,
                                        self._l_color,
                                        self._l_shape,
                                        self._l_att,
                                        self._l_out,
                                        self._num_users,
                                        self._num_items)

    @property
    def name(self):
        return "FashionExpl" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0
        loss = 0
        steps = 0
        it = 0

        with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
            for batch in self._next_batch:
                steps += 1
                loss += self._model.train_step(batch)
                t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                t.update()

                # epoch is over
                if steps == self._data.transactions // self._batch_size:
                    t.reset()
                    if not (it + 1) % self._validation_rate:
                        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                        result_dict = self.evaluator.eval(recs)
                        self._results.append(result_dict)

                        self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / steps:.3f}')

                        if self._results[-1][self._validation_k]["val_results"][
                            self._validation_metric] > best_metric_value:
                            best_metric_value = self._results[-1][self._validation_k]["val_results"][
                                self._validation_metric]
                            if self._save_weights:
                                self._model.save_weights(self._saving_filepath)
                            if self._save_recs:
                                store_recommendation(recs,
                                                     self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")
                            with open(self._config.path_output_rec_result + f"{self.name}_attention.pkl",
                                      "wb") as f:
                                pickle.dump(self._attention_dict, f)
                    it += 1
                    steps = 0
                    loss = 0

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        steps = 0
        color_features = np.empty((self._num_items, self._factors))
        shape_features = np.empty((self._num_items, self._factors))
        class_features = np.empty((self._num_items, self._factors))
        for batch in self._next_eval_batch:
            item, shape, col, class_ = batch
            output_col = self._model.color_encoder(col, training=False)
            output_shape = self._model.shape_encoder(shape, training=False)
            color_features[steps:steps + output_col.shape[0]] = output_col.numpy()
            shape_features[steps:steps + output_shape.shape[0]] = output_shape.numpy()
            class_features[steps:steps + output_col.shape[0]] = class_.numpy()
            steps += output_col.shape[0]

        color_features = tf.Variable(color_features, dtype=tf.float32)
        shape_features = tf.Variable(shape_features, dtype=tf.float32)
        class_features = tf.Variable(class_features, dtype=tf.float32)

        for index, offset in enumerate(range(0, self._num_users, self._params.batch_size)):
            offset_stop = min(offset + self._params.batch_size, self._num_users)
            predictions = np.empty((offset_stop - offset, self._num_items))
            attention = np.empty((offset_stop - offset, self._num_items, 3))
            for i_ in range(self._num_items):
                p, a = self._model.predict_batch(offset, offset_stop,
                                                 tf.repeat(tf.expand_dims(self._model.Gi[i_], 0), repeats=(offset_stop - offset), axis=0),
                                                 tf.repeat(tf.expand_dims(color_features[i_], 0), repeats=(offset_stop - offset), axis=0),
                                                 tf.repeat(tf.expand_dims(shape_features[i_], 0), repeats=(offset_stop - offset), axis=0),
                                                 tf.repeat(tf.expand_dims(class_features[i_], 0), repeats=(offset_stop - offset), axis=0))
                predictions[:(offset_stop - offset), i_], \
                    attention[:(offset_stop - offset), i_, :] = p.numpy(), a.numpy()

            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
            self._attention_dict = {**self._attention_dict,
                                    **{u_abs: attention[u_rel, self._data.sp_i_train.toarray()[u_abs] == 1]
                                       for u_abs, u_rel in zip(range(offset, offset_stop), range(offset_stop - offset))}}

        return predictions_top_k
