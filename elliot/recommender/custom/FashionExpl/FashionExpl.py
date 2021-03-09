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
from tqdm import tqdm

from elliot.recommender.base_recommender_model import init_charger

import elliot.dataset.samplers.pipeline_features_sampler as pfs
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
            ("_factors", "factors", "factors", 100, None, None),
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_l_w", "l_w", "l_w", 0.000025, None, None),
            ("_l_color", "l_color", "l_color", 0.000025, None, None),
            ("_l_shape", "l_shape", "l_shape", 0.000025, None, None),
            ("_l_att", "l_att", "l_att", 0.000025, None, None),
            ("_l_out", "l_out", "l_out", 0.000025, None, None),
            ("_mlp_color", "mlp_color", "mlp_color", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_cnn_channels", "cnn_channels", "cnn_channels", "(32,)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_cnn_kernels", "cnn_kernels", "cnn_kernels", "((3,3),)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_cnn_strides", "cnn_strides", "cnn_strides", "((1,1),)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_mlp_cnn", "mlp_cnn", "mlp_cnn", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_mlp_att", "mlp_att", "mlp_att", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_mlp_out", "mlp_out", "mlp_out", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "drop", 0.2, None, None),
            ("_item_feat_agg", "item_feat_agg", "item_feat_aggr", "multiplication", None, None),
            ("_sampler", "sampler", "sampler", "pairwise", None, None)
        ]

        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = pfs.Sampler(self._data.i_train_dict,
                                    item_indices,
                                    self._data.side_information_data.shapes_src_folder,
                                    self._data.side_information_data.colors_src_folder,
                                    self._data.side_information_data.classes_src_folder,
                                    self._data.output_shape_size,
                                    self._epochs)

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        # only for evaluation purposes
        self._next_eval_batch = self._sampler.pipeline_eval(self._batch_size)

        self._model = FashionExpl_model(self._factors,
                                        self._mlp_color,
                                        self._mlp_att,
                                        self._mlp_out,
                                        self._dropout,
                                        self._learning_rate,
                                        self._l_w,
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
                    it += 1
                    steps = 0
                    loss = 0

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._params.batch_size)):
            offset_stop = min(offset + self._params.batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k
