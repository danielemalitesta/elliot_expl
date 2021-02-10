"""
Module description:

"""
from utils import logging

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import random
from tqdm import tqdm

from dataset.samplers import sparse_sampler as sp
from evaluation.evaluator import Evaluator
from recommender import BaseRecommenderModel
from utils.folder import build_model_folder
from utils.write import store_recommendation

from recommender.autoencoders.vae.multi_vae_utils import VariationalAutoEncoder
from recommender.recommender_utils_mixin import RecMixin

np.random.seed(42)
random.seed(0)


class MultiVAE(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        """
        """
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._random_p = random

        self._ratings = self._data.train_dict
        self._sampler = sp.Sampler(self._data.sp_i_train)

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################


        self._params_list = [
            ("_intermediate_dim", "intermediate_dim", "intermediate_dim", 600, int, None),
            ("_latent_dim", "latent_dim", "latent_dim", 200, int, None),
            ("_lambda", "reg_lambda", "reg_lambda", 0.01, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_dropout_rate", "dropout_pkeep", "dropout_pkeep", 1, None, None),
        ]
        self.autoset_params()

        self._dropout_rate = 1. - self._dropout_rate

        self._model = VariationalAutoEncoder(self._num_items,
                                           self._intermediate_dim,
                                           self._latent_dim,
                                           self._learning_rate,
                                           self._dropout_rate,
                                           self._lambda)

        # the total number of gradient updates for annealing
        self._total_anneal_steps = 200000
        # largest annealing parameter
        self._anneal_cap = 0.2

        self.evaluator = Evaluator(self._data, self._params)
        self._params.name = self.name
        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'
        self.logger = logging.get_logger(self.__class__.__name__)

    @property
    def name(self):
        return "MultiVAE" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0
        self._update_count = 0

        for it in range(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._num_users // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._num_users, self._batch_size):
                    steps += 1

                    if self._total_anneal_steps > 0:
                        anneal = min(self._anneal_cap, 1. * self._update_count / self._total_anneal_steps)
                    else:
                        anneal = self._anneal_cap

                    loss += self._model.train_step(batch, anneal)
                    t.set_postfix({'loss': f'{loss.numpy()/steps:.5f}'})
                    t.update()
                    self._update_count += 1

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss/steps:.5f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def restore_weights(self):
        try:
            self._model.load_weights(self._saving_filepath)
            print(f"Model correctly Restored")

            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)
            self._results.append(result_dict)

            print("******************************************")
            if self._save_recs:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")
            return True

        except Exception as ex:
            print(f"Error in model restoring operation! {ex}")

        return False

