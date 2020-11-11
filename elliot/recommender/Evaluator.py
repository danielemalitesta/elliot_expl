import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from time import time
import datetime
import heapq

_feed_dict = None
_dataset = None
_model = None
_sess = None
_K = None


def _init_eval_model(data):
    global _dataset
    _dataset = data

    pool = Pool(cpu_count() - 1)
    feed_dicts = pool.map(_evaluate_input_list, range(_dataset.num_users))
    pool.close()
    pool.join()

    return feed_dicts


def _evaluate_input_list(user):
    validation_items = _dataset.validation_list[user]

    if len(validation_items) > 0:
        item_input = set(range(_dataset.num_items)) - set(_dataset.train_list[user])

        for validation_item in validation_items:
            if validation_item in item_input:
                item_input.remove(validation_item)

        item_input = list(item_input)

        for validation_item in validation_items:
            item_input.append(validation_item)

        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input
    else:
        print('User {} has no validation list!'.format(user))
        return 0, 0


def _eval_by_user(user, curr_pred):
    # get predictions of data in validation set
    user_input, item_input = _feed_dicts[user]
    if type(user_input) != np.ndarray:
        return ()

    # AREA UNDER CURVE (AUC)
    predictions = curr_pred[list(item_input.reshape(-1))]
    neg_predict, pos_predict = predictions[:-len(_dataset.validation_list[user])], \
                               predictions[-len(_dataset.validation_list[user]):]

    position = 0
    for t in range(len(_dataset.validation_list[user])):
        position += (neg_predict >= pos_predict[t]).sum()

    auc = 1 - (position / (len(neg_predict) * len(pos_predict)))
    # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

    # HIT RATIO (HR)
    item_score = {}
    for i in list(item_input.reshape(-1)):
        item_score[i] = curr_pred[i]

    k_max_item_score = heapq.nlargest(_K, item_score, key=item_score.get)

    r = []
    for i in k_max_item_score:
        if i in item_input[-len(_dataset.validation_list[user]):]:
            r.append(1)
        else:
            r.append(0)

    hr = 1. if sum(r) > 0 else 0.

    # PRECISION (P)
    prec = sum(r) / len(r)

    # RECALL (R)
    rec = sum(r) / len(pos_predict)

    return hr, prec, rec, auc


class Evaluator:
    def __init__(self, model, data, k):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.data = data
        self.k = k
        self.eval_feed_dicts = _init_eval_model(data)
        self.model = model

    def eval(self, epoch=0, results={}, epoch_text='', start_time=0):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        global _model
        global _K
        global _dataset
        global _feed_dicts
        _dataset = self.data
        _model = self.model
        _K = self.k
        _feed_dicts = self.eval_feed_dicts

        res = []

        eval_start_time = time()
        all_predictions = self.model.predict_all().numpy()

        for user in range(self.model.data.num_users):
            current_prediction = all_predictions[user, :]
            res.append(_eval_by_user(user, current_prediction))

        res = list(filter(None, res))
        hr, prec, rec, auc = (np.array(res).mean(axis=0)).tolist()
        print("%s \tTrain Time: %s \tValidation Time: %s \tMetrics@%d ==> HR: %.4f \tPrec: %.4f \tRec: %.4f \tAUC: %.4f" % (
            epoch_text,
            datetime.timedelta(seconds=(time() - start_time)),
            datetime.timedelta(seconds=(time() - eval_start_time)),
            _K,
            hr,
            prec,
            rec,
            auc))

        if len(epoch_text) != '':
            results[epoch] = {'hr': hr, 'auc': auc}

    def store_recommendation(self, attack_name="", path=""):
        """
        Store recommendation list (top-k) in order to be used for the ranksys framework (anonymized)
        attack_name: The name for the attack stored file
        :return:
        """
        results = self.model.predict_all().numpy()
        with open(path, 'w') as out:
            for u in range(results.shape[0]):
                results[u][self.data.train_list[u]] = -np.inf
                top_k_id = results[u].argsort()[-self.k:][::-1]
                top_k_score = results[u][top_k_id]
                for i, value in enumerate(top_k_id):
                    out.write(str(u) + '\t' + str(value) + '\t' + str(top_k_score[i]) + '\n')
