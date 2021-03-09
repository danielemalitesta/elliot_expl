"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import tensorflow as tf
from PIL import Image

import numpy as np
import random
np.random.seed(42)
random.seed(42)


class Sampler:
    def __init__(self, indexed_ratings, item_indices, shapes_path, colors_path, classes_path, output_shape_size, epochs):
        self._indexed_ratings = indexed_ratings
        self._item_indices = item_indices
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

        self._shapes_path = shapes_path
        self._colors_path = colors_path
        self._classes_path = classes_path
        self._output_shape_size = output_shape_size
        self._epochs = epochs

    def read_images_triple(self, user, item, pos_neg):
        # load positive and negative item images
        im = Image.open(self._shapes_path + str(item.numpy()) + '.tiff')

        col = np.load(self._colors_path + str(item.numpy()) + '.npy')

        class_ = np.load(self._classes_path + str(item.numpy()) + '.npy')

        try:
            im.load()
        except ValueError:
            print(f'Image at path {item}.tiff was not loaded correctly!')

        im = np.expand_dims(np.array(im.resize(self._output_shape_size)) / np.float32(255.0), axis=2)

        col = col / np.max(np.abs(col))

        return user.numpy(), item.numpy(), pos_neg, im, col, class_

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        user, item, pos_neg = [], [], []

        actual_inter = (events // batch_size) * batch_size * self._epochs

        counter_inter = 1

        def sample():
            u = r_int(n_users)
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            b = random.getrandbits(1)
            if b:
                i = ui[r_int(lui)]
            else:
                i = r_int(n_items)
                while i in ui:
                    i = r_int(n_items)
            user.append(u), item.append(i), pos_neg.append(b)

        for ep in range(self._epochs):
            for _ in range(0, events):
                sample()
                if counter_inter == actual_inter:
                    return user, item, pos_neg
                else:
                    counter_inter += 1

        return user, item, pos_neg

    def pipeline(self, num_users, batch_size):
        def load_func(u, i, p_n):
            b = tf.py_function(
                self.read_images_triple,
                (u, i, p_n,),
                (np.int32, np.int32, np.int32, np.float32, np.float32, np.float32)
            )
            return b
        all_triples = self.step(events=num_users, batch_size=batch_size)
        data = tf.data.Dataset.from_tensor_slices(all_triples)
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    # this is only for evaluation
    def pipeline_eval(self, batch_size):
        def load_func(i):
            b = tf.py_function(
                self.read_image,
                (i,),
                (np.int32, np.float32, np.float32, np.float32)
            )
            return b

        data = tf.data.Dataset.from_tensor_slices(self._items)
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    # this is only for evaluation
    def read_image(self, item):
        # load positive image
        im = Image.open(self._shapes_path + str(item.numpy()) + '.tiff')
        col = np.load(self._colors_path + str(item.numpy()) + '.npy')
        class_ = np.load(self._classes_path + str(item.numpy()) + '.npy')

        try:
            im.load()
        except ValueError:
            print(f'Image at path {item}.jpg was not loaded correctly!')

        im = np.expand_dims(np.array(im.resize(self._output_shape_size)) / np.float32(255.0), axis=2)
        col = col / np.max(np.abs(col))

        return item, im, col, class_
