from PIL import Image
from config.configs import *
import tensorflow as tf
import numpy as np
import os


class Dataset:
    def __init__(self, dataset):
        self.directory = images_path.format(dataset)
        self.filenames = os.listdir(self.directory)
        self.filenames.sort(key=lambda x: int(x.split(".")[0]))
        self.num_samples = len(self.filenames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = Image.open(self.directory + self.filenames[idx])

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        sample = np.expand_dims(tf.keras.applications.resnet50.preprocess_input(np.array(sample)), axis=0)

        return sample, self.filenames[idx]
