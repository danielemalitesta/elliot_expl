import tensorflow as tf
import numpy as np
import os


class FeatureExtractor:
    def __init__(self, output_layer='avg_pool', model_name='ResNet50', imagenet=None):
        self.model_name = model_name
        self.output_layer = output_layer
        self.imagenet = imagenet

        if self.model_name == 'ResNet50':
            self.classifier = tf.keras.applications.ResNet50()
            self.model = tf.keras.Model(self.classifier.input, self.classifier.get_layer(output_layer).output)
        elif self.model_name == 'ResNet152':
            self.classifier = tf.keras.applications.ResNet152()
            self.model = tf.keras.Model(self.classifier.input, self.classifier.get_layer(output_layer).output)
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def classify(self, sample):
        image, filename = sample
        output = self.classifier.predict(image, batch_size=1)

        return {'ImageID': os.path.splitext(filename)[0],
                'ClassStr': self.imagenet[int(np.argmax(output))],
                'ClassNum': np.argmax(output),
                'Prob': np.amax(output)}
