import logging
import os
from abc import ABC
from PIL import Image
from copy import deepcopy

from time import time
from multiprocessing import Pool
from multiprocessing import cpu_count

from recommender.RecommenderModel import RecommenderModel
from utils.read import find_checkpoint
from recommender.visual.cnn import *
from recommender.Evaluator import Evaluator
from utils.write import save_obj
from config.configs import *

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _read_images_triple(batch):
    user, pos, neg, dataset = batch
    pos = pos[0]
    neg = neg[0]
    # load positive and negative item images
    im_pos = Image.open(images_path.format(dataset) + str(pos) + '.jpg')
    im_neg = Image.open(images_path.format(dataset) + str(neg) + '.jpg')

    try:
        im_pos.load()
    except ValueError:
        print(f'Image at path {pos}.jpg was not loaded correctly!')

    try:
        im_neg.load()
    except ValueError:
        print(f'Image at path {neg}.jpg was not loaded correctly!')

    if im_pos.mode != 'RGB':
        im_pos = im_pos.convert(mode='RGB')
    if im_neg.mode != 'RGB':
        im_neg = im_neg.convert(mode='RGB')

    im_pos = (np.array(im_pos.resize((224, 224))) - np.float32(127.5)) / np.float32(127.5)
    im_neg = (np.array(im_neg.resize((224, 224))) - np.float32(127.5)) / np.float32(127.5)
    return user, im_pos, im_neg


class DVBPR(RecommenderModel, ABC):
    def __init__(self, data, params):
        """
        Create a DVBPR instance.
        (see https://arxiv.org/pdf/1711.02231.pdf for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {[lambda1, lambda2]: regularization,
                                      lr: learning rate}
        """
        super(DVBPR, self).__init__(data, params)
        self.lambda1 = self.params.lambda1
        self.lambda2 = self.params.lambda2
        self.learning_rate = self.params.lr
        self.embed_k = self.params.embed_k

        self.evaluator = Evaluator(self, data, params.k)

        # Initialize Model Parameters
        initializer = tf.initializers.GlorotUniform()
        self.Tu = tf.Variable(initializer(shape=[self.num_users, self.embed_k]), name='Tu', dtype=tf.float32)
        self.cnn = CNN(self.embed_k)
        self.Phi = np.empty(shape=[self.num_items, self.embed_k], dtype=np.float32)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    def _forward_cnn(self, item):
        return self.cnn(inputs=item, training=False)

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
        cnn_output = self.cnn(inputs=item, training=True)
        theta_u = tf.squeeze(tf.nn.embedding_lookup(self.Tu, user))

        xui = tf.reduce_sum(theta_u * cnn_output, 1)

        return xui, theta_u, cnn_output

    def prediction(self, user, item):
        with tf.name_scope("prediction"):
            pred, _, phi = self._forward(user, item)

            return pred

    def predict_all(self):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        # load all images and calculate phi for each of them
        # assign phi to Phi to get the overall Phi vector
        # calculate the prediction for all users-items
        images_list = os.listdir(images_path.format(self.dataset_name))
        images_list.sort(key=lambda x: int(x.split(".")[0]))
        for index, item in enumerate(images_list):
            im = Image.open(images_path.format(self.dataset_name) + item)
            try:
                im.load()
            except ValueError:
                print(f'Image at path {images_path.format(self.dataset_name) + item} was not loaded correctly!')
            if im.mode != 'RGB':
                im = im.convert(mode='RGB')
            im = np.reshape((np.array(im.resize((224, 224))) - np.float32(127.5)) / np.float32(127.5), (1, 224, 224, 3))
            phi = self._forward_cnn(im)
            self.Phi[index, :] = phi
        return tf.tensordot(self.Tu, tf.Variable(self.Phi), axes=[[1], [1]])

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
        user_batch, pos_batch, neg_batch = batch
        zip_batch = zip(user_batch, pos_batch, neg_batch, [self.dataset_name]*len(user_batch))
        pool = Pool(cpu_count())
        res = pool.map(_read_images_triple, zip_batch)
        pool.close()
        pool.join()

        list_users, list_pos, list_neg = [], [], []
        for r in res:
            list_users.append(r[0])
            list_pos.append(r[1])
            list_neg.append(r[2])

        list_users = np.array(list_users)
        list_pos = np.array(list_pos)
        list_neg = np.array(list_neg)

        with tf.GradientTape() as tape:
            # Clean Inference
            xu_pos, beta_pos, theta_u = self(inputs=(list_users, list_pos), training=True)
            xu_neg, beta_neg, _ = self(inputs=(list_users, list_neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))

            # Regularization Component
            reg_loss = self.lambda1 * tf.nn.l2_loss(theta_u) \
                       + self.lambda2 * tf.reduce_sum([tf.nn.l2_loss(layer)
                                                       for layer in self.cnn.trainable_variables
                                                       if 'bias' not in layer.name])

            # Loss to be optimized
            loss += reg_loss

        params = [self.Tu,
                  *self.cnn.trainable_variables]

        grads = tape.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

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
            if max_hr < results[self.epoch]['hr'][self.evaluator.k - 1]:
                max_hr = results[self.epoch]['hr'][self.evaluator.k - 1]
                best_epoch = self.epoch
                best_model = deepcopy(self)

            if self.epoch % self.verbose == 0 or self.epoch == 1:
                self.saver_ckpt.save('{0}/weights-{1}-DVBPR'.format(weight_dir, self.epoch))

        self.evaluator.store_recommendation()
        save_obj(results, '{0}/{1}-results'.format(results_dir, self.path_output_rec_result.split('/')[-2]))

        # Store the best model
        print("Store Best Model at Epoch {0}".format(best_epoch))
        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=best_model)
        saver_ckpt.save('{0}/best-weights-{1}'.format(self.path_output_rec_weight, best_epoch))
        best_model.evaluator.store_recommendation()

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
