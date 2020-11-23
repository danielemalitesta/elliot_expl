from vision.FeatureExtractor import *
from vision.Dataset import *
from config.configs import *
from utils.write import *
from utils.read import *
import argparse
import numpy as np
import time
import cv2
import sys

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

# Reference for background removal
# https://medium.com/@chris.s.park/image-background-removal-using-opencv-part-1-da3695ac66b6

amazon_men = os.listdir('../data/amazon_men/original/images')
amazon_women = os.listdir('../data/amazon_women/original/images')
tradesy = os.listdir('../data/amazon_men/original/images')

random.shuffle(amazon_men)
random.shuffle(amazon_women)
random.shuffle(tradesy)

first_amazon_men = amazon_men[0]
first_amazon_women = amazon_women[0]
first_tradesy = tradesy[0]

list_images = ['amazon_men/original/images/' + first_amazon_men,
               'amazon_women/original/images/' + first_amazon_women,
               'tradesy/original/images/' + first_tradesy]

for index, path in enumerate(list_images):
    try:
        image_vec = cv2.imread('../data/' + path, 1)
        plt.figure()
        plt.title(index)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image_vec, cv2.COLOR_BGR2RGB))
        plt.show()

        gray = cv2.cvtColor(image_vec, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        Ie1 = cv2.Canny(gray, 255/3, 255)
        # according to https://stackoverflow.com/questions/25125670/best-value-for-threshold-in-canny
        f = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Ie2 = cv2.filter2D(gray, -1, f)
        Ie = Ie1 + Ie2
        Ie_end = np.clip(255 - Ie, a_min=0, a_max=255)

        plt.figure()
        plt.title(index)
        plt.axis("off")
        plt.imshow(Ie_end, cmap='gray')
        plt.show()

        # cv2.imshow('Title', Ie_end)
        # cv2.waitKey()

        contours, hierarchy = cv2.findContours(np.clip(Ie, a_min=0, a_max=255), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_info = []
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda cont: cont[2], reverse=True)
        max_contour = contour_info[0]
        mask = np.copy(image_vec)
        cv2.fillPoly(mask, pts=[max_contour[0]], color=(0, 0, 0))

        plt.figure()
        plt.title(index)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        plt.show()

        image_for_clustering = np.copy(image_vec)
        image_for_clustering = image_for_clustering[(mask == 0).all(axis=2)]
        # https://stackoverflow.com/questions/22397094/using-opencv-numpy-to-find-white-pixels-in-a-color-image-using-python

        # Reference for k-means
        # https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
        # https://buzzrobot.com/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036

        clt = KMeans(n_clusters=10, random_state=1234)
        clt.fit(image_for_clustering / np.float32(255.0))

        _, counts_elements = np.unique(clt.labels_, return_counts=True)
        clusters_dict = [{'color': (clt.cluster_centers_[i] * 255).astype("uint8"),
                          'size': counts_elements[i]} for i in range(counts_elements.shape[0])]
        top_clusters_dict = sorted(clusters_dict, key=lambda k: k['size'], reverse=True)[:3]


        def centroid_histogram(topk):
            # grab the number of different clusters and create a histogram
            # based on the number of pixels assigned to each cluster
            numLabels = np.arange(0, topk + 1)
            (hist, _) = np.histogram(np.arange(0, topk), bins=numLabels)
            # normalize the histogram, such that it sums to one
            hist = hist.astype("float")
            hist /= hist.sum()
            # return the histogram
            return hist


        def plot_colors(hist, top_centroids_dict):
            # initialize the bar chart representing the relative frequency
            # of each of the colors
            bar = np.zeros((50, 150, 3), dtype="uint8")
            startX = 0
            # loop over the percentage of each cluster and the color of
            # each cluster

            centroids = [value['color'] for value in top_centroids_dict]

            for (percent, color) in zip(hist, centroids):
                # plot the relative percentage of each cluster
                endX = startX + (percent * 150)
                cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                              color.tolist(), -1)
                startX = endX

            # return the bar chart
            return bar


        hist = centroid_histogram(topk=3)
        bar = plot_colors(hist, top_clusters_dict)
        # show our color bart
        plt.figure()
        plt.title(index)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
        plt.show()
        # https://stackoverflow.com/questions/43554819/find-most-frequent-row-or-mode-of-a-matrix-of-vectors-python-numpy
    except:
        print(path)
