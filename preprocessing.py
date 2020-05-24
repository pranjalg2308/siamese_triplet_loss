import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import random
from zipfile import ZipFile


class PreProcessing:

    def __init__(self, data_path, verbose=True):

        self.train_images = []
        self.val_images = []
        self.verbose = verbose
        self.data_path = data_path
        self.train_path = os.path.join(self.data_path, 'images_background')
        self.val_path = os.path.join(self.data_path, 'images_evaluation')
        self.train_pickle = os.path.join(self.data_path, 'train.pickle')
        self.val_pickle = os.path.join(self.data_path, 'validation.pickle')
        if not os.path.exists(self.train_path):
            with ZipFile(self.train_path + '.zip', 'r') as zip:
                if verbose: print('Extracting image background...')
                zip.extractall()
                if verbose: print('Image Background Extracted')

        if not os.path.exists(self.val_path):
            with ZipFile(self.val_path + '.zip', 'r') as zip:
                if verbose: print('Extracting image evaluation...')
                zip.extractall()
                if verbose: print('Image Evaluation Extracted')

        if os.path.exists(self.train_pickle):
            with open(os.path.join(data_path, "train.pickle"), "rb") as f:
                self.train_images = pickle.load(f)
        else:
            self.train_images = self.load_image(self.train_path)
            self.save_pickle('train.pickle', self.train_images)

        if os.path.exists(self.val_pickle):
            with open(os.path.join(data_path, "train.pickle"), "rb") as f:
                self.val_images = pickle.load(f)
        else:
            self.val_images = self.load_image(self.val_path)
            self.save_pickle('validation.pickle', self.val_images)

    def load_image(self, path):
        X = []
        if self.verbose:
            print('Loading Image From ', path)

        for alphabet in os.listdir(path):
            if self.verbose:
                print("Loading Alphabet : ", alphabet)
            alphabet_path = os.path.join(path, alphabet)
            for letter in os.listdir(alphabet_path):
                letter_path = os.path.join(alphabet_path, letter)
                letter_images = []
                for filename in os.listdir(letter_path):
                    file_path = os.path.join(letter_path, filename)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = np.reshape(img, (105, 105, 1))
                    img = img / 255
                    letter_images.append(img)
                try:
                    X.append(np.stack(letter_images))
                except ValueError as e:
                    print(e)
        X = np.stack(X)
        return X

    def save_pickle(self, pickle_name, obj):
        with open(os.path.join(self.data_path, pickle_name), "wb") as f:
            pickle.dump(obj, f)

    def get_triplet(self, shape):
        class_1 = random.randint(0, shape[0] - 1)
        class_2 = random.randint(0, shape[0] - 1)
        a, p = (class_1, random.randint(0, shape[1] - 1)), (class_1, random.randint(0, shape[1] - 1))
        n = (class_2, random.randint(0, shape[1] - 1))
        # print(a, p, n)
        return a, p, n

    def get_triplet_batch(self, batch_size, train_data=True):
        anchor_image = []
        positive_image = []
        negative_image = []
        if train_data:
            X = self.train_images
        else:
            X = self.val_images

        for _ in range(batch_size):
            ida, idp, idn = self.get_triplet(X.shape)
            anchor_image.append(X[ida])
            positive_image.append(X[idp])
            negative_image.append(X[idn])

        ai = np.array(anchor_image)
        pi = np.array(positive_image)
        ni = np.array(negative_image)
        return [ai, pi, ni]


