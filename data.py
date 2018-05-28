from __future__ import print_function
import numpy as np
from scipy.misc import imread, imresize
from sklearn.preprocessing import OneHotEncoder
# from tqdm import tqdm

from multiprocessing.pool import ThreadPool

import os
import hashlib
import _pickle as cPickle

dims = (128, 128)
random_seed = 231

dims = (128, 128)
random_seed = 231

DEFAULT_X1 = 'data/dogcat/PetImages/Cat'
DEFAULT_X2 = 'data/dogcat/PetImages/Dog'
# DEFAULT_Y = 'data/cars/y.npy'
TRAIN_DEV_TEST = (0.65, 0.20, 0.15)

PATH = '/Users/vineetedupuganti/Downloads/231N_Final_Project/231N-project'

class Data(object):
    def __init__(self, xpath1=DEFAULT_X1, xpath2=DEFAULT_X2, useCache=True, cacheSmash=False, threads=8, first=10000000, x_transpose=(0,1,2,3)):      # 8 seems best on Google Cloud
        assert(sum(TRAIN_DEV_TEST) == 1.0)

        h = hashlib.sha1(bytearray("".join(os.listdir(xpath1)) + xpath2 + str(dims), 'utf-8')).hexdigest()
        p = os.path.join(PATH, "cache", h + ".pkl")
        useCache = os.access(p, os.R_OK) and useCache and not cacheSmash
        if useCache:
            print('Attempting to load from cached pkl object.')
            try:
                with open(p, 'rb') as pkl:
                    self.X, self.y = cPickle.load(pkl)
                    print('success.')
            except:
                print ('Cache load failed.')
                useCache = False
        if not useCache:
            self.loaded = 0
            files1 = [(int(image[:-4]), os.path.join(xpath1, image)) for image in os.listdir(xpath1) if 'Thum' not in image][:first]
            files2 = [(int(image[:-4]), os.path.join(xpath2, image)) for image in os.listdir(xpath2) if 'Thum' not in image][:first]
            pool = ThreadPool(threads)
            X1s = [im[0] for im in list(pool.map(self.get_image, files1)) if im[0].shape == (128, 128, 3) and im[1]] 
            X1 = np.concatenate(X1s)
            X2s = [im[0] for im in list(pool.map(self.get_image, files2)) if im[0].shape == (128, 128, 3) and im[1]] 
            X2 = np.concatenate(X2s)

            self.X = np.concatenate([X1s, X2s])

            self.y = np.concatenate([np.zeros((len(X1s))), np.ones((len(X2s)))])
            if useCache:
                try:
                    with open(p, 'wb+') as pkl:
                        cPickle.dump((self.X, self.y), pkl)
                except:
                    print ('Cache dump failed.')

        self.X = self.X.transpose(x_transpose)
        self.num_examples = self.X.shape[0]
        np.random.seed(random_seed)
        self.indices = np.random.permutation(range(0, self.num_examples))
        self.X = self.X[self.indices]
        self.y = self.y[self.indices]

    def get_image(self, index_file):
        index, file = index_file
        image_data = imread(file)
        self.loaded += 1
        if self.loaded % 100 == 0: print('loaded ' + str(self.loaded))
        keep = True
        ratio = image_data.shape[0] / image_data.shape[1]
        if ratio < 0.7 or ratio > 1/0.7:
            keep = False
        image_data = imresize(image_data, dims)
        return image_data, keep 

    def get_image_(self, index_file):
        initial_y = self.initial_y      # not mutated by threads; everyone just has a reference
        index, file = index_file
        self.loaded += 1                # yeah yeah yeah concurrency issues but I don't care
        if self.loaded % 100 == 0: print('loaded ' + str(self.loaded))
        image_data = imread(file)
        if len(image_data.shape) < 3: return {'X': np.zeros(1, dtype=np.bool), 'y': np.zeros(1, dtype=np.bool)}
        image_data = imresize(image_data, dims)
        X = image_data[None]
        y = initial_y[index][None]
        return {'X': X, 'y': y}

    def get_train(self, split=TRAIN_DEV_TEST):
        return self.X[:int(self.num_examples*split[0])].astype(np.float), self.y[:int(self.num_examples*split[0])]

    def get_dev(self, split=TRAIN_DEV_TEST):
        return self.X[int(self.num_examples*split[0]):int(self.num_examples*(split[0]+split[1]))].astype(np.float), self.y[int(self.num_examples*split[0]):int(self.num_examples*(split[0]+split[1]))]

    def get_test(self, split=TRAIN_DEV_TEST):
        return self.X[int(self.num_examples*(split[0]+split[1])):].astype(np.float), self.y[int(self.num_examples*(split[0]+split[1])):]

    def get_10(self):
        first_10 = np.where(self.y < 10)[0]
        return self.X[first_10].astype(np.float), self.y[first_10]
