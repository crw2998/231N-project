from __future__ import print_function
import numpy as np
from scipy.misc import imread, imresize
from sklearn.preprocessing import OneHotEncoder
# from tqdm import tqdm

from multiprocessing.pool import ThreadPool

import os
import hashlib
import _pickle as cPickle

dims = (96, 128)
random_seed = 231

DEFAULT_X = 'data/cars/car_ims'
DEFAULT_Y = 'data/cars/y.npy'
TRAIN_DEV_TEST = (0.65, 0.20, 0.15)

PATH = '/home/colewinstanley/231N-project'

class Data(object):
    def __init__(self, xpath=DEFAULT_X, yfile=DEFAULT_Y, useCache=True, cacheSmash=False, threads=8, first=10000000):      # 8 seems best on Google Cloud
        assert(sum(TRAIN_DEV_TEST) == 1.0)

        h = hashlib.sha1(bytearray("".join(os.listdir(xpath)) + yfile + str(dims), 'utf-8')).hexdigest()
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
            self.initial_y = np.genfromtxt(yfile)
            files = [(int(image[:6]), os.path.join(xpath, image)) for image in os.listdir(xpath)][:first]
            pool = ThreadPool(threads)
            self.loaded = 0
            results = pool.map(self.get_image, files)
            self.X = np.concatenate([d['X'] for d in results if d['X'].dtype != np.bool])       # using dtype as sentinel
            self.y = np.concatenate([d['y'] for d in results if d['X'].dtype != np.bool])
            if useCache:
                try:
                    with open(p, 'wb+') as pkl:
                        cPickle.dump((self.X, self.y), pkl)
                except:
                    print ('Cache dump failed.')

            del self.initial_y

        self.num_examples = self.X.shape[0]
        np.random.seed(random_seed)
        self.indices = np.random.permutation(range(0, self.num_examples))
        self.X = self.X[self.indices]
        self.y = self.y[self.indices]

    def get_image(self, index_file):
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