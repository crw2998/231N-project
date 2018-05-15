from __future__ import print_function
import numpy as np
from scipy.misc import imread, imresize
from tqdm import tqdm

import os
import hashlib
import _pickle as cPickle

dims = (240, 320)
random_seed = 231

DEFAULT_X = 'data/cars/car_ims'
DEFAULT_Y = 'data/cars/y.npy'
TRAIN_DEV_TEST = (0.65, 0.20, 0.15)

PATH = '/home/colewinstanley/231N-project'

class Data(object):
    def __init__(self, xpath=DEFAULT_X, yfile=DEFAULT_Y, cacheSmash=False):
        assert(sum(TRAIN_DEV_TEST) == 1.0)
        self.X = None

        initial_y = np.genfromtxt(yfile)

        h = hashlib.sha1(bytearray("".join(os.listdir(xpath)) + yfile + str(dims), 'utf-8')).hexdigest()
        p = os.path.join(PATH, "cache", h + ".pkl")
        useCache = os.access(p, os.R_OK) and not cacheSmash
        if useCache:
            print('Attempting to load from cached pkl object:' + str(h))
            try:
                with open(p, 'rb') as pkl:
                    self.X, self.y = cPickle.load(pkl)
                    print('success.')
            except:
                print ('Cache load failed.')
                useCache = False
        if not useCache:
            for image in tqdm(os.listdir(xpath)):
                index = int(image[:6])
                image_data = imread(os.path.join(xpath, image))
                if len(image_data.shape) < 3: continue
                image_data = imresize(image_data, dims)
                if self.X is None:
                    self.X = image_data[None]
                    self.y = initial_y[index][None]
                else:
                    self.X = np.concatenate((self.X, image_data[None]))
                    self.y = np.concatenate((self.y, initial_y[index][None]))
            try:
                with open(p, 'wb+') as pkl:
                    cPickle.dump((self.X, self.y), pkl)
            except:
                print ('Cache dump failed.')


        self.num_examples = self.X.shape[0]
        np.random.seed(random_seed)
        self.indices = np.random.permutation(range(0, self.num_examples))

    def get_train(self, split=TRAIN_DEV_TEST):
        return self.X[:int(self.num_examples*split[0])], self.y[:int(self.num_examples*split[0])]

    def get_dev(self, split=TRAIN_DEV_TEST):
        return self.X[int(self.num_examples*split[0]):int(self.num_examples*split[1])], self.y[int(self.num_examples*split[0]):int(self.num_examples*split[1])]

    def get_test(self, split=TRAIN_DEV_TEST):
        return self.X[int(self.num_examples*split[1]):], self.y[int(self.num_examples*split[1]):]