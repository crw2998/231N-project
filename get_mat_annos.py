import sys
import scipy.io as sio
import numpy as np


def main(*args):
	file = sys.argv[1]
	anno = sio.loadmat(file)
	y = np.zeros(anno['annotations'].shape[1] + 1, dtype=np.int16)
	for ex in anno['annotations'][0]:
		index = int(ex[0][0][8:14])
		y[index] = ex[5]
	np.savetxt('y.npy', y)

if __name__ == '__main__':
	main()