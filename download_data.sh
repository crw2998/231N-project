mkdir data
mkdir cache
cd data
mkdir cars
cd cars
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
tar -xvzf car_ims.tgz
rm car_ims.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat
python ../../get_mat_annos.py cars_annos.mat
