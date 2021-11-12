# co-regularied-alignment-for-domain-adaptation

Original repository: https://github.com/jakc4103/co-regularized-alignment-for-domain-adaptation

## 1. Data Generation
### 1.1 Data
MNIST already in ./data/dataset/MNIST/ (http://yann.lecun.com/exdb/mnist/)

Download from (http://ufldl.stanford.edu/housenumbers/ **Format2**) and put the two .mat files in ./data/dataset/SVHN/ 

### 1.2 Convert to LMDB format
Run ./data/data_utils.py, this will create "./data/dataset/MNIST/lmdb" and "./data/dataset/SVHN/lmdb". If "mdb_put: MDB_MAP_FULL: Environment mapsize limit reached" is encountered, consider increasing the "map_size" (line 87 data_utils.py) to 3e9. 

## 2. Prerequisite
* python 3
* PyTorch 1.1
* tensorboardX
* numpy
* skimage
* tqdm

## 3.
