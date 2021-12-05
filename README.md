# co-regularied-alignment-for-domain-adaptation

Original repository: https://github.com/jakc4103/co-regularized-alignment-for-domain-adaptation

## 1. Prerequisite
* python 3
* PyTorch
* tensorboardX
* numpy
* PIL
* tqdm
  
## 2. Data Generation
### 2.1 Data
MNIST already in ./data/dataset/MNIST/ (http://yann.lecun.com/exdb/mnist/)

Download from (http://ufldl.stanford.edu/housenumbers/ **Format2**) and put the two .mat files in ./data/dataset/SVHN/ 

MNIST-M is downloaded at runtime from https://github.com/liyxi/mnist-m/. You do not need to download it before running ``data_utils.py``.

### 2.2 Convert to LMDB format
Run ./data/data_utils.py, this will create "./data/dataset/MNIST/lmdb", "./data/dataset/SVHN/lmdb" and "./data/dataset/MNIST-M/lmdb". It will also download the MNIST-M dataset if it does not exist.

If "mdb_put: MDB_MAP_FULL: Environment mapsize limit reached" is encountered, consider increasing the "map_size" (line 87 data_utils.py) to 3e9. 

