import os

MODEL_INDEX = [9]   # Integer or list
"""
0: M -> MM
1: S -> MM
2: M -> MM, VADA
3: S -> MM, VADA
4: MM -> M  (TODO)
5: MM -> S
6: M -> MM, noadapt  (TODO)
7: MM -> M, noadapt  (TODO)
8: MM -> M, VADA  (TODO)
"""

SOURCES = {4: 'mnist-m', 6: 'mnist', 7: 'mnist-m', 8: 'mnist-m'}
TARGETS = {4: 'mnist', 6: 'mnist-m', 7: 'mnist', 8: 'mnist'}
VADAS = {4: False, 6: False, 7: False, 8: True}
NOADAPTS = {4: False, 6: True, 7: True, 8: False}

class Configs:
    def __init__(self, model_index=MODEL_INDEX):
        self.gpu = True
        self.model_index = model_index
        self.data_source = SOURCES.get(model_index, 'mnist')  # 'mnist' if model_index % 2 == 0 else 'svhn' #'mnist', 'svhn', 'mnist-m'
        self.data_target = TARGETS.get(model_index, 'mnist-m')
        self.total_epoch = 20 # 80 for batch 64
        self.batch_size = 64
        self.lr = 1e-3
        self.run_VADA = VADAS.get(model_index, False) # Whether to run this as plan VADA: 
                              # If True, automatically ignore lambda_div and lambda_agree
        self.run_noadapt = NOADAPTS.get(model_index, False) # Whether to run this as no adaptation: Train on source only, then test on target
        self.lambda_closs = 5 # source cross entropy loss
        self.lambda_dom = 1 # discriminator
        self.lambda_ent = 1e-2 # conditional entropy
        self.lambda_div = 1e-3 # co-regularized divergence (ignored with VADA)
        self.div_margin = 10 # co-regularized divergence
        self.lambda_agree = 1e-1 # co-regularized agreement (ignored with VADA)
        self.vat_epsilon = 1.0 # vat epsilon

        self.ins_norm = True
        self.save_path = './model_{}'.format(self.model_index)
        #self.source_lmdb = './data/dataset/MNIST/lmdb'
        #self.target_lmdb = './data/dataset/SVHN/lmdb'
        self.source_lmdb = os.path.join('./data/dataset', self.data_source.upper(), 'lmdb')  # Won't work for "SynNumbers", but whatever
        self.target_lmdb = os.path.join('./data/dataset', self.data_target.upper(), 'lmdb')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.mode = 0 # 0 for training, 1 for testing
        self.checkpoint = './model_0/model_best_0_0'

    def dump_to_file(self, path):
        with open(path, 'a+') as writer:
            writer.write('mode: {}\nmodel_index:{}\nsource: {}\ntarget: {}\n'.format(self.mode,\
                 self.model_index, self.data_source, self.data_target))
            if self.mode == -1:
                writer.write('checkpoint: {}'.format(self.checkpoint))
            writer.write('save_path: {}\n'.format(self.save_path))

            writer.write('total_epoch: {}\nbatch_size: {}\nlr: {}\n'.format(self.total_epoch, self.batch_size, self.lr))
            writer.write('lambda_dom: {}\n'.format(self.lambda_dom))
            writer.write('lambda_ent: {}\n'.format(self.lambda_ent))
            writer.write('vat_epsilon: {}\n'.format(self.vat_epsilon))
            if self.lambda_div != 0:
                writer.write('lambda_div: {}\n'.format(self.lambda_div))
                writer.write('div_margin: {}\n'.format(self.div_margin))
                writer.write('lambda_agree: {}\n'.format(self.lambda_agree))