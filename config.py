import os

MODEL_INDEX = 0

class Configs:
    def __init__(self, model_index=MODEL_INDEX):
        self.gpu = True
        self.data_source = 'mnist' #'mnist', 'svhn', 'mnist-m'
        self.data_target = 'svhn'
        self.total_epoch = 80 # 80 for batch 64
        self.batch_size = 64
        self.lr = 1e-3
        self.run_VADA = False # Whether to run this as plan VADA: 
                              # If True, automatically ignore lambda_div and lambda_agree
        self.lambda_closs = 5 # source cross entropy loss
        self.lambda_dom = 1 # discriminator
        self.lambda_ent = 1e-2 # conditional entropy
        self.lambda_div = 1e-3 # co-regularized divergence (ignored with VADA)
        self.div_margin = 10 # co-regularized divergence
        self.lambda_agree = 1e-1 # co-regularized agreement (ignored with VADA)
        self.vat_epsilon = 1.0 # vat epsilon

        self.model_index = model_index
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