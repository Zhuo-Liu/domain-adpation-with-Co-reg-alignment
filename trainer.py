import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from loss_function.loss_co_reg import diverse_loss, agreement_loss, cross_entropy_loss
from loss_function.loss_vat import VATLoss
import copy
import time

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, nets, train_data_loader, val_data_loader, test_data_loader, config):
        if config.gpu:
            self.nets = [net.cuda() for net in nets]
        else:
            self.nets = nets

        self.train_data = train_data_loader
        self.val_data = val_data_loader
        self.test_data = test_data_loader
        if self.train_data is not None:
            self.len_train_data = len(train_data_loader)
        self.config = config

        self.exp_net = copy.deepcopy(self.nets)

        if self.train_data is not None:
            self.writer = SummaryWriter(os.path.join(config.save_path, 'tensorboard', 'exp_' + str(int(time.time()))))

        self.optim = []
        self.optim_dom = []
        self.dom_criterion = [] # loss of domain alignment for each classifier
        self.dom_inv_criterion = []
        self.criterion = [] # Cross entropy for each classifier
        self.vat_criterion = [] # Virtual adversarial loss for each classifier

        for net in self.nets:
            category_param, dom_param = self.get_param_list(net)

            self.optim.append(torch.optim.Adam(category_param, lr=config.lr, betas=(0.5, 0.999)))
            self.optim_dom.append(torch.optim.Adam(dom_param, lr=config.lr, betas=(0.5, 0.999)))
    
            self.dom_criterion.append(torch.nn.BCEWithLogitsLoss())
            self.dom_inv_criterion.append(torch.nn.BCEWithLogitsLoss())
            self.criterion.append(torch.nn.CrossEntropyLoss())
            self.vat_criterion.append(VATLoss(xi=1e-6, eps=3.5))


    def get_param_list(self, net):
        category_param = []
        dom_param = []
        for name, param in net.named_parameters():
            if 'FE' in name or 'CC' in name:
                #print("CC", name)
                category_param.append(param)
            elif 'DD' in name:
                #print("DD", name)
                dom_param.append(param)
            else:
                assert False, 'PARAM NAME NOT FOUND'

        return category_param, dom_param


    def train_all(self):
        acc_best = [0] * len(self.exp_net)
        for idx, net in enumerate(self.exp_net):
            _ = self.val_(self.exp_net[idx], -1, idx, mode='plot')
        
        for epoch in range(self.config.total_epoch):
            self.train_1_epoch(epoch)
            #self.save_net(self.exp_net, 'model_'+str(epoch+1))

            for idx, net in enumerate(self.exp_net):
                self.save_net(self.exp_net[idx], 'model_'+str(epoch+1), idx)
                acc = self.val_(self.exp_net[idx], epoch, idx)
                if acc > acc_best[idx]:
                    self.save_net(self.exp_net[idx], 'model_best_{}'.format(idx), idx)
                    acc_best[idx] = acc

                _ = self.val_(self.exp_net[idx], epoch, idx, mode='test')
                _ = self.val_(self.exp_net[idx], epoch, idx, mode='plot')


    def train_1_epoch(self, epoch):
        for idx in range(2):
            self.nets[idx].train()
        pbar = tqdm(enumerate(self.train_data), total=len(self.train_data), ncols=100)
        for step, batchSample in pbar:
            simage = batchSample['simage'].float()
            slabel = batchSample['slabel'].long()

            timage = batchSample['timage'].float()
            tlabel = batchSample['tlabel'].long()

            dom_label = torch.cat([torch.zeros(simage.size(0)), torch.ones(timage.size(0))]).view(-1, 1)
            dom_inv_label = torch.cat([torch.ones(simage.size(0)), torch.zeros(timage.size(0))]).view(-1, 1)

            if self.config.gpu:
                simage = simage.cuda()
                slabel = slabel.cuda()
                timage = timage.cuda()
                tlabel = tlabel.cuda()
                dom_label = dom_label.cuda()
                dom_inv_label = dom_inv_label.cuda()

            slogit_list = []
            tlogit_list = []
            
            dom_logit_list = []

            feat_list = []
            prob_list = []
            
            svat_loss_list = []
            tvat_loss_list = []
            #images_iter = torch.cat([simage, timage], 0)
            for idx, net in enumerate(self.nets): # inference for all nets
                tvat_loss, timage_vat = self.vat_criterion[idx](self.nets[idx], timage)
                svat_loss, simage_vat = self.vat_criterion[idx](self.nets[idx], simage)

                svat_loss_list.append(svat_loss)
                tvat_loss_list.append(tvat_loss)

                slogits, sdom_logits, sfeats = self.nets[idx](simage)  # sfeats are features: id * channel * height * width
                tlogits, tdom_logits, _ = self.nets[idx](timage, update_stats=True)

                slogit_list.append(slogits)
                tlogit_list.append(tlogits)

                prob_list.append(torch.nn.functional.softmax(tlogits, 1))

                dom_logit_list.append(torch.cat([sdom_logits, tdom_logits], dim=0))
                
                feat_list.append(sfeats)

            agree_loss = 0
            div_loss = 0
            
            for i in range(len(feat_list) - 1):
                for j in range(i + 1, len(feat_list)):
                    # diversoty loss
                    tmp_div_loss = diverse_loss(feat_list[i], feat_list[j], self.config.div_margin)
                    div_loss = tmp_div_loss + div_loss

                    # agree_loss
                    tmp_agree_loss = agreement_loss(prob_list[i], prob_list[j])
                    agree_loss = agree_loss + tmp_agree_loss 

            self.writer.add_scalar('div_loss', float(div_loss), self.len_train_data * epoch + step)
            self.writer.add_scalar('agree_loss', float(agree_loss), self.len_train_data * epoch + step)

            for idx in range(len(slogit_list)):                
                closs = self.criterion[idx](slogit_list[idx], slabel)
                ent_loss = cross_entropy_loss(tlogit_list[idx])
                dom_loss = self.dom_criterion[idx](dom_logit_list[idx], dom_label) * 0.5
                dom_inv_loss = self.dom_inv_criterion[idx](dom_logit_list[idx], dom_inv_label) * 0.5
                
                # VADA loss:
                # closs: L_y(f_i;P_s)
                # dom_loss: L_d(g_i#P_s,g_i#P_t) * 0.5 ?
                # ent_loss: L_ce(f_i;P_t)
                # svat_loss_list[i]: L_vt(f_i;P_s)
                # tvat_loss_list[i]: L_vt(f_i;P_t)
                # Co-DA loss:
                # div_loss: D_g(g_1,g_2)
                # agree_loss: L_p(f_1,f_2;P_t)
                CCloss = self.config.lambda_closs * closs + self.config.lambda_dom * dom_loss + self.config.lambda_ent * ent_loss \
                    - (self.config.lambda_div if not self.config.run_VADA else 0) * div_loss \
                    + (self.config.lambda_agree if not self.config.run_VADA else 0) * agree_loss \
                    + svat_loss_list[idx] + tvat_loss_list[idx] * self.config.lambda_ent
                #CCloss = (5*closs + svat_loss_list[idx]) + 1e-2*(ent_loss +  tvat_loss_list[idx]) + 1*dom_loss + 0.0 * agree_loss - 0.0 * div_loss
                if self.config.run_noadapt:
                    CCloss = self.config.lambda_closs * closs
                DDloss = dom_inv_loss

                self.writer.add_scalar('net_{}/closs'.format(idx), float(closs), self.len_train_data * epoch + step)
                self.writer.add_scalar('net_{}/ent_loss'.format(idx), float(ent_loss), self.len_train_data * epoch + step)
                self.writer.add_scalar('net_{}/dom_loss'.format(idx), float(dom_loss), self.len_train_data * epoch + step)
                self.writer.add_scalar('net_{}/dom_inv_loss'.format(idx), float(dom_inv_loss), self.len_train_data * epoch + step)
                self.writer.add_scalar('net_{}/CCloss'.format(idx), float(CCloss), self.len_train_data * epoch + step)
                self.writer.add_scalar('net_{}/svat_loss'.format(idx), float(svat_loss_list[idx]), self.len_train_data * epoch + step)
                self.writer.add_scalar('net_{}/tvat_loss'.format(idx), float(tvat_loss_list[idx]), self.len_train_data * epoch + step)

                optim = self.optim[idx]
                optim_dom = self.optim_dom[idx]

                optim.zero_grad()
                CCloss.backward(retain_graph=True)
                optim_dom.zero_grad()
                DDloss.backward(retain_graph=True)

            for idx in range(len(slogit_list)):        
                optim = self.optim[idx]
                optim_dom = self.optim_dom[idx]   
                optim.step()
                optim_dom.step()

                self.update_exp_net(idx)

            pbar.set_description("CCloss %5f DDloss %5f epoch %d" % \
                (round(float(CCloss), 5), round(float(DDloss), 5), epoch))

            if step == len(self.train_data) - 1:# save visualization vat images
                tmp_img = (((simage + 1. ) / 2. ))
                self.writer.add_images('original_source', tmp_img, epoch)

                tmp_img = (((timage + 1. ) / 2. ))
                self.writer.add_images('original_target', tmp_img, epoch)

                tmp_img = (((simage_vat + 1. ) / 2. ))
                self.writer.add_images('vat_source', tmp_img, epoch)

                tmp_img = (((timage_vat + 1. ) / 2. ))
                self.writer.add_images('vat_target', tmp_img, epoch)


    def val_(self, net, epoch, idx, mode='val'):  # mode = 'val', 'test', 'plot'
        net = net.eval()
        loss = []
        total_correct = 0
        total_data = 0
        total_correct2 = 0
        total_data2 = 0
        data = (self.val_data if mode == 'val' 
                else self.test_data if mode == 'test'
                else self.val_data)  # 'plot'
        pbar = tqdm(enumerate(data), total=len(data), ncols=100)

        # For plotting
        sfeats_all = []
        tfeats_all = []
        slabels_all = []
        tlabels_all = []

        with torch.no_grad():
            for step, batchSample in pbar:

                simage = batchSample['simage'].float()
                slabel = batchSample['slabel'].long()

                timage = batchSample['timage'].float()
                tlabel = batchSample['tlabel'].long()

                dom_label = torch.cat([torch.zeros(simage.size(0)), torch.ones(timage.size(0))]).view(-1, 1)
                dom_inv_label = torch.cat([torch.ones(simage.size(0)), torch.zeros(timage.size(0))]).view(-1, 1)

                if self.config.gpu:
                    simage = simage.cuda()
                    slabel = slabel.cuda()
                    timage = timage.cuda()
                    tlabel = tlabel.cuda()
                    dom_label = dom_label.cuda()
                    dom_inv_label = dom_inv_label.cuda()

                #logits, dom_logits, _ = net(torch.cat([simage, timage], 0))
                logits, dom_logits, feats = net(torch.cat([simage, timage], 0))  # Input is source images followed by target images
                #print(feats.shape)

                closs = self.criterion[idx](logits[:simage.size(0)], slabel).cpu()
                ent_loss = cross_entropy_loss(logits[simage.size(0):]).cpu()
                
                dom_loss = self.dom_criterion[idx](dom_logits, dom_label).cpu()
                dom_inv_loss = self.dom_inv_criterion[idx](dom_logits, dom_inv_label).cpu()
                CCloss = closs + self.config.lambda_dom * dom_loss

                loss.append([closs, dom_loss, dom_inv_loss, CCloss, ent_loss])

                num_correct, num_all, _ = self.cal_accuracy(logits[simage.size(0):], tlabel)
                total_correct += num_correct
                total_data += num_all

                num_correct2, num_all2, _ = self.cal_accuracy(logits[:simage.size(0)], slabel)
                total_correct2 += num_correct2
                total_data2 += num_all2

                pbar.set_description(mode+(" acc: %5f epoch %d" % (round(float(num_correct) / num_all, 5), epoch)))

                if mode == 'plot':
                    sfeats_all.append(feats[:simage.size(0)])
                    tfeats_all.append(feats[simage.size(0):])
                    slabels_all.extend(slabel)
                    tlabels_all.extend(tlabel)

            acc = float(total_correct) / total_data
            acc2 = float(total_correct2) / total_data2
            if mode != 'plot':
                print('accuracy on source data is:', acc2)
                print('accuracy on target data is:', acc)
                if self.train_data is not None:
                    self.writer.add_scalar('net_{}/'.format(idx)+mode+'/acc', acc, epoch)

            loss = np.mean(np.array(loss), 0)

            if self.train_data is not None and mode != 'plot':
                self.writer.add_scalar('net_{}/'.format(idx)+mode+'/closs', float(loss[0]), epoch)
                self.writer.add_scalar('net_{}/'.format(idx)+mode+'/dom_loss', float(loss[1]), epoch)
                self.writer.add_scalar('net_{}/'.format(idx)+mode+'/dom_inv_loss', float(loss[2]), epoch)
                self.writer.add_scalar('net_{}/'.format(idx)+mode+'/CCloss', float(loss[3]), epoch)
                self.writer.add_scalar('net_{}/'.format(idx)+mode+'/entloss', float(loss[4]), epoch)
                #self.writer.add_scalar('net_{}/'.format(idx)+mode+'/vatloss', float(loss[5]), epoch)

            if mode == 'plot':
                sfeats = torch.cat(sfeats_all, 0)
                sfeats = torch.flatten(sfeats, start_dim=1)
                tfeats = torch.cat(tfeats_all, 0)
                tfeats = torch.flatten(tfeats, start_dim=1)
                #print(sfeats.shape, len(slabels_all))
                self.plot(sfeats, slabels_all, tfeats, tlabels_all, 
                          filename=('net_{0}_epoch_{1}.png'.format(idx, epoch)),
                          title='Net {0}, Epoch {1}'.format(idx, epoch),
                          class_num=10)
                
            return acc


    def cal_accuracy(self, logits, label):
        prob = torch.nn.functional.softmax(logits, 1)
        max, pred = torch.max(prob, 1)

        num_correct = torch.sum(pred == label)

        return int(num_correct), logits.size(0), float(num_correct) / logits.size(0)


    def save_net(self, net, name, idx):
        torch.save(net.state_dict(), os.path.join(self.config.save_path, name + '_' + str(idx)))


    def getfiles(self, dirpath):
        a = [s for s in os.listdir(dirpath)
            if os.path.isfile(os.path.join(dirpath, s)) and 'model_' in s and 'cur' not in s and 'best' not in s]
        a.sort(key=lambda s: int(s.split('_')[-1]))
        return a


    def update_exp_net(self, idx, momentum=0.002):
        new_state_dict = self.nets[idx].state_dict()
        old_state_dict = self.exp_net[idx].state_dict()
        for key in old_state_dict:
            if(old_state_dict[key].dtype != torch.int64): # don't update batch number!!
                old_state_dict[key].mul_(1 - momentum).add_(momentum * new_state_dict[key])
            else:
                old_state_dict[key] = new_state_dict[key]

        self.exp_net[idx].load_state_dict(old_state_dict)

    def plot(self, sdata, slabel, tdata, tlabel, filename='tsne_untitled.png',
             title='tsne plot of certain data', class_num=10):
        # Source in dark color, target in light color
        # sdata and tdata are 2D matrices
        tsne = TSNE(n_components=2, verbose=0, random_state=123)
        data = torch.cat([sdata, tdata], 0).cpu()
        slabel = [x.cpu() for x in slabel]
        tlabel = [x.cpu() for x in tlabel]
        Embedded = tsne.fit_transform(data)

        #print("Plot!")
        # Plot source
        df1 = pd.DataFrame()
        df1["y"] = [('s' + str(x.numpy())) for x in slabel]
        df1["comp-1"] = Embedded[:sdata.size(0), 0]
        df1["comp-2"] = Embedded[:sdata.size(0), 1]
        df1.sort_values(by=['y'], inplace=True)  # So that labels appear in order
        sns.scatterplot(x="comp-1", y="comp-2", hue=df1.y.tolist(), palette=sns.color_palette("deep", n_colors=class_num),
                        data=df1).set(title=title)

        # Plot target
        df2 = pd.DataFrame()
        df2["y"] = [('t' + str(x.numpy())) for x in tlabel]
        df2["comp-1"] = Embedded[sdata.size(0):, 0]
        df2["comp-2"] = Embedded[sdata.size(0):, 1]
        df2.sort_values(by=['y'], inplace=True)
        sns.scatterplot(x="comp-1", y="comp-2", hue=df2.y.tolist(), palette=sns.color_palette("pastel", n_colors=class_num),
                        data=df2).set(title=title)

        # Fix legends
        #handles, labels = plt.gca().get_legend_handles_labels()
        #order = [(i//2)+(i%2)*10 for i in range(20)]
        #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize='xx-small')
        plt.legend(ncol=2, fontsize='xx-small')

        os.makedirs(os.path.join(self.config.save_path, 'plots'), exist_ok=True)
        filename = os.path.join(self.config.save_path, 'plots', filename)
        plt.savefig(filename)
        plt.clf()