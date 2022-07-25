import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from collections import defaultdict
import heapq
import math
import csv
import codecs
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
import pandas as pd
from rrl.components import BinarizeLayer
from rrl.components import UnionLayer, LRLayer,weight_net
from rrl.utils import cal_ndcg,cal_auc

LOG_CNT_MOD = 2000
TEST_CNT_MOD = 500


class MLLP_bottom(nn.Module):
    def __init__(self, dim_list, use_not=False, left=None, right=None, estimated_grad=False):
        super(MLLP_bottom, self).__init__()

        self.dim_list = dim_list
        self.use_not = use_not
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])

        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            if i >= 4:
                num += self.layer_list[-2].output_dim

            if i == 1:
                layer = BinarizeLayer(dim_list[i], num, self.use_not, self.left, self.right)
                layer_name = 'binary{}'.format(i)
            else:
                layer = UnionLayer(dim_list[i], num, estimated_grad=estimated_grad)
                layer_name = 'union{}'.format(i)
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)
            self.last_layer_dim = layer.output_dim

    def forward(self, x):
        return self.continuous_forward(x), self.binarized_forward(x)

    def continuous_forward(self, x):
        x_res = None
        for i, layer in enumerate(self.layer_list):
            if i <= 1:
                x = layer(x)
            else:
                x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                x_res = x
                x = layer(x_cat)
        return x

    def binarized_forward(self, x):
        """Equivalent to using the extracted Concept Rule Sets."""
        with torch.no_grad():
            x_res = None
            for i, layer in enumerate(self.layer_list):
                if i <= 1:
                    x = layer.binarized_forward(x)
                else:
                    x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                    x_res = x
                    x = layer.binarized_forward(x_cat)
            return x

class MLLP_top(nn.Module):
    def __init__(self, dim_list, bottom_structure_list, use_not=False, left=None, right=None, estimated_grad=False):
        super(MLLP_top, self).__init__()

        self.dim_list = dim_list
        self.use_not = use_not
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])

        prev_layer_dim = 0 
        for bottom_structure in bottom_structure_list:
            prev_layer_dim += list(map(int, bottom_structure.split('@')))[-1]
        prev_layer_dim *=2
        layer_dim_list=[prev_layer_dim]
        for i in range(len(dim_list)):
            num = prev_layer_dim
            if i >= 1:
                num += layer_dim_list[-2]
            layer = UnionLayer(dim_list[i], num, estimated_grad=estimated_grad)
            layer_name = 'top_union{}'.format(i)
            prev_layer_dim = layer.output_dim
            layer_dim_list.append(prev_layer_dim)
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def forward(self, continuous_output, binarized_output):
        x_res_continuous = None
        for i, layer in enumerate(self.layer_list):
            x_cat = torch.cat([continuous_output, x_res_continuous], dim=1) if x_res_continuous is not None else continuous_output
            x_res_continuous = continuous_output
            continuous_output = layer(x_cat)

        with torch.no_grad():
            x_res_binarized = None
            for i, layer in enumerate(self.layer_list):
                x_cat = torch.cat([binarized_output, x_res_binarized], dim=1) if x_res_binarized is not None else binarized_output
                x_res_binarized = binarized_output
                binarized_output = layer.binarized_forward(x_cat)
        return continuous_output,binarized_output


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list

class MLLP_tower(nn.Module):
    def __init__(self,item_num,user_num,embedding_dim, db_enc_list, bottom_structure_list, top_structure, device_id, use_not=False, left=None, right=None, estimated_grad=False, distributed=True):
        super(MLLP_tower, self).__init__()
        self.db_enc_list = db_enc_list
        self.bottom_structure_list = bottom_structure_list
        self.top_structure = top_structure

        self.device_id = device_id
        self.use_not = use_not
        self.left = left
        self.right = right

        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0).cuda(self.device_id)
        self.user_embedding = nn.Embedding(user_num + 1, embedding_dim, padding_idx=0).cuda(self.device_id)

        self.bottom_net_list = nn.ModuleList([])
        for db_enc, bottom_structure in zip(self.db_enc_list, self.bottom_structure_list):
            discrete_flen = db_enc.discrete_flen
            continuous_flen = db_enc.continuous_flen
            dim_list = [(discrete_flen, continuous_flen)] + list(map(int, bottom_structure.split('@')))
            net = MLLP_bottom(dim_list, use_not=use_not, left=left, right=right, estimated_grad=estimated_grad)
            
            net.cuda(self.device_id)
            if distributed:
                net = MyDistributedDataParallel(net, device_ids=[self.device_id])

            self.bottom_net_list.append(net)

        self.botton_layer_dim = self.bottom_net_list[0].layer_list[-1].output_dim
        if len(top_structure) > 0:
            self.top_net = MLLP_top(list(map(int, top_structure.split('@'))),
                                    bottom_structure_list=self.bottom_structure_list,
                                    use_not=use_not, left=left, right=right, estimated_grad=estimated_grad)
        self.top_net.cuda(self.device_id)
        if distributed:
            self.top_net = MyDistributedDataParallel(self.top_net, device_ids=[self.device_id])

        self.final_weight0=weight_net(embedding_dim*2,self.top_net.layer_list[-1].output_dim).cuda(self.device_id)
        self.final_weight1 =weight_net(embedding_dim*2,self.top_net.layer_list[-1].output_dim).cuda(self.device_id)

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for bottom_net in self.bottom_net_list:
            for layer in bottom_net.layer_list[:]:
                layer.clip()
        for layer in self.top_net.layer_list[:]:
            layer.clip()

    def forward(self, u,i,x_list):

        user_idx = u
        item_idx = i
        user_vec = self.user_embedding(user_idx).cuda(self.device_id, non_blocking=True)
        item_vec = self.item_embedding(item_idx).cuda(self.device_id, non_blocking=True)

        bottom_output_list=[]
        for x, botton_net in zip(x_list, self.bottom_net_list):
            bottom_output_list.append(botton_net.forward(x))

        bottom_continuous, bottom_binarized = torch.Tensor().cuda(self.device_id, non_blocking=True), torch.Tensor().cuda(self.device_id, non_blocking=True)
        for bottom_output in bottom_output_list:
            bottom_continuous = torch.cat((bottom_continuous,bottom_output[0]),1)
            bottom_binarized = torch.cat((bottom_binarized,bottom_output[1]),1)

        top_continuous,top_binarized=self.top_net.forward(bottom_continuous.float(), bottom_binarized.float())

        ui_vec=torch.cat((user_vec,item_vec),1)
        final_weight0=self.final_weight0(ui_vec)
        final_weight1 = self.final_weight1(ui_vec)
        
        continuous0=torch.matmul(top_continuous,final_weight0.T)
        continuous0 = torch.diag(continuous0)
        continuous0=continuous0.reshape(continuous0.shape[0], 1)
        continuous1=torch.matmul(top_continuous, final_weight1.T)
        continuous1 = torch.diag(continuous1)
        continuous1 = continuous1.reshape(continuous1.shape[0], 1)

        binarized0 = torch.matmul(top_binarized, final_weight0.T)
        binarized0 = torch.diag(binarized0)
        binarized0 = binarized0.reshape(binarized0.shape[0], 1)
        binarized1 = torch.matmul(top_binarized, final_weight1.T)
        binarized1 = torch.diag(binarized1)
        binarized1 = binarized1.reshape(binarized1.shape[0], 1)
        
        return torch.cat((continuous0,continuous1),1),torch.cat((binarized0,binarized1),1)




class RRL:
    def __init__(self,item_num,user_num,embedding_dim, db_enc_list, bottom_structure_list, top_structure,  topk, device_id, use_not=False, is_rank0=False, log_file=None,
                 writer=None, left=None, right=None, save_best=False, estimated_grad=False, save_path=None, distributed=True):
        super(RRL, self).__init__()
        self.db_enc_list = db_enc_list
        self.bottom_structure_list=bottom_structure_list
        self.top_structure=top_structure
        self.use_not = use_not

        self.best_AUC = -1.
        self.best_NDCG10 = -1.

        self.topk=topk
        self.device_id = device_id
        self.is_rank0 = is_rank0
        self.save_best = save_best
        self.estimated_grad = estimated_grad
        self.save_path = save_path
        if self.is_rank0:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
            if log_file is None:
                logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
            else:
                logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
        self.writer = writer

        self.net=MLLP_tower(item_num=item_num,user_num=user_num,embedding_dim=embedding_dim,
            db_enc_list=db_enc_list, bottom_structure_list=bottom_structure_list, top_structure=top_structure,
            device_id=device_id, use_not=use_not, left=left, right=right, estimated_grad=estimated_grad, distributed=distributed)
        self.net.cuda(self.device_id)

    def data_transform(self, X, y):
        X = X.astype(np.float)
        if y is None:
            return torch.tensor(X)
        y = y.astype(np.float)
        return torch.tensor(X), torch.tensor(y)

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer


    def train_model(self, X=None, y=None, X_validation=None, y_validation=None, data_loader_list=None, valid_loader_list=None,
                    user=None,item=None,valid_user=None,valid_item=None,epoch=50, lr=0.01, lr_decay_epoch=100, lr_decay_rate=0.75, batch_size=64, weight_decay=0.0,
                    log_iter=50):

        if (X is None or y is None) and data_loader_list is None:
            raise Exception("Both data set and data loader are unavailable.")
        if data_loader_list is None:
            X, y = self.data_transform(X, y)
            if X_validation is not None and y_validation is not None:
                X_validation, y_validation = self.data_transform(X_validation, y_validation)
            data_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        loss_log = []
        AUC,AUC_b,NDCG5,NDCG5_b,NDCG10,NDCG10_b = [],[],[],[],[],[]

        criterion = nn.CrossEntropyLoss().cuda(self.device_id)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        cnt = -1
        avg_batch_loss_mllp = 0.0
        avg_batch_loss_rrl = 0.0
        epoch_histc = defaultdict(list)
        for epo in range(epoch):
            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate,
                                              lr_decay_epoch=lr_decay_epoch)
            epoch_loss_mllp = 0.0
            epoch_loss_rrl = 0.0
            abs_gradient_max = 0.0
            abs_gradient_avg = 0.0

            ba_cnt = 0
            for data_loader_zip in zip(*data_loader_list,user,item):
                ba_cnt += 1
                X_list=[]
                for data in data_loader_zip[:-2]:
                    X = data[0].cuda(self.device_id, non_blocking=True)
                    y = data[1].cuda(self.device_id, non_blocking=True)
                    X_list.append(X)
                optimizer.zero_grad()  # Zero the gradient buffers.
                u,i=data_loader_zip[-2].cuda(self.device_id, non_blocking=True),data_loader_zip[-1].cuda(self.device_id, non_blocking=True)
                y_pred_mllp, y_pred_rrl = self.net.forward(u,i,X_list)
                with torch.no_grad():
                    y_prob = torch.softmax(y_pred_rrl, dim=1)
                    y_arg = torch.argmax(y, dim=1)
                    loss_mllp = criterion(y_pred_mllp, y_arg)
                    loss_rrl = criterion(y_pred_rrl, y_arg)
                    ba_loss_mllp = loss_mllp.item()
                    ba_loss_rrl = loss_rrl.item()
                    epoch_loss_mllp += ba_loss_mllp
                    epoch_loss_rrl += ba_loss_rrl
                    avg_batch_loss_mllp += ba_loss_mllp
                    avg_batch_loss_rrl += ba_loss_rrl
                y_pred_mllp.backward((y_prob - y) / y.shape[0])  # for CrossEntropy Loss
                cnt += 1
                optimizer.step()

                if self.is_rank0:
                    for i, param in enumerate(self.net.parameters()):
                        abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                        abs_gradient_avg += torch.sum(torch.abs(param.grad)) / (param.grad.numel())

                self.net.clip()

                if self.is_rank0 and cnt % TEST_CNT_MOD == 0:
                    if X_validation is not None and y_validation is not None:
                        auc,auc_b,ndcg5,ndcg5_b,ndcg10,ndcg10_b = self.test(X_validation, y_validation,valid_user,valid_item,
                                                        batch_size=batch_size,need_transform=False, set_name='Validation')
                    elif valid_loader_list is not None:
                        auc,auc_b,ndcg5,ndcg5_b,ndcg10,ndcg10_b = self.test(test_loader_list=valid_loader_list,
                                                        user=valid_user,item=valid_item,need_transform=False, set_name='Validation')
                    elif data_loader_list is not None:
                        auc,auc_b,ndcg5,ndcg5_b,ndcg10,ndcg10_b = self.test(test_loader_list=data_loader_list,
                                                        user=valid_user,item=valid_item,need_transform=False, set_name='Training')
                    else:
                        auc,auc_b,ndcg5,ndcg5_b,ndcg10,ndcg10_b = self.test(X, y,valid_user,valid_item,
                                                        batch_size=batch_size, need_transform=False,set_name='Training')
                    if ndcg10_b>self.best_NDCG10:
                        self.best_NDCG10=ndcg10_b
                        self.save_model()
                    AUC.append(auc)
                    AUC_b.append(auc_b)
                    NDCG5.append(ndcg5)
                    NDCG5_b.append(ndcg5_b)
                    NDCG10.append(ndcg10)
                    NDCG10_b.append(ndcg10_b)
                    if self.writer is not None:
                        self.writer.add_scalar('AUC_MLLP', auc, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('AUC_RRL', auc_b, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('NDCG5_MLLP', ndcg5, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('NDCG5_RRL', ndcg5_b, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('NDCG10_MLLP', ndcg10, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('NDCG10_RRL', ndcg10_b, cnt // TEST_CNT_MOD)
            if self.is_rank0:
                logging.info('epoch: {}'.format(epo))
                for name, param in self.net.named_parameters():
                    maxl = 1 if 'con_layer' in name or 'dis_layer' in name else 0
                    epoch_histc[name].append(torch.histc(param.data, bins=10, max=maxl).cpu().numpy())
                if self.writer is not None:
                    self.writer.add_scalar('Training_Loss_MLLP', epoch_loss_mllp, epo)
                    self.writer.add_scalar('Training_Loss_RRL', epoch_loss_rrl, epo)
                    self.writer.add_scalar('Abs_Gradient_Max', abs_gradient_max, epo)
                    self.writer.add_scalar('Abs_Gradient_Avg', abs_gradient_avg / ba_cnt, epo)
                loss_log.append(epoch_loss_rrl)
        if self.is_rank0 and not self.save_best:
            self.save_model()
        return epoch_histc

    def test(self, X=None, y=None, test_loader_list=None, user=None, item=None,  batch_size=32, need_transform=True, set_name='Validation'):
        if X is not None and y is not None and need_transform:
            X, y = self.data_transform(X, y)
        with torch.no_grad():
            if X is not None and y is not None:
                test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

            y_true = np.array([])
            for X, y in test_loader_list[0]:
                y_true=np.concatenate((y_true,np.array(y[:,1]).squeeze()))

            user_list =  np.array([])
            for u in user:
                user_list = np.concatenate((user_list, np.array(u).squeeze()))

            y_pred_list = torch.as_tensor([]).cuda(self.device_id, non_blocking=True)
            y_pred_b_list = torch.as_tensor([]).cuda(self.device_id, non_blocking=True)
            for test_loader_zip in zip(*test_loader_list,user,item):
                X_list=[]
                for data in test_loader_zip[:-2]:
                    X = data[0].cuda(self.device_id, non_blocking=True)
                    X_list.append(X)
                u, i = test_loader_zip[-2].cuda(self.device_id, non_blocking=True), test_loader_zip[-1].cuda(self.device_id, non_blocking=True)
                y_pred_mllp, y_pred_rrl = self.net.forward(u,i,X_list)
                y_pred_mllp = torch.softmax(y_pred_mllp, dim=1)
                y_pred_rrl = torch.softmax(y_pred_rrl, dim=1)
                y_pred_list=torch.cat([y_pred_list,y_pred_mllp.squeeze()])
                y_pred_b_list=torch.cat([y_pred_b_list,y_pred_rrl.squeeze()])
            y_pred_list=y_pred_list.narrow(1,1,1).squeeze().cpu()
            y_pred_b_list=y_pred_b_list.narrow(1,1,1).squeeze().cpu()

            auc=cal_auc(y_pred_list, y_true, user_list)
            auc_b=cal_auc(y_pred_b_list, y_true, user_list)

            ndcg5 =cal_ndcg(y_pred_list, y_true, user_list, 5)
            ndcg5_b =cal_ndcg(y_pred_b_list, y_true, user_list, 5)

            ndcg10 = cal_ndcg(y_pred_list, y_true, user_list, 10)
            ndcg10_b = cal_ndcg(y_pred_b_list, y_true, user_list, 10)
            
            logging.debug('y_mllp: {} '.format(len(y_pred_list),))

            logging.debug('y_rrl: {} '.format(len(y_pred_b_list),))

            logging.info('-' * 60)
            logging.info('On {} Set:\n\tAUC of MLLP Model: {}'
                         '\n\tAUC of RRL  Model: {}'.format(set_name, auc, auc_b))
            logging.info('On {} Set:\n\tNDCG5 of MLLP Model: {}'
                         '\n\tNDCG5 of RRL  Model: {}'.format(set_name, ndcg5, ndcg5_b))
            logging.info('On {} Set:\n\tNDCG10 of MLLP Model: {}'
                         '\n\tNDCG10 of RRL  Model: {}'.format(set_name, ndcg10, ndcg10_b))
            logging.info('-' * 60)
        return auc,auc_b,ndcg5,ndcg5_b,ndcg10,ndcg10_b

    def data_write_csv(self,file_name, datas):
        file_csv = codecs.open(file_name,'w+','utf-8')
        writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for name,par in datas.items():
            writer.writerow(name)
            writer.writerow(par)
    
    def save_model(self):
        rrl_args = {'db_enc_list': self.db_enc_list,'bottom_structure_list':self.bottom_structure_list,
                    'top_structure':self.top_structure, 'use_not': self.use_not, 'estimated_grad': self.estimated_grad}
        torch.save({'model_state_dict': self.net.state_dict(), 'rrl_args': rrl_args}, self.save_path)

    def detect_dead_node(self, data_loader_list=None):
        with torch.no_grad():
            for net in self.net.bottom_net_list:
                for layer in net.layer_list[:]:
                    layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=self.device_id)
                    layer.forward_tot = 0

            for data_loader_zip in zip(*data_loader_list):
                x_list = []
                cnt=0
                for data in data_loader_zip:
                    x = data[0].cuda(self.device_id, non_blocking=True)
                    x_res = None
                    for i, layer in enumerate(self.net.bottom_net_list[cnt].layer_list[:]):
                        if i <= 1:
                            x = layer.binarized_forward(x)
                        else:
                            x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                            x_res = x
                            x = layer.binarized_forward(x_cat)
                        layer.node_activation_cnt += torch.sum(x, dim=0)
                        layer.forward_tot += x.shape[0]
                    x_list.append(x)
                    cnt+=1

    def rule_print(self, db_enc_list, train_loader_list, file=sys.stdout):
        if self.net.bottom_net_list[0].layer_list[1] is None and train_loader_list is None:
            raise Exception("Need train_loader for the dead nodes detection.")
        if self.net.bottom_net_list[0].layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(data_loader_list=train_loader_list)

        botton_rule_dict,tower_id = [{},{}],0
        for db_enc,net in zip(db_enc_list,self.net.bottom_net_list):
            bound_name = net.layer_list[0].get_bound_name(db_enc.X_fname, db_enc.mean, db_enc.std)
            net.layer_list[1].get_rules(net.layer_list[0], None)
            net.layer_list[1].get_rule_description((None, bound_name))

            if len(net.layer_list) >= 4:
                net.layer_list[2].get_rules(net.layer_list[1], None)
                net.layer_list[2].get_rule_description((None, net.layer_list[1].rule_name), wrap=True)

            if len(net.layer_list) >= 5:
                for i in range(3, len(net.layer_list) - 1):
                    net.layer_list[i].get_rules(net.layer_list[i - 1], net.layer_list[i - 2])
                    net.layer_list[i].get_rule_description(
                        (net.layer_list[i - 2].rule_name, net.layer_list[i - 1].rule_name), wrap=True)

            prev_layer = net.layer_list[-2]
            skip_connect_layer = net.layer_list[-3]
            if skip_connect_layer.layer_type == 'union':
                shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
                prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
                merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            else:
                merged_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}

            for label_id in range(2):
                Wl, bl = list(self.net.top_net.layer_list[-1].parameters())
                Wl = Wl[label_id].cpu().detach().numpy()
                marked = defaultdict(float)
                for i, w in enumerate(Wl):
                    if(i>=len(merged_dim2id)):
                        break
                    rid = merged_dim2id[i]
                    if rid == -1 or rid[1] == -1:
                        continue
                    marked[rid] += w

                kv_list = sorted(marked.items(), key=lambda x: abs(x[1]), reverse=True)
                for k, v in kv_list:
                    rid = k
                    botton_rule_dict[label_id][self.net.bottom_net_list[0].layer_list[-1].output_dim*tower_id+rid[1]]=net.layer_list[-1 + rid[0]].rule_name[rid[1]]
            tower_id+=1

        top_layer=self.net.top_net.layer_list[-1]
        for label_id in range(2):
            print('Class: {}\n'.format(label_id), file=file)
            con_Wb = (top_layer.con_layer.W > 0.5).type(torch.int).detach().cpu().numpy()
            for ri, row in enumerate(con_Wb):
                rule,flag,cnt=[],0,0
                for i, w in enumerate(row):
                    if w>0:
                        cnt+=1
                        if i in botton_rule_dict[label_id]:
                            rule.append(botton_rule_dict[label_id][i])
                            flag+=1
                if flag==cnt and flag>0:
                    print('rule',ri, file=file)
                    print('(',') & ('.join(rule),')', end='\n\n', file=file)

            dis_Wb = (top_layer.dis_layer.W > 0.5).type(torch.int).detach().cpu().numpy()
            for ri, row in enumerate(dis_Wb):
                rule,flag=[],0
                for i, w in enumerate(row):
                    if w>0 and i in botton_rule_dict[label_id]:
                        rule.append(botton_rule_dict[label_id][i])
                        flag+=1
                if flag>0:
                    print('rule',ri+len(dis_Wb), file=file)
                    print('(',') | ('.join(rule),')', end='\n\n', file=file)


