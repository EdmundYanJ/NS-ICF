import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import random_split,Subset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import copy
from sklearn.model_selection import KFold
import csv
import codecs

from rrl.utils import read_csv, read_itemvec, DBEncoder
from rrl.models import RRL
from args import rrl_args

DATA_DIR = './dataset/'+rrl_args.dataset_folder
div_place=list(map(int, rrl_args.div_place.split('@')))

data=pd.read_csv(DATA_DIR,header=None)
item_num=data[1].max()
user_num=data[0].max()
embedding_dim=64

def get_data_loader_train(train_dataset, valid_dataset, world_size, rank, batch_size, div_place, pin_memory=False,
                    save_best=True):
    train_data_path = os.path.join(DATA_DIR, train_dataset + '.data')
    train_info_path = os.path.join(DATA_DIR, train_dataset + '.info')
    X_df, y_df, f_df, label_pos, uid, iid = read_csv(train_data_path, train_info_path, div_place, shuffle=True)
    db_enc_list, train_loader_list = [], []

    valid_data_path = os.path.join(DATA_DIR, valid_dataset + '.data')
    valid_info_path = os.path.join(DATA_DIR, train_dataset + '.info')
    X_df_valid, y_df_valid, f_df_valid, label_pos, valid_uid, valid_iid = read_csv(valid_data_path, valid_info_path, div_place, shuffle=True)
    valid_loader_list = []

    db_enc = DBEncoder(f_df, discrete=False)
    # db_enc = DBEncoder(f_df, discrete=True)
    db_enc.fit(X_df, y_df)
    #train_set
    X, Y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    train_set = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.float32)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader_list.append(
        DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=train_sampler))

    #valid set
    X_valid, Y_valid = db_enc.transform(X_df_valid, y_df_valid, normalized=True, keep_stat=True)
    valid_set = TensorDataset(torch.tensor(X_valid.astype(np.float32)), torch.tensor(Y_valid.astype(np.float32)))
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set, num_replicas=world_size, rank=rank)
    valid_loader_list.append(
        DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=valid_sampler))

    #make batch data for uid iid
    uid_sampler = torch.utils.data.distributed.DistributedSampler(uid, num_replicas=world_size, rank=rank)
    uid=DataLoader(uid,batch_size=batch_size,shuffle=False, pin_memory=pin_memory, sampler=uid_sampler)
    iid_sampler = torch.utils.data.distributed.DistributedSampler(iid, num_replicas=world_size, rank=rank)
    iid = DataLoader(iid, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=iid_sampler)

    valid_uid_sampler = torch.utils.data.distributed.DistributedSampler(valid_uid, num_replicas=world_size, rank=rank)
    valid_uid = DataLoader(valid_uid, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=valid_uid_sampler)
    valid_iid_sampler = torch.utils.data.distributed.DistributedSampler(valid_iid, num_replicas=world_size, rank=rank)
    valid_iid = DataLoader(valid_iid, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=valid_iid_sampler)

    return db_enc_list, train_loader_list, valid_loader_list,uid,iid,valid_uid,valid_iid

def get_data_loader_test(train_dataset,test_dataset, world_size, rank, batch_size, div_place, pin_memory=False,
                    save_best=True):
    test_data_path = os.path.join(DATA_DIR, test_dataset + '.data')
    test_info_path = os.path.join(DATA_DIR, train_dataset + '.info')
    X_df_test, y_df_test, f_df_test, label_pos,test_uid,test_iid = read_csv(test_data_path, test_info_path, div_place, shuffle=True)
    db_enc_list,test_loader_list = [],[]

    db_enc = DBEncoder(f_df_test, discrete=False)
    # db_enc = DBEncoder(f_df, discrete=True)
    db_enc.fit(X_df_test, y_df_test)
    db_enc_list.append(copy.deepcopy(db_enc))
    #test_set
    X_test, Y_test = db_enc.transform(X_df_test, y_df_test, normalized=True, keep_stat=True)
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(Y_test.astype(np.float32)))
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    test_loader_list.append(
        DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=test_sampler))
    #make batch for uid iid
    test_uid_sampler = torch.utils.data.distributed.DistributedSampler(test_uid, num_replicas=world_size, rank=rank,
                                                                  shuffle=False)
    test_uid = DataLoader(test_uid, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=test_uid_sampler)
    test_iid_sampler = torch.utils.data.distributed.DistributedSampler(test_iid, num_replicas=world_size, rank=rank,
                                                                  shuffle=False)
    test_iid = DataLoader(test_iid, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=test_iid_sampler)

    return db_enc_list, test_loader_list, test_uid, test_iid

def train_model(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    torch.manual_seed(42)
    device_id = args.device_ids[gpu]
    torch.cuda.set_device(device_id)

    if gpu == 0:
        writer = SummaryWriter(args.folder_path)
        is_rank0 = True
    else:
        writer = None
        is_rank0 = False

    train_dataset = args.train_dataset
    valid_dataset = args.valid_dataset
    db_enc_list, train_loader_list, valid_loader_list, uid, iid, valid_uid, valid_iid = get_data_loader_train(
                                                        train_dataset,valid_dataset, args.world_size, rank, args.batch_size,
                                                        div_place=div_place,pin_memory=True, save_best=args.save_best)

    rrl = RRL(item_num=item_num,
              user_num=user_num,
              embedding_dim=embedding_dim,
              db_enc_list=db_enc_list,
              bottom_structure_list=[args.structure,args.structure,args.structure],
              top_structure=args.top_structure,
              topk=args.topk_target,
              device_id=device_id,
              use_not=args.use_not,
              is_rank0=is_rank0,
              log_file=args.log,
              writer=writer,
              save_best=args.save_best,
              estimated_grad=args.estimated_grad,
              save_path=args.model)

    rrl.train_model(
        data_loader_list=train_loader_list,
        valid_loader_list=valid_loader_list,
        user=uid,
        item=iid,
        valid_user=valid_uid,
        valid_item=valid_iid,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter)


def load_model(path, device_id,  topk_target, log_file=None, distributed=True):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['rrl_args']
    rrl = RRL(
        item_num=item_num,
        user_num=user_num,
        embedding_dim=embedding_dim,
        db_enc_list=saved_args['db_enc_list'],
        bottom_structure_list=saved_args['bottom_structure_list'],
        top_structure=saved_args['top_structure'],
        topk=topk_target,
        device_id=device_id,
        is_rank0=True,
        use_not=saved_args['use_not'],
        log_file=log_file,
        distributed=distributed,
        estimated_grad=saved_args['estimated_grad'])
    stat_dict = checkpoint['model_state_dict']
    return rrl

def test_model(args):
    rrl = load_model(args.model, args.device_ids[0],  args.topk_target, log_file=args.test_res, distributed=False)

    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    db_enc_list, test_loader_list, test_uid,test_iid = get_data_loader_test(train_dataset,test_dataset, 1, 0, args.batch_size,
                                        div_place=div_place, pin_memory=True, save_best=args.save_best)
    rrl.test(test_loader_list=test_loader_list,user=test_uid,item=test_iid, set_name='Test')

def train_main(args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    # for arg in vars(rrl_args):
    #     print(arg, getattr(rrl_args, arg))
    train_main(rrl_args)
    test_model(rrl_args)
    # train_model()
    # test_model(test_set)

