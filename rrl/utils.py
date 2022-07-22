import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
import torch.nn as nn

def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    return f_list[:-1], int(f_list[-1][-1])

def read_itemvec(data_path):
    itemvec = pd.read_csv(data_path, header=None)
    itemvec = np.array(itemvec)
    itemvec_tensor=torch.tensor(itemvec)
    return itemvec_tensor

def read_csv(data_path, info_path, div_place, shuffle=False):
    D = pd.read_csv(data_path,sep=',', header=None)
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    #for embedding
    uid = D[0]
    iid = D[1]
    #drop uid
    D.drop([0,1],axis=1,inplace=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    D.columns = f_df.iloc[:, 0]
    y_df=D.iloc[:, [label_pos]]
    X_df, f = [],[]
    for i in range(1,len(div_place)):
        X_df.append(D.drop(D.columns[label_pos], axis=1).iloc[:,div_place[i-1]:div_place[i]])
        f.append(f_df.drop(f_df.index[label_pos]).iloc[div_place[i-1]:div_place[i]])
    X_df.append(D.drop(D.columns[label_pos], axis=1).iloc[:, div_place[- 1]:])
    f.append(f_df.drop(f_df.index[label_pos]).iloc[div_place[- 1]:])
    return X_df, y_df, f, label_pos,uid,iid


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def split_data(self, X_df):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(np.float)
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df)
        self.y_fname = list(self.label_enc.get_feature_names(y_df.columns)) if self.y_one_hot else y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            self.X_fname = list(self.feature_enc.get_feature_names(feature_names))
            self.discrete_flen = len(self.X_fname)
            if not self.discrete:
                self.X_fname.extend(continuous_data.columns)
        else:
            self.X_fname = continuous_data.columns
            self.discrete_flen = 0
        self.continuous_flen = continuous_data.shape[1]

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.
        y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std
        if not discrete_data.empty:
            # One-hot encoding
            discrete_data = self.feature_enc.transform(discrete_data)
            if not self.discrete:
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                X_df = pd.DataFrame(discrete_data.toarray())
        else:
            X_df = continuous_data
        return X_df.values, y


class UnionFind:
    """Union-Find algorithm used for merging the identical nodes in MLLP."""

    def __init__(self, keys):
        self.stu = {}
        for k in keys:
            self.stu[k] = k

    def find(self, x):
        try:
            self.stu[x]
        except KeyError:
            return x
        if x != self.stu[x]:
            self.stu[x] = self.find(self.stu[x])
        return self.stu[x]

    def union(self, x, y):
        xf = self.find(x)
        yf = self.find(y)
        if xf != yf:
            self.stu[yf] = xf
            return True
        return False

def cal_ndcg(predicts, labels, user_ids, k):
        d = {'user': np.squeeze(user_ids), 'predict': np.squeeze(predicts), 'label': np.squeeze(labels)}
        df = pd.DataFrame(d)
        user_unique = df.user.unique()

        ndcg = []
        for user_id in user_unique:
            user_srow = df.loc[df['user'] == user_id]
            upred = user_srow['predict'].tolist()
            if len(upred) < 2:
                # print('less than 2', user_id)
                continue
            # supred = [upred] if len(upred)>1 else [upred + [-1]]  # prevent error occured if only one sample for a user
            ulabel = user_srow['label'].tolist()
            # sulabel = [ulabel] if len(ulabel)>1 else [ulabel +[1]]

            ndcg.append(ndcg_score([ulabel], [upred], k=k))

        return np.mean(np.array(ndcg))

def AUC(label,pred):
        value=[]
        for i in range(len(label)):
            value.append([label[i],pred[i]])
        value.sort(key=lambda x: x[1])
        rank,n0,n1=0,0,0
        for i in range(len(value)):
            if value[i][0]==0:
                n0+=1
            else:
                n1+=1
                rank+=(i+1)
        if n0==0 or n1==0:
            return 0
        else:
            return (rank-n1*(n1+1)/2)/(n0*n1)

def cal_auc(predicts, labels, user_ids):
        d = {'user': np.squeeze(user_ids), 'predict': np.squeeze(predicts), 'label': np.squeeze(labels)}
        df = pd.DataFrame(d)
        user_unique = df.user.unique()

        auc = []
        for user_id in user_unique:
            user_srow = df.loc[df['user'] == user_id]
            upred = user_srow['predict'].tolist()
            if len(upred) < 2:
                # print('less than 2', user_id)
                continue
            # supred = [upred] if len(upred)>1 else [upred + [-1]]  # prevent error occured if only one sample for a user
            ulabel = user_srow['label'].tolist()
            auc.append(AUC(ulabel,upred))
        return np.mean(np.array(auc))
