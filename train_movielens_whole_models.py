import numpy as np
import pandas as pd
import torch
import torchvision
import random
import os

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat

from model.IntTower import IntTower
from model.dssm import DSSM
from model.deepfm import DeepFM
from model.dcn import DCN
from model.dat import DAT
from model.cold import Cold
from model.autoint import AutoInt
from model.wdm import WideDeep
from model.dual_tower import DualTower

from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint


# 删掉rating=3的样本，rating>3的样本为正样本（设置rating=1），rating<3的样本为负样本（设置rating=0）。
# 按时间戳排序，拆分训练集和测试集
def data_process(data_path):
    data = pd.read_csv(data_path)
    data = data.drop(data[data['rating'] == 3].index)
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    data = data.sort_values(by='timestamp', ascending=True)
    train, test = train_test_split(data, test_size=0.2)
    return train, test, data


def get_user_feature(data):
    # 只选择正样本
    data_group = data[data['rating'] == 1]
    # 根据 movie_id，创造新的列'user_hist'，把单个user看过的所有正样本movie_id都聚合，用|分隔
    data_group = data_group[['user_id', 'movie_id']].groupby('user_id').agg(list).reset_index()
    data_group['user_hist'] = data_group['movie_id'].apply(lambda x: '|'.join([str(i) for i in x]))
    data = pd.merge(data_group.drop('movie_id', axis=1), data, on='user_id')
    # （不区分正负样本）根据 rating，创造新的列'user_mean_rating'，把单个user看过的所有movie的评分rating，求个均值
    data_group = data[['user_id', 'rating']].groupby('user_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'user_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='user_id')
    return data


def get_item_feature(data):
    # 根据 rating，创造新的列'item_mean_rating'，把单个movie所有看过的人的评分rating，求个均值
    data_group = data[['movie_id', 'rating']].groupby('movie_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'item_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='movie_id')
    return data


# 将变长的序列特征变为定长的特征（长度不足就补0），将字符串转化为数字
def get_var_feature(data, col):
    key2index = {}

    # 把用|分隔的seq_feature，从字符串转化为数字，数字是按照字符串出现的先后顺序进行编码的
    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",\
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    var_feature = list(map(split, data[col].values))
    var_feature_length = np.array(list(map(len, var_feature)))
    max_len = max(var_feature_length)
    var_feature = pad_sequences(var_feature, maxlen=max_len, padding='post', )
    # key2index为全体【字符串与数字】的映射关系，var_feature 为所需要的编码后的信息，max_len为最大字符串长度。
    return key2index, var_feature, max_len


def get_test_var_feature(data, col, key2index, max_len):
    print("user_hist_list: \n")

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    test_hist = list(map(split, data[col].values))
    test_hist = pad_sequences(test_hist, maxlen=max_len, padding='post')
    return test_hist


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    use_cuda, cuda_number = True, 'cuda:0'
    # ["int_tower", "dssm", "dual_tower", "dat", "deep_fm", "dcn", "cold", "auto_int", "wide_and_deep"]
    model_name = "int_tower"
    data_path = './data/movielens_test.txt'
    ckpt_fold = './checkpoints'

    embedding_dim = 32
    epoch = 10
    batch_size = 2048
    lr = 0.001
    dropout = 0.3
    setup_seed(seed=1023)

    ckpt_path = '{}/{}.ckpt'.format(ckpt_fold, model_name)
    # 检查文件夹是否存在
    if not os.path.exists(ckpt_fold):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(ckpt_fold)
        print(f"文件夹'{ckpt_fold}'已创建。")
    else:
        print(f"文件夹'{ckpt_fold}'已存在。")

    train, test, data = data_process(data_path)

    train = get_user_feature(train)
    train = get_item_feature(train)

    test = get_user_feature(test)
    test = get_item_feature(test)

    target = ['rating']

    user_sparse_features, user_dense_features = ['user_id', 'gender', 'age', 'occupation'], ['user_mean_rating']
    item_sparse_features, item_dense_features = ['movie_id'], ['item_mean_rating']
    sparse_features = user_sparse_features + item_sparse_features
    dense_features = user_dense_features + item_dense_features

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        train[feat] = lbe.transform(train[feat])
        test[feat] = lbe.transform(test[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(train[dense_features])
    mms.fit(test[dense_features])
    train[dense_features] = mms.transform(train[dense_features])
    test[dense_features] = mms.transform(test[dense_features])

    # 2.preprocess the sequence feature
    genres_key2index, train_genres_list, genres_maxlen = get_var_feature(train, 'genres')
    user_key2index, train_user_hist, user_maxlen = get_var_feature(train, 'user_hist')

    user_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                            for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               user_dense_features]
    item_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                            for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               item_dense_features]
    print("genres_maxlen: ", genres_maxlen, "; user_maxlen: ", user_maxlen)
    # 处理序列特征
    item_varlen_feature_columns = [
        VarLenSparseFeat(SparseFeat('genres', vocabulary_size=1000, embedding_dim=embedding_dim),
                         maxlen=genres_maxlen, combiner='mean', length_name=None)]

    user_varlen_feature_columns = [
        VarLenSparseFeat(SparseFeat('user_hist', vocabulary_size=4000, embedding_dim=embedding_dim),
                         maxlen=user_maxlen, combiner='mean', length_name=None)]

    # 3.generate input data for model
    user_feature_columns += user_varlen_feature_columns
    item_feature_columns += item_varlen_feature_columns

    # add user history as user_varlen_feature_columns
    train_model_input = {name: train[name] for name in sparse_features + dense_features}
    train_model_input["genres"] = train_genres_list
    train_model_input["user_hist"] = train_user_hist

    # %%
    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = cuda_number

    # print(train_model_input)

    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1,
                       patience=5, mode='max', baseline=None)
    mdckpt = ModelCheckpoint(filepath=ckpt_path, monitor='val_auc',
                             mode='max', verbose=1, save_best_only=True, save_weights_only=True)

    # 仅用于非双塔的模型
    linear_feature_columns = user_feature_columns + item_feature_columns
    dnn_feature_columns = linear_feature_columns
    if model_name == "int_tower":
        model = IntTower(user_feature_columns, item_feature_columns, field_dim=64, task='binary', dnn_dropout=dropout,
                         device=device, user_head=32, item_head=32, user_filed_size=5, item_filed_size=2)
    elif model_name == "dssm":
        model = DSSM(user_feature_columns, item_feature_columns, task='binary', device=device)
    elif model_name == "dual_tower":
        model = DualTower(user_feature_columns, item_feature_columns, task='binary',
                          device=device)
    elif model_name == "dat":
        model = DAT(user_feature_columns, item_feature_columns, task='binary', dnn_dropout=dropout,
                    device=device)
    elif model_name == "deep_fm":
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout,
                       device=device)
    elif model_name == "dcn":
        model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout,
                    device=device)
    elif model_name == "cold":
        model = Cold(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout,
                     device=device)
    elif model_name == "auto_int":
        model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout,
                        device=device)
    elif model_name == "wide_and_deep":
        model = WideDeep(linear_feature_columns, dnn_feature_columns, task='binary',
                         device=device)
    else:
        raise ValueError("There is no such value for model_name")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['auc', 'accuracy', 'logloss']
                  , lr=lr)
    # 因为加了early stopping，所以保留的模型是在验证集上val_auc表现最好的模型
    model.fit(train_model_input, train[target].values, batch_size=batch_size, epochs=epoch, verbose=2,
              validation_split=0.2,
              callbacks=[es, mdckpt])

    # 开始模型评估
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    test_model_input = {name: test[name] for name in sparse_features + dense_features}
    test_model_input["genres"] = get_test_var_feature(test, 'genres', genres_key2index, genres_maxlen)
    test_model_input["user_hist"] = get_test_var_feature(test, 'user_hist', user_key2index, user_maxlen)

    # %%
    # 6.Evaluate
    # 看下最佳模型在完整的训练集上的表现
    eval_tr = model.evaluate(train_model_input, train[target].values)
    print(eval_tr)

    # %%
    pred_ts = model.predict(test_model_input, batch_size=2048)
    print("test LogLoss", round(log_loss(test[target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ts), 4))
