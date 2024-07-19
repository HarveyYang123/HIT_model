import numpy as np
import pandas as pd
import torch
import random

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
class movieDataProcess():
    def __init__(self, log, data_path, embedding_dim):
        self.log = log
        self.data_path = data_path
        train, test, data = self.data_process()

        train = self.get_user_feature(train)
        train = self.get_item_feature(train)
        self.train = train

        test = self.get_user_feature(test)
        test = self.get_item_feature(test)
        self.test = test

        self.target = ['rating']
        user_sparse_features, user_dense_features = ['user_id', 'gender', 'age', 'occupation'], ['user_mean_rating']
        item_sparse_features, item_dense_features = ['movie_id'], ['item_mean_rating']
        self.sparse_features = user_sparse_features + item_sparse_features
        self.dense_features = user_dense_features + item_dense_features

        # 1.Label Encoding for sparse features,and process sequence features
        for feat in self.sparse_features:
            lbe = LabelEncoder()
            lbe.fit(data[feat])
            train[feat] = lbe.transform(train[feat])
            test[feat] = lbe.transform(test[feat])

        mms = MinMaxScaler(feature_range=(0, 1))
        mms.fit(train[self.dense_features])
        mms.fit(test[self.dense_features])
        self.train[self.dense_features] = mms.transform(train[self.dense_features])
        self.test[self.dense_features] = mms.transform(test[self.dense_features])

        # 2.preprocess the sequence feature
        genres_key2index, train_genres_list, genres_maxlen = self.get_var_feature(train, 'genres')
        user_key2index, train_user_hist, user_maxlen = self.get_var_feature(train, 'user_hist')

        user_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                                   user_dense_features]
        item_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                                   item_dense_features]

        # 处理序列特征
        item_varlen_feature_columns = [
            VarLenSparseFeat(SparseFeat('genres', vocabulary_size=1000, embedding_dim=embedding_dim),
                             maxlen=genres_maxlen, combiner='mean', length_name=None)]

        user_varlen_feature_columns = [
            VarLenSparseFeat(SparseFeat('user_hist', vocabulary_size=4000, embedding_dim=embedding_dim),
                             maxlen=user_maxlen, combiner='mean', length_name=None)]

        # 3.generate input data for model
        self.user_feature_columns = user_feature_columns + user_varlen_feature_columns
        self.item_feature_columns = item_feature_columns + item_varlen_feature_columns

        # add user history as user_varlen_feature_columns
        self.train_model_input = {name: train[name] for name in self.sparse_features + self.dense_features}
        self.train_model_input["genres"] = train_genres_list
        self.train_model_input["user_hist"] = train_user_hist

        self.test_model_input = {name: test[name] for name in self.sparse_features + self.dense_features}
        self.test_model_input["genres"] = self.get_test_var_feature(test, 'genres', genres_key2index, genres_maxlen)
        self.test_model_input["user_hist"] = self.get_test_var_feature(test, 'user_hist', user_key2index, user_maxlen)


    def data_process(self):
        # 删掉rating=3的样本，rating>3的样本为正样本（设置rating=1），rating<3的样本为负样本（设置rating=0）
        # 按时间戳排序，拆分训练集和测试集
        data = pd.read_csv(self.data_path)
        data = data.drop(data[data['rating'] == 3].index)
        data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
        data = data.sort_values(by='timestamp', ascending=True)
        train, test = train_test_split(data, test_size=0.2)
        return train, test, data

    def get_user_feature(self, data):
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

    def get_item_feature(self, data):
        # 根据 rating，创造新的列'item_mean_rating'，把单个movie所有看过的人的评分rating，求个均值
        data_group = data[['movie_id', 'rating']].groupby('movie_id').agg('mean').reset_index()
        data_group.rename(columns={'rating': 'item_mean_rating'}, inplace=True)
        data = pd.merge(data_group, data, on='movie_id')
        return data


    # 将变长的序列特征变为定长的特征（长度不足就补0），将字符串转化为数字
    def get_var_feature(self, data, col):
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


    def get_test_var_feature(self, data, col, key2index, max_len):
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



# class amazon_DataProcess():