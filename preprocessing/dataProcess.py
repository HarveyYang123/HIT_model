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
        user_feature_for_recon = ['gender', 'age']
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

        # self.user_feature_columns_for_recon = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
        #                         for i, feat in enumerate(user_sparse_features)] + user_varlen_feature_columns
        # self.item_feature_columns_for_recon = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
        #                         for i, feat in enumerate(item_sparse_features)] + item_varlen_feature_columns
        self.user_feature_columns_for_recon = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                for i, feat in enumerate(user_feature_for_recon)]
        self.item_feature_columns_for_recon = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                for i, feat in enumerate(item_sparse_features)]

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


class amazonDataProcess():
    def __init__(self, log, data_path, embedding_dim):
        data = self.data_process(data_path)
        data = self.get_user_feature(data)
        data = self.get_item_feature(data)

        sparse_features = ['reviewerID', 'asin', 'categories']
        dense_features = ['user_mean_rating', 'item_mean_rating', 'price']
        self.target = ['overall']

        user_sparse_features, user_dense_features = ['reviewerID'], ['user_mean_rating']
        item_sparse_features, item_dense_features = ['asin', 'categories'], ['item_mean_rating', 'price']

        # 1.Label Encoding for sparse features,and process sequence features
        for feat in sparse_features:
            lbe = LabelEncoder()
            lbe.fit(data[feat])
            data[feat] = lbe.transform(data[feat])

        mms = MinMaxScaler(feature_range=(0, 1))
        mms.fit(data[dense_features])
        data[dense_features] = mms.transform(data[dense_features])

        self.train, self.test = train_test_split(data, test_size=0.2)

        self.user_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                                   user_dense_features]
        self.item_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                                   item_dense_features]

        self.user_feature_columns_for_recon = self.user_feature_columns
        self.item_feature_columns_for_recon = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                               for i, feat in enumerate(item_sparse_features)]

        self.train_model_input = {name: self.train[name] for name in sparse_features + dense_features}
        self.test_model_input = {name: self.test[name] for name in sparse_features + dense_features}

    def data_process(self, data_path):
        data = pd.read_csv(data_path)
        # data = data.drop(data[data['overall'] == 3].index)
        data['overall'] = data['overall'].apply(lambda x: 1 if x >= 4 else 0)
        data['price'] = data['price'].fillna(data['price'].mean())
        data = data.sort_values(by='unixReviewTime', ascending=True)
        # train = data.iloc[:int(len(data)*0.8)].copy()
        # test = data.iloc[int(len(data)*0.8):].copy()
        # train, test = train_test_split(data, test_size=0.2)
        # return train, test, data
        return data

    def get_user_feature(self, data):
        data_group = data[data['overall'] == 1]
        data_group = data_group[['reviewerID', 'asin']].groupby('reviewerID').agg(list).reset_index()
        data_group['user_hist'] = data_group['asin'].apply(lambda x: '|'.join([str(i) for i in x]))
        data = pd.merge(data_group.drop('asin', axis=1), data, on='reviewerID')
        data_group = data[['reviewerID', 'overall']].groupby('reviewerID').agg('mean').reset_index()
        data_group.rename(columns={'overall': 'user_mean_rating'}, inplace=True)
        data = pd.merge(data_group, data, on='reviewerID')
        return data

    def get_item_feature(self, data):
        data_group = data[['asin', 'overall']].groupby('asin').agg('mean').reset_index()
        data_group.rename(columns={'overall': 'item_mean_rating'}, inplace=True)
        data = pd.merge(data_group, data, on='asin')
        return data

    def get_test_var_feature(self, data, col, key2index, max_len):
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

class taobaoDataProcess():
    def __init__(self, log, profile_path, ad_path, user_path, embedding_dim, sample_rate=1.):
        self.log = log
        self.sample_rate = sample_rate
        data = self.data_process(profile_path, ad_path, user_path)
        data = self.get_user_feature(data)

        sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id',
                           'final_gender_code', 'shopping_level', 'occupation', 'cate_id', 'campaign_id',
                           'customer', 'age_level', 'brand', 'pvalue_level', 'new_user_class_level']

        dense_features = ['price']

        user_sparse_features, user_dense_features = ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code',
                                                     'age_level', 'pvalue_level', 'shopping_level', 'occupation',
                                                     'new_user_class_level', ], []
        item_sparse_features, item_dense_features = ['adgroup_id', 'cate_id', 'campaign_id', 'customer',
                                                     'brand', 'pid'], ['price']
        self.target = ['clk_y']

        # 1.Label Encoding for sparse features,and process sequence features
        for feat in sparse_features:
            lbe = LabelEncoder()
            lbe.fit(data[feat])
            data[feat] = lbe.transform(data[feat])

        mms = MinMaxScaler(feature_range=(0, 1))
        mms.fit(data[dense_features])
        data[dense_features] = mms.transform(data[dense_features])

        self.train, self.test = train_test_split(data, test_size=0.2)

        # 2.preprocess the sequence feature
        self.user_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                                   user_dense_features]
        self.item_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                                   item_dense_features]

        user_sparse_features_recon = ['cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
                                      'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', ]
        item_sparse_features_recon = ['cate_id', 'customer', 'brand', 'pid']
        self.user_feature_columns_for_recon = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                               for i, feat in enumerate(user_sparse_features_recon)]

        self.item_feature_columns_for_recon = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                               for i, feat in enumerate(item_sparse_features_recon)]

        self.train_model_input = {name: self.train[name] for name in sparse_features + dense_features}

        self.test_model_input = {name: self.test[name] for name in sparse_features + dense_features}


    def optimiz_memory_profile(self, raw_data):
        optimized_gl = raw_data.copy()

        gl_int = raw_data.select_dtypes(include=['int'])
        converted_int = gl_int.apply(pd.to_numeric, downcast='unsigned')
        optimized_gl[converted_int.columns] = converted_int

        gl_obj = raw_data.select_dtypes(include=['object']).copy()
        converted_obj = pd.DataFrame()
        for col in gl_obj.columns:
            num_unique_values = len(gl_obj[col].unique())
            num_total_values = len(gl_obj[col])
            if num_unique_values / num_total_values < 0.5:
                converted_obj.loc[:, col] = gl_obj[col].astype('category')
            else:
                converted_obj.loc[:, col] = gl_obj[col]

        optimized_gl[converted_obj.columns] = converted_obj
        return optimized_gl

    def optimiz_memory(self, raw_data):
        optimized_g2 = raw_data.copy()

        g2_int = raw_data.select_dtypes(include=['int'])
        converted_int = g2_int.apply(pd.to_numeric, downcast='unsigned')
        optimized_g2[converted_int.columns] = converted_int

        g2_float = raw_data.select_dtypes(include=['float'])
        converted_float = g2_float.apply(pd.to_numeric, downcast='float')
        optimized_g2[converted_float.columns] = converted_float
        return optimized_g2

    def data_process(self, profile_path, ad_path, user_path):
        profile_data = pd.read_csv(profile_path)
        if self.sample_rate < 0.9:
            profile_data = profile_data.sample(frac=self.sample_rate, random_state=1)

        ad_data = pd.read_csv(ad_path)
        user_data = pd.read_csv(user_path)
        profile_data = self.optimiz_memory_profile(profile_data)
        ad_data = self.optimiz_memory(ad_data)
        user_data = self.optimiz_memory(user_data)
        profile_data.rename(columns={'user': 'userid'}, inplace=True)
        user_data.rename(columns={'new_user_class_level ': 'new_user_class_level'}, inplace=True)
        df1 = profile_data.merge(user_data, on="userid")
        data = df1.merge(ad_data, on="adgroup_id")
        data['brand'] = data['brand'].fillna('-1', ).astype('int32')
        # data['age_level'] = data['age_level'].fillna('-1', )
        # data['cms_segid'] = data['cms_segid'].fillna('-1', )
        # data['cms_group_id'] = data['cms_group_id'].fillna('-1', )
        # data['final_gender_code'] = data['final_gender_code'].fillna('-1', )
        data['pvalue_level'] = data['pvalue_level'].fillna('-1', ).astype('int32')
        # data['shopping_level'] = data['shopping_level'].fillna('-1', )
        # data['occupation'] = data['occupation'].fillna('-1', )
        data['new_user_class_level'] = data['new_user_class_level'].fillna('-1', ).astype('int32')
        data = data.sort_values(by='time_stamp', ascending=True)
        return data

    def get_user_feature(self, data):
        data_group = data[data['clk'] == 1]
        data_group = data_group[['userid', 'adgroup_id']].groupby('userid').agg(list).reset_index()
        data_group['user_hist'] = data_group['adgroup_id'].apply(lambda x: '|'.join([str(i) for i in x]))
        data = pd.merge(data_group.drop('adgroup_id', axis=1), data, on='userid')
        data_group = data[['userid', 'clk']].groupby('userid').agg('mean').reset_index()
        #     data_group.rename(columns={'overall': 'user_mean_rating'}, inplace=True)
        data = pd.merge(data_group, data, on='userid')
        return data

    def get_test_var_feature(self, data, col, key2index, max_len):
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