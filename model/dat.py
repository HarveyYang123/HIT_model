"""



"""

from model.base_tower import BaseTower
from model.dual_tower import DualTower
from preprocessing.inputs import combined_dnn_input, compute_input_dim
from layers.core import DNN
import torch
from preprocessing.utils import Cosine_Similarity
from preprocessing.utils import col_score
from preprocessing.utils import col_score_2
from preprocessing.utils import single_score
from layers.interaction import SENETLayer

class DAT(DualTower):
    """https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf"""
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 32), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-5,
                 dnn_dropout = 0, init_std=0.0001, seed=124, task='binary', device='cpu', gpus=None,
                 user_aug_vector_dim=32, item_aug_vector_dim=32):
        super(DAT, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                    l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                    device=device, gpus=gpus)

        self.user_aug_vector_dim = user_aug_vector_dim
        self.item_aug_vector_dim = item_aug_vector_dim
        if len(user_dnn_feature_columns) > 0:
            self.user_dnn = DNN(compute_input_dim(user_dnn_feature_columns)+self.user_aug_vector_dim, dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.user_dnn_embedding = None

        if len(item_dnn_feature_columns) > 0:
            self.item_dnn = DNN(compute_input_dim(item_dnn_feature_columns)+self.item_aug_vector_dim, dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_dnn_embedding = None


        # self.item_dnn_feature_columns = item_dnn_feature_columns
        #
        # self.user_dnn_feature_columns = user_dnn_feature_columns
        # self.dnn_hidden_units = dnn_hidden_units
        # self.dnn_activation = dnn_activation
        # self.l2_reg_dnn = l2_reg_dnn
        # self.dnn_dropout = dnn_dropout
        # self.dnn_use_bn = dnn_use_bn
        # self.init_std = init_std
        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus
        self.user_aug_vector = None
        self.item_aug_vector = None

        self.user_filed_size = 2
        self.item_filed_size = 2

        self.User_SE = SENETLayer(self.user_filed_size, 3, seed, device)
        self.Item_SE = SENETLayer(self.item_filed_size, 3, seed, device)


    def forward(self, inputs):
        # user tower
        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)
            # print(user_sparse_embedding_list,len(user_sparse_embedding_list))
            # print(user_sparse_embedding_list[-1],user_sparse_embedding_list[-1].shape)

            if torch.cuda.is_available():
                self.user_aug_vector = torch.rand(user_sparse_embedding_list[-1].shape).cuda()
            else:
                self.user_aug_vector = torch.rand(user_sparse_embedding_list[-1].shape)

            # ensure the input dimension of DNN is initialized correctly
            assert self.user_aug_vector.shape[-1] == self.user_aug_vector_dim

            user_sparse_embedding_list.append(self.user_aug_vector)

            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

            self.user_dnn_embedding = self.user_dnn(user_dnn_input)

            # user_sparse_embedding = torch.cat(user_sparse_embedding_list, dim= 1)
            # # print(user_sparse_embedding.shape)
            # User_sim_embedding = self.User_SE(user_sparse_embedding)
            # sparse_dnn_input = torch.flatten(User_sim_embedding, start_dim=1)
            # dense_dnn_input = torch.flatten(torch.cat(user_dense_value_list, dim=-1), start_dim=1)
            #
            # user_dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input],axis=-1)

            # user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)


            # self.user_dnn_embedding = self.user_dnn(user_dnn_input)

            # print(self.user_dnn_embedding.shape)

            # self.user_dnn_embedding = self.user_col_dense(self.user_dnn_embedding)
            # self.user_dnn_embedding = self.dense(self.user_dnn_embedding)

        # item tower
        if len(self.item_dnn_feature_columns) > 0:
            item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)

            if torch.cuda.is_available():
                self.item_aug_vector = torch.rand(item_sparse_embedding_list[-1].shape).cuda()
            else:
                self.item_aug_vector = torch.rand(item_sparse_embedding_list[-1].shape)

            # ensure the input dimension of DNN is initialized correctly
            assert self.item_aug_vector.shape[-1] == self.item_aug_vector_dim

            item_sparse_embedding_list.append(self.item_aug_vector)
            item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
            # print(item_dnn_input.shape)

            self.item_dnn_embedding = self.item_dnn(item_dnn_input)

            # item_sparse_embedding = torch.cat(item_sparse_embedding_list, dim=1)
            # Item_sim_embedding = self.Item_SE(item_sparse_embedding)
            # sparse_dnn_input = torch.flatten(Item_sim_embedding, start_dim=1)
            # dense_dnn_input = torch.flatten(torch.cat(item_dense_value_list, dim=-1), start_dim=1)
            #
            # item_dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], axis=-1)
            #
            # # item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
            #
            # self.item_dnn_embedding = self.item_dnn(item_dnn_input)

            # self.item_dnn_embedding = self.item_col_dense(self.item_dnn_embedding)
            # self.item_dnn_embedding = self.dense(self.item_dnn_embedding)

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            score = Cosine_Similarity(self.user_dnn_embedding, self.item_dnn_embedding, gamma=self.gamma)
            # print(score.shape)
            # score = col_score(self.user_dnn_embedding, self.item_dnn_embedding,len(self.user_dnn_feature_columns))
            # score = col_score_2(self.user_dnn_embedding, self.item_dnn_embedding, len(self.user_dnn_feature_columns),\
            #                   len(self.item_dnn_feature_columns),128)
            # score = single_score(self.item_dnn_embedding)
            # print(score.shape)
            output = self.out(score)
            return output, self.user_dnn_embedding, \
                   self.item_dnn_embedding, self.user_aug_vector,self.item_aug_vector

        elif len(self.user_dnn_feature_columns) > 0:
            return self.user_dnn_embedding

        elif len(self.item_dnn_feature_columns) > 0:
            return self.item_dnn_embedding

        else:
            raise Exception("input Error! user and item feature columns are empty.")