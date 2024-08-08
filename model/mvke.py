"""

"""

from model.base_tower import BaseTower
from preprocessing.inputs import combined_dnn_input, compute_input_dim
from layers.core import DNN
import torch
import torch.nn as nn
from preprocessing.utils import Cosine_Similarity
from preprocessing.utils import col_score
from preprocessing.utils import col_score_2
from preprocessing.utils import single_score
from layers.interaction import SENETLayer
from layers.interaction import LightSE
from layers.mmoe import MMOE_ExpertAttention
from layers.attention import target_dot_attention

class MVKE(BaseTower):
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=[300, 300, 128], dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-5,
                 dnn_dropout = 0, init_std=0.0001, seed=124, batch_size = 128, task='binary', device='cpu', gpus=None):
        super(MVKE, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                    l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                    device=device, gpus=gpus)

        self.target_dot_attention = target_dot_attention(attention_dropout=dnn_dropout, device=device)
        self.towers_hidden = 64
        self.tasks = 4
        if len(user_dnn_feature_columns) > 0:
            self.user_dnn = DNN(compute_input_dim(user_dnn_feature_columns), dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.mmoe = MMOE_ExpertAttention(input_size=dnn_hidden_units[-1], num_experts=6, experts_out=16, experts_hidden=32,
                             towers_hidden=self.towers_hidden, tasks=self.tasks, dnn_dropout=dnn_dropout,device=device)

            self.virtualKernel = nn.Parameter(torch.randn(batch_size, dnn_hidden_units[-1]))

            self.user_dnn_2 = DNN(self.towers_hidden * self.tasks, [dnn_hidden_units[-1]],
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)

            self.user_dnn_embedding = None

        if len(item_dnn_feature_columns) > 0:
            self.item_dnn = DNN(compute_input_dim(item_dnn_feature_columns), dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_dnn_embedding = None

        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus

        self.user_filed_size = 4
        self.item_filed_size = 2

        self.User_SE = SENETLayer(self.user_filed_size, 3, seed, device)
        self.Item_SE = SENETLayer(self.item_filed_size, 3, seed, device)
    def forward(self, inputs):
        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)

            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
            user_embedding = self.user_dnn(user_dnn_input)

            user_mmoe_embedding = self.mmoe(user_embedding, self.virtualKernel)
            user_mmoe_embedding = torch.flatten(torch.cat(user_mmoe_embedding, dim=-1), start_dim=1)
            self.user_dnn_embedding = self.user_dnn_2(user_mmoe_embedding)


        if len(self.item_dnn_feature_columns) > 0:
            item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)

            item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
            self.item_dnn_embedding = self.item_dnn(item_dnn_input)
            target_embed = self.target_dot_attention(q=self.item_dnn_embedding, k=self.virtualKernel,
                                                                v=self.user_dnn_embedding)
            self.user_dnn_embedding = self.user_dnn_embedding + target_embed


        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            score = Cosine_Similarity(self.user_dnn_embedding, self.item_dnn_embedding, gamma=self.gamma)
            output = self.out(score)
            return output, self.user_dnn_embedding, self.item_dnn_embedding

        elif len(self.user_dnn_feature_columns) > 0:
            return self.user_dnn_embedding

        elif len(self.item_dnn_feature_columns) > 0:
            return self.item_dnn_embedding

        else:
            raise Exception("input Error! user and item feature columns are empty.")


