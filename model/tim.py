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
from layers.interaction import SENETLayer, ImplicitInteraction
import torch.nn as nn
from layers.activation import activation_layer


class TimTower(DualTower):
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 32), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-5,
                 dnn_dropout=0, init_std=0.0001, seed=124, task='binary', device='cpu', gpus=None,
                 hidden_units_for_recon=(160, 64, 32), activation_for_recon='relu'):
        super(TimTower, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                       l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                       device=device, gpus=gpus)

        self.item_dnn_feature_columns = item_dnn_feature_columns

        self.user_dnn_feature_columns = user_dnn_feature_columns
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activation = dnn_activation
        self.l2_reg_dnn = l2_reg_dnn
        self.dnn_dropout = dnn_dropout
        self.dnn_use_bn = dnn_use_bn
        self.init_std = init_std
        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus
        # self.user_aug_vector = None
        # self.item_aug_vector = None
        self.user_dnn_embedding = None

        self.user_filed_size = 2
        self.item_filed_size = 2

        # implicit interaction network for user
        self.hidden_units_for_recon = hidden_units_for_recon
        self.activation_for_recon = activation_for_recon

        # self.User_SE = SENETLayer(self.user_filed_size, 3, seed, device)
        # self.Item_SE = SENETLayer(self.item_filed_size, 3, seed, device)

        # self.dense = torch.nn.Linear(128*len(self.user_dnn_feature_columns),1).cuda()
        #
        # self.user_col_dense = torch.nn.Linear(128, 128*len(self.user_dnn_feature_columns)).cuda()
        # self.item_col_dense = torch.nn.Linear(128, 128*len(self.item_dnn_feature_columns)).cuda()

    def forward(self, inputs):
        # user towel
        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)
            # print(user_sparse_embedding_list,len(user_sparse_embedding_list))
            # print(user_sparse_embedding_list[-1],user_sparse_embedding_list[-1].shape)

            # if torch.cuda.is_available():
            #     self.user_aug_vector = torch.rand(user_sparse_embedding_list[-1].shape).cuda()
            # else:
            #     self.user_aug_vector = torch.rand(user_sparse_embedding_list[-1].shape)
            # user_sparse_embedding_list.append(self.user_aug_vector)

            # implicit interaction user start
            target_recon_user_fc = combined_dnn_input(sparse_embedding_list=user_sparse_embedding_list, dense_value_list=[])
            target_recon_user = ImplicitInteraction(target_recon_user_fc.size()[-1], self.hidden_units_for_recon,
                                                    activation=self.activation_for_recon,
                                                    l2_reg=self.l2_reg_dnn, dropout_rate=self.dnn_dropout,
                                                    use_bn=self.dnn_use_bn, init_std=self.init_std, device=self.device)

            target_recon_output_for_user = target_recon_user(target_recon_user_fc)

            non_target_recon_user = ImplicitInteraction(target_recon_user_fc.size()[-1], self.hidden_units_for_recon,
                                                        activation=self.activation_for_recon,
                                                        l2_reg=self.l2_reg_dnn, dropout_rate=self.dnn_dropout,
                                                        use_bn=self.dnn_use_bn, init_std=self.init_std,
                                                        device=self.device)

            non_target_recon_output_for_user = non_target_recon_user(target_recon_user_fc)
            # implicit interaction user end

            user_sparse_embedding_list.append(torch.unsqueeze(target_recon_output_for_user, dim=1))
            user_sparse_embedding_list.append(torch.unsqueeze(non_target_recon_output_for_user, dim=1))

            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

            if len(self.user_dnn_feature_columns) > 0:
                # print(f"len(user_dnn_feature_columns):{len(self.user_dnn_feature_columns)}")
                user_dnn = DNN(user_dnn_input.size()[-1], self.dnn_hidden_units, activation=self.dnn_activation,
                               l2_reg=self.l2_reg_dnn, dropout_rate=self.dnn_dropout,
                               use_bn=self.dnn_use_bn, init_std=self.init_std, device=self.device)

            self.user_dnn_embedding = user_dnn(user_dnn_input)

        # item towel
        if len(self.item_dnn_feature_columns) > 0:
            item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)

            # implicit interaction user start
            target_recon_item_fc = combined_dnn_input(sparse_embedding_list=item_sparse_embedding_list,
                                                      dense_value_list=[])
            target_recon_item = ImplicitInteraction(target_recon_item_fc.size()[-1], self.hidden_units_for_recon,
                                                    activation=self.activation_for_recon,
                                                    l2_reg=self.l2_reg_dnn, dropout_rate=self.dnn_dropout,
                                                    use_bn=self.dnn_use_bn, init_std=self.init_std, device=self.device)

            target_recon_output_for_item = target_recon_item(target_recon_item_fc)

            non_target_recon_item = ImplicitInteraction(target_recon_item_fc.size()[-1], self.hidden_units_for_recon,
                                                        activation=self.activation_for_recon,
                                                        l2_reg=self.l2_reg_dnn, dropout_rate=self.dnn_dropout,
                                                        use_bn=self.dnn_use_bn, init_std=self.init_std,
                                                        device=self.device)

            non_target_recon_output_for_item = non_target_recon_item(target_recon_item_fc)
            # implicit interaction user end

            # if torch.cuda.is_available():
            #     self.item_aug_vector = torch.rand(item_sparse_embedding_list[-1].shape).cuda()
            # else:
            #     self.item_aug_vector = torch.rand(item_sparse_embedding_list[-1].shape)
            # item_sparse_embedding_list.append(self.item_aug_vector)

            item_sparse_embedding_list.append(torch.unsqueeze(target_recon_output_for_item, dim=1))
            item_sparse_embedding_list.append(torch.unsqueeze(non_target_recon_output_for_item, dim=1))
            item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
            # print(item_dnn_input.shape)

            if len(self.item_dnn_feature_columns) > 0:
                self.item_dnn = DNN(item_dnn_input.size()[-1], self.dnn_hidden_units,
                                    activation=self.dnn_activation, l2_reg=self.l2_reg_dnn,
                                    dropout_rate=self.dnn_dropout,
                                    use_bn=self.dnn_use_bn, init_std=self.init_std, device=self.device)
                self.item_dnn_embedding = None

            self.item_dnn_embedding = self.item_dnn(item_dnn_input)

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            score = Cosine_Similarity(self.user_dnn_embedding, self.item_dnn_embedding, gamma=self.gamma)
            output = self.out(score)
            return output, self.user_dnn_embedding, self.item_dnn_embedding, \
                target_recon_output_for_user, non_target_recon_output_for_user, \
                target_recon_output_for_item, non_target_recon_output_for_item
        elif len(self.user_dnn_feature_columns) > 0:
            return self.user_dnn_embedding
        elif len(self.item_dnn_feature_columns) > 0:
            return self.item_dnn_embedding

        else:
            raise Exception("input Error! user and item feature columns are empty.")
