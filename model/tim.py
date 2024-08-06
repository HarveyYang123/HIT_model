"""



"""
import torch
import torch.nn as nn

from model.base_tower import BaseTower
from model.dual_tower_for_tim import DualTowerForTim
from layers.interaction import SENETLayer, ImplicitInteraction
from layers.core import DNN
from layers.activation import activation_layer
from layers.interaction import LightSE
from layers.core import User_Fe_DNN,Item_Fe_DNN
from layers.attention import MultiHeadAttention
from preprocessing.utils import fe_score
from preprocessing.utils import Cosine_Similarity
from preprocessing.utils import col_score
from preprocessing.utils import col_score_2
from preprocessing.utils import single_score
from preprocessing.inputs import combined_dnn_input, compute_input_dim



class TimTower(DualTowerForTim):
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, user_input_for_recon, item_input_for_recon,
                 gamma=1, dnn_use_bn=True, dnn_hidden_units=(300, 300, 32), field_dim = 16, user_head=2,item_head=2,
                 dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-5,
                 dnn_dropout=0, init_std=0.0001, seed=124, task='binary', device='cpu', gpus=None, user_filed_size = 5,
                 item_filed_size = 2, hidden_units_for_recon=(32, 32), activation_for_recon='relu',
                 use_target=True, use_non_target=True, only_output_fe=True, use_mha=True):
        super(TimTower, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                       l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                       device=device, gpus=gpus, use_target=True, use_non_target=True)

        self.item_dnn_feature_columns = item_dnn_feature_columns
        self.user_dnn_feature_columns = user_dnn_feature_columns
        self.user_input_for_recon = user_input_for_recon
        self.item_input_for_recon = item_input_for_recon
        self.use_mha = use_mha
        print("self.user_input_for_recon: ", self.user_input_for_recon)
        print("self.item_input_for_recon: ", self.item_input_for_recon)

        self.user_head = user_head
        self.item_head = item_head
        self.field_dim = field_dim
        self.only_output_fe = only_output_fe

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
        # self.user_dnn_embedding = None

        self.user_filed_size = user_filed_size
        self.item_filed_size = item_filed_size
        self.User_sim_non_local = LightSE(self.user_filed_size, seed, device)
        self.Item_sim_non_local = LightSE(self.item_filed_size,  seed, device)

        # implicit interaction network for user
        self.hidden_units_for_recon = hidden_units_for_recon
        self.activation_for_recon = activation_for_recon
        self.use_target = use_target
        self.use_non_target = use_non_target

        input_user_dim = compute_input_dim(user_dnn_feature_columns)
        input_item_dim = compute_input_dim(item_dnn_feature_columns)
        self.query_dim = 0
        if self.use_target:
            self.target_recon_user = ImplicitInteraction(compute_input_dim(user_input_for_recon),
                                                         self.hidden_units_for_recon,
                                                         activation=self.activation_for_recon, device=self.device)
            self.target_recon_item = ImplicitInteraction(compute_input_dim(item_input_for_recon),
                                                         self.hidden_units_for_recon,
                                                         activation=self.activation_for_recon, device=self.device)
            input_user_dim += self.hidden_units_for_recon[-1]
            self.query_dim += self.hidden_units_for_recon[-1]
            input_item_dim += self.hidden_units_for_recon[-1]


        if self.use_non_target:
            self.non_target_recon_user = ImplicitInteraction(compute_input_dim(user_input_for_recon),
                                                             self.hidden_units_for_recon,
                                                             activation=self.activation_for_recon, device=self.device)
            self.non_target_recon_item = ImplicitInteraction(compute_input_dim(item_input_for_recon),
                                                             self.hidden_units_for_recon,
                                                             activation=self.activation_for_recon, device=self.device)
            input_user_dim += self.hidden_units_for_recon[-1]
            self.query_dim += self.hidden_units_for_recon[-1]
            input_item_dim += self.hidden_units_for_recon[-1]

        if len(user_dnn_feature_columns) > 0:
            if (self.use_non_target or self.use_target) and self.use_mha:
                self.user_dnn = DNN(input_user_dim, hidden_units=[200],
                                    activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                    use_bn=dnn_use_bn, init_std=init_std, device=device)

                self.user_query_dnn = DNN(self.query_dim, hidden_units=[200],
                                    activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                    use_bn=dnn_use_bn, init_std=init_std, device=device)

                self.user_mha = MultiHeadAttention(model_dim=200, num_heads=8, dropout=self.dnn_dropout, device=device)
                self.user_fe_dnn = User_Fe_DNN(200, field_dim, dnn_hidden_units,
                                               activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                               use_bn=dnn_use_bn, user_head=user_head, init_std=init_std,
                                               use_kan=False, only_output_fe=self.only_output_fe, device=device)
            else:
                self.user_fe_dnn = User_Fe_DNN(input_user_dim, field_dim, dnn_hidden_units,
                                               activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                               use_bn=dnn_use_bn, user_head=user_head, init_std=init_std,
                                               use_kan=False, only_output_fe=self.only_output_fe, device=device)

            self.user_dnn_embedding = None

        if len(item_dnn_feature_columns) > 0:
            if (self.use_non_target or self.use_target) and self.use_mha:
                self.item_dnn = DNN(input_item_dim, hidden_units=[200],
                                    activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                    use_bn=dnn_use_bn, init_std=init_std, device=device)
                self.item_query = DNN(self.hidden_units_for_recon[-1], hidden_units=[200],
                                      activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                      use_bn=dnn_use_bn, init_std=init_std, device=device)
                self.item_mha = MultiHeadAttention(model_dim=200, num_heads=8, dropout=self.dnn_dropout, device=device)
                self.item_fe_dnn = Item_Fe_DNN(200, field_dim, dnn_hidden_units,
                                               activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                               use_bn=dnn_use_bn, item_head=item_head, init_std=init_std, device=device)

            else:
                self.item_fe_dnn = Item_Fe_DNN(input_item_dim, field_dim, dnn_hidden_units,
                                               activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                               use_bn=dnn_use_bn, item_head=item_head, init_std=init_std, device=device)

            self.item_dnn_embedding = None
        # self.User_SE = SENETLayer(self.user_filed_size, 3, seed, device)
        # self.Item_SE = SENETLayer(self.item_filed_size, 3, seed, device)

        # self.dense = torch.nn.Linear(128*len(self.user_dnn_feature_columns),1).cuda()
        #
        # self.user_col_dense = torch.nn.Linear(128, 128*len(self.user_dnn_feature_columns)).cuda()
        # self.item_col_dense = torch.nn.Linear(128, 128*len(self.item_dnn_feature_columns)).cuda()

    def forward(self, inputs):
        # user tower
        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)

            # initialization
            target_recon_output_for_user = torch.zeros(user_sparse_embedding_list[-1].shape)
            non_target_recon_output_for_user = torch.zeros(user_sparse_embedding_list[-1].shape)
            if torch.cuda.is_available():
                target_recon_output_for_user = target_recon_output_for_user.cuda()
                non_target_recon_output_for_user = non_target_recon_output_for_user.cuda()
            # print(user_sparse_embedding_list,len(user_sparse_embedding_list))
            # print(user_sparse_embedding_list[-1],user_sparse_embedding_list[-1].shape)

            # implicit interaction user start
            user_sparse_embedding_list_for_recon, user_dense_value_list_for_recon = \
                self.input_from_feature_columns(inputs, self.user_input_for_recon, self.user_embedding_dict)
            target_recon_user_fc = combined_dnn_input(sparse_embedding_list=user_sparse_embedding_list_for_recon,
                                                      dense_value_list=user_dense_value_list_for_recon)
            user_query_embedding_list = []
            if self.use_target:
                target_recon_output_for_user = self.target_recon_user(target_recon_user_fc)

                target_recon_output_for_user_new = target_recon_output_for_user.detach()
                # target_recon_output_for_user_new = torch.nn.functional.normalize(target_recon_output_for_user_new, p=2, dim=-1)
                user_sparse_embedding_list.append(torch.unsqueeze(target_recon_output_for_user_new, dim=1))
                user_query_embedding_list.append(torch.unsqueeze(target_recon_output_for_user_new, dim=1))

            if self.use_non_target:
                non_target_recon_output_for_user = self.non_target_recon_user(target_recon_user_fc)

                non_target_recon_output_for_user_new = non_target_recon_output_for_user.detach()
                # non_target_recon_output_for_user_new = torch.nn.functional.normalize(non_target_recon_output_for_user_new, p=2,
                #                                                               dim=-1)
                user_sparse_embedding_list.append(torch.unsqueeze(non_target_recon_output_for_user_new, dim=1))
                user_query_embedding_list.append(torch.unsqueeze(non_target_recon_output_for_user_new, dim=1))

                # implicit interaction user end

            # user_sparse_embedding = torch.cat(user_sparse_embedding_list, dim=1)
            # User_sim_embedding = self.User_sim_non_local(user_sparse_embedding)
            # sparse_dnn_input = torch.flatten(User_sim_embedding, start_dim=1)
            # if(len(user_dense_value_list)>0):
            #     dense_dnn_input = torch.flatten(torch.cat(user_dense_value_list, dim=-1), start_dim=1)
            #     user_dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input],axis=-1)
            # else:
            #     user_dnn_input = sparse_dnn_input


            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
            if (self.use_non_target or self.use_target) and self.use_mha:
                user_dnn_embed = self.user_dnn(user_dnn_input)
                query_embed = torch.flatten(torch.cat(user_query_embedding_list, dim=-1), start_dim=1)
                user_query = self.user_query_dnn(query_embed)
                user_mha_output, _ = self.user_mha(key=user_dnn_embed, value=user_dnn_embed, query=user_query)
                self.user_fe_rep = self.user_fe_dnn(user_mha_output)
            else:
                self.user_fe_rep = self.user_fe_dnn(user_dnn_input)

            self.user_dnn_embedding = self.user_fe_rep[-1]

        # item tower
        if len(self.item_dnn_feature_columns) > 0:
            item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)

            # initialization
            target_recon_output_for_item = torch.zeros(item_sparse_embedding_list[-1].shape)
            non_target_recon_output_for_item = torch.zeros(item_sparse_embedding_list[-1].shape)
            if torch.cuda.is_available():
                target_recon_output_for_item = target_recon_output_for_item.cuda()
                non_target_recon_output_for_item = non_target_recon_output_for_item.cuda()

            # implicit interaction user start
            item_sparse_embedding_list_for_recon, item_dense_value_list_for_recon = \
                self.input_from_feature_columns(inputs, self.item_input_for_recon, self.item_embedding_dict)
            target_recon_item_fc = combined_dnn_input(sparse_embedding_list=item_sparse_embedding_list_for_recon,
                                                      dense_value_list=item_dense_value_list_for_recon)

            item_query_embedding_list = []
            if self.use_target:
                target_recon_output_for_item = self.target_recon_item(target_recon_item_fc)

                target_recon_output_for_item_new = target_recon_output_for_item.detach()
                # target_recon_output_for_item_new = torch.nn.functional.normalize(
                #     target_recon_output_for_item_new, p=2,
                #     dim=-1)
                item_sparse_embedding_list.append(torch.unsqueeze(target_recon_output_for_item_new, dim=1))
                item_query_embedding_list.append(torch.unsqueeze(target_recon_output_for_item_new, dim=1))

            if self.use_non_target:
                non_target_recon_output_for_item = self.non_target_recon_item(target_recon_item_fc)

                non_target_recon_output_for_item_new = non_target_recon_output_for_item.detach()
                # non_target_recon_output_for_item_new = torch.nn.functional.normalize(
                #     non_target_recon_output_for_item_new, p=2,
                #     dim=-1)
                item_sparse_embedding_list.append(torch.unsqueeze(non_target_recon_output_for_item_new, dim=1))
                item_query_embedding_list.append(torch.unsqueeze(non_target_recon_output_for_item_new, dim=1))

            # implicit interaction user end
            # item_sparse_embedding = torch.cat(item_sparse_embedding_list, dim=1)
            # Item_sim_embedding = self.Item_sim_non_local(item_sparse_embedding)
            # sparse_dnn_input = torch.flatten(Item_sim_embedding, start_dim=1)
            # dense_dnn_input = torch.flatten(torch.cat(item_dense_value_list, dim=-1), start_dim=1)
            # item_dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], axis=-1)

            item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
            # self.item_dnn_embedding = self.item_dnn(item_dnn_input)

            if (self.use_non_target or self.use_target) and self.use_mha:
                item_dnn_embed = self.item_dnn(item_dnn_input)
                item_query_embed = torch.flatten(torch.cat(item_query_embedding_list, dim=-1), start_dim=1)
                item_query = self.user_query_dnn(item_query_embed)

                item_mha_output, _ = self.item_mha(key=item_dnn_embed, value=item_dnn_embed, query=item_query)
                self.item_fe_rep = self.item_fe_dnn(item_mha_output)
            else:
                self.item_fe_rep = self.item_fe_dnn(item_dnn_input)


            self.item_dnn_embedding = self.item_fe_rep[-1]

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            # score = Cosine_Similarity(self.user_dnn_embedding, self.item_dnn_embedding, gamma=self.gamma)
            # output = self.out(score)
            field_dim_len = len(self.dnn_hidden_units)
            if self.only_output_fe:
                field_dim_len = 1

            score = fe_score(self.user_fe_rep, self.item_fe_rep, self.user_head, self.item_head,
                             [self.field_dim] * field_dim_len,
                             [self.field_dim] * field_dim_len)

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
