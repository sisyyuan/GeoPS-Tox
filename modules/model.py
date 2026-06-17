import torch
import math
from .chemistryProcess import graph_from_substructure
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogitsLoss



class Attention(torch.nn.Module):
    def __init__(self, Qdim, Kdim, Mdim):
        super(Attention, self).__init__()
        self.model_dim = Mdim
        self.WQ = torch.nn.Linear(Qdim, Mdim)
        self.WK = torch.nn.Linear(Kdim, Mdim)

    def forward(self, Q, K):
        Q, K = self.WQ(Q), self.WK(K)
        att = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        return att


class Framework(torch.nn.Module):
    def __init__(
        self, encoder, subencoder, num_class,
        base_dim, sub_dim, drop_ratio = 0.5,guide_model=False,
    ):
        super(Framework, self).__init__()
        self.encoder = encoder
        self.subencoder =  subencoder
        self.attn = Attention(
            base_dim, sub_dim, max(base_dim, sub_dim)
        )
        self.guide_model=guide_model
        self.num_class = num_class
        ###predicterlayers
        predictor_layers = [torch.nn.Linear(sub_dim, 2 * sub_dim)]
        if drop_ratio < 1 and drop_ratio > 0:
            predictor_layers.append(torch.nn.Dropout(drop_ratio))
        predictor_layers.append(torch.nn.ReLU())
        predictor_layers.append(torch.nn.Linear(2 * sub_dim, num_class))
        self.predictor_head0 = torch.nn.Sequential(*predictor_layers)
        self.predictor_head1 = torch.nn.Sequential(*predictor_layers)
        self.predictor_head2 = torch.nn.Sequential(*predictor_layers)
        self.predictor_head3 = torch.nn.Sequential(*predictor_layers)
        self.last_layer0 = nn.Linear(base_dim, 1)
        self.last_layer1 = nn.Linear(base_dim, 1)
        self.last_layer2 = nn.Linear(base_dim, 1)
        self.last_layer3 = nn.Linear(base_dim, 1)
        #### soft_predict_layersbase_dim
        soft_predictor_layers = [torch.nn.Linear(sub_dim, 2 * sub_dim)]
        if drop_ratio < 1 and drop_ratio > 0:
            soft_predictor_layers.append(torch.nn.Dropout(drop_ratio))
        soft_predictor_layers.append(torch.nn.ReLU())
        soft_predictor_layers.append(torch.nn.Linear(2 * sub_dim, num_class))
        self.fusion = torch.nn.Linear(2, 1)
        self.soft_predictor_head = torch.nn.Sequential(*soft_predictor_layers) 
        # environment_layers = [torch.nn.Linear(sub_dim, 2 * sub_dim)]
        # if drop_ratio < 1 and drop_ratio > 0:
        #     environment_layers.append(torch.nn.Dropout(drop_ratio))
        # environment_layers.append(torch.nn.ReLU())
        # environment_layers.append(torch.nn.Linear(2 * sub_dim, num_class))
        # self.environment_head = torch.nn.Sequential(*environment_layers)     

    def sub_feature_from_graphs(self, smiles,subs, device, return_mask=False, return_subidx=False):
        # 从子结构图获取数据，增加sub_to_idx的返回
        substructure_graph, mask, sub_to_idx = graph_from_substructure(smiles,subs, True, 'pyg', True)
        # 将图结构放置到指定设备
        substructure_graph = substructure_graph.to(device)
        # 子结构编码得到特征
        substructure_feat = self.subencoder(substructure_graph)
        # 根据条件返回不同的数据组合
        if return_subidx:
            return substructure_feat, mask, sub_to_idx
        elif return_mask:
            return substructure_feat, mask
        else:
            return substructure_feat
    def sub_feature_from_graphs_egnn(self, batch_data,subs, device, return_mask=False, return_subidx=False):
        # 从子结构图获取数据，增加sub_to_idx的返回
        graph_feat = self.encoder(batch_data)
        substructure_graph, mask, sub_to_idx = graph_from_substructure(batch_data.smiles,subs, True, 'pyg', True)
        # 将图结构放置到指定设备
        substructure_graph = substructure_graph.to(device)
        # 子结构编码得到特征
        substructure_feat = self.subencoder(substructure_graph)
        # # 根据条件返回不同的数据组合
        if return_subidx:
            return substructure_feat, mask, sub_to_idx
        elif return_mask:
            return graph_feat,substructure_feat, mask
        else:
            return graph_feat
    def forward(self, substructures, batch_data, head = None, reverse_att = False,y_hat=None,loss=None,type='gin'):
        #### 得到图级别的fea和子结构的fea，再做attention
        if type=='gin':
            s = self.encoder(batch_data)
            ## key
            # substructure_feat, att_mask= self.sub_feature_from_graphs(smiles=batch_data.smiles,
            #     subs = substructures, device = batch_data.x.device,
            #     return_mask = True,return_subidx=False
            #     )
        else:
            # graph_feat = self.encoder(batch_data)
            ### key
            graph_feat,substructure_feat, att_mask = self.sub_feature_from_graphs_egnn(batch_data=batch_data,
                                                                       subs=substructures, device=batch_data.x.device,
                                                                       return_mask=True, return_subidx=False
                                                                       )


            att_mask = torch.from_numpy(att_mask).to(batch_data.x.device)
            ####0
            att_mask = torch.logical_not(att_mask)
            att = self.attn(Q = graph_feat, K = substructure_feat)
            # # if head == 'environment':
            # #     if reverse_att:
            # #         att = -att
            # #         att = torch.masked_fill(att, att_mask, -(1 << 32))
            # #     activation = torch.softmax(att, dim = -1)
            # #     molecule_feat = torch.matmul(activation, substructure_feat)
            # #     result = self.environment_head(molecule_feat)
            # #     return result
            # ### 在前面先取 反逻辑，就把mask矩阵里面的false变true，代表该分子里面没有出现的子结构，再把这些子结构mask掉。
            if att_mask is not None:
                att = torch.masked_fill(att, att_mask, -(1 << 32))
            activation = torch.softmax(att, dim = -1)
            s = torch.matmul(activation, substructure_feat)
            # # ##### 编写soft头

        if self.guide_model:
            if self.num_class > 1:
                xy = self.soft_predictor_head(molecule_feat.unsqueeze(dim=-2) * y_hat.unsqueeze(dim=-1))
            else:
                xy = self.soft_predictor_head(molecule_feat * y_hat)  ###h_graph 代表伪标签可靠度，
            loss = loss.unsqueeze(dim=-1)
            output = torch.cat([xy, loss], dim=-1)
            output = self.fusion(output)
            # result = self.soft_predictor_head(molecule_feat)
            return torch.sigmoid(output)

        # # else:
        tag=batch_data.tag
        batch=batch_data.batch
        output = torch.zeros(len(s), device=s.device).unsqueeze(1)

        for t in range(4):
            mask = (tag== t)
            # mask1 = (tag == t)
            # 找到当前 tag 的样本索引
            if mask.any():  # 如果有样本属于当前 tag
                subset = s[mask]  # 提取当前 tag 对应的子集
                if t == 0:
                    output[mask] = self.last_layer0(subset)
                elif t == 1:
                    output[mask] = self.last_layer1(subset)
                elif t == 2:
                    output[mask] = self.last_layer2(subset)
                elif t == 3:
                    output[mask] = self.last_layer3(subset)

        # output = output * self.y_std + self.y_mean
        #
        #     for t in batch_data.tag.unique():
        #         mask = (batch_data.tag == t)
        #         # mask1 = (tag == t)
        #         # 找到当前 tag 的样本索引
        #         if mask.any():  # 如果有样本属于当前 tag
        #
        #             subset = graph_feat[mask]  # 提取当前 tag 对应的子集
        #             if t == 0:
        #                 output[mask] = self.predictor_head0(subset)
        #             elif t == 1:
        #                 output[mask] = self.predictor_head1(subset)
        #             elif t == 2:
        #                 output[mask] = self.predictor_head2(subset)
        #             elif t == 3:
        #                 output[mask] = self.predictor_head3(subset)


        return output
    def predict(self, substructures, batch_data, head = None, reverse_att = False,y_hat=None,loss=None):
        #### 得到图级别的fea和子结构的fea，再做attention
        graph_feat = self.encoder(batch_data)
        ### key
        substructure_feat, att_mask,sub_idx = self.sub_feature_from_graphs(
            smiles=batch_data.smiles, subs = substructures, device = batch_data.x.device,
            return_mask = True,return_subidx=True
            )
        att_mask = torch.from_numpy(att_mask).to(batch_data.x.device)
        ####
        att_mask = torch.logical_not(att_mask)
        att = self.attn(Q = graph_feat, K = substructure_feat)
        # if head == 'environment':
        #     if reverse_att:
        #         att = -att
        #         att = torch.masked_fill(att, att_mask, -(1 << 32))
        #     activation = torch.softmax(att, dim = -1)
        #     molecule_feat = torch.matmul(activation, substructure_feat)
        #     result = self.environment_head(molecule_feat)
        #     return result
        ### 在前面先取 反逻辑，就把mask矩阵里面的false变true，代表该分子里面没有出现的子结构，再把这些子结构mask掉。
        if att_mask is not None:
            att = torch.masked_fill(att, att_mask, -(1 << 32))
        activation = torch.softmax(att, dim = -1)
        molecule_feat = torch.matmul(activation, substructure_feat)
        tag = batch_data.tag
        result = torch.zeros(len(molecule_feat), device=molecule_feat.device).unsqueeze(1)
        for t in range(4):
            mask = (tag == t)
            if mask.any():
                layer = getattr(self, f"last_layer{t}")
                result[mask] = layer(molecule_feat[mask])
        return result, activation,sub_idx
    

def DeviationLoss(preds, y, device):
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = BCEWithLogitsLoss(preds * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph = True)[0]
    return torch.sum(grad**2)


import math
from math import pi
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import Embedding

from torch_geometric.nn import radius_graph
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter


def nan_to_num(vec, num=0.0):
    idx = torch.isnan(vec)
    vec[idx] = num
    return vec


def _normalize(vec, dim=-1):
    return nan_to_num(
        torch.div(vec, torch.norm(vec, dim=dim, keepdim=True)))


def swish(x):
    return x * torch.sigmoid(x)


## radial basis function to embed distances
class rbf_emb(nn.Module):
    def __init__(self, num_rbf, soft_cutoff_upper, rbf_trainable=False):
        super().__init__()
        self.soft_cutoff_upper = soft_cutoff_upper
        self.soft_cutoff_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value)) ** -2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        soft_cutoff = 0.5 * \
                      (torch.cos(dist * pi / self.soft_cutoff_upper) + 1.0)
        soft_cutoff = soft_cutoff * (dist < self.soft_cutoff_upper).float()
        return soft_cutoff * torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class NeighborEmb(MessagePassing):
    def __init__(self, hid_dim: int):
        super(NeighborEmb, self).__init__(aggr='add')
        self.embedding = nn.Embedding(95, hid_dim)
        self.hid_dim = hid_dim

    def forward(self, z, s, edge_index, embs):
        s_neighbors = self.embedding(z)
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)

        s = s + s_neighbors
        return s

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j


class S_vector(MessagePassing):
    def __init__(self, hid_dim: int):
        super(S_vector, self).__init__(aggr='add')
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU())

    def forward(self, s, v, edge_index, emb):
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v

        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class EquiMessagePassing(MessagePassing):
    def __init__(
            self,
            hidden_channels,
            num_radial,
    ):
        super(EquiMessagePassing, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.num_radial = num_radial
        self.inv_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_channels + self.num_radial, self.hidden_channels * 3), nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 3), )

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, edge_rbf, weight, edge_vector):
        xh = self.x_proj(x)

        rbfh = self.rbf_proj(edge_rbf)
        weight = self.inv_proj(weight)
        rbfh = rbfh * weight
        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class FTE(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.equi_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xequi_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.equi_proj.weight)
        nn.init.xavier_uniform_(self.xequi_proj[0].weight)
        self.xequi_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xequi_proj[2].weight)
        self.xequi_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec = self.equi_proj(vec)
        vec1, vec2 = torch.split(
            vec, self.hidden_channels, dim=-1
        )
        # scalrization = torch.sum(vec1.unsqueeze(2) , dim=1)
        # scalrization = torch.sum(vec1.unsqueeze(2) * node_frame.unsqueeze(-1), dim=1)
        # scalrization = torch.abs(scalrization.clone())
        scalar = torch.norm(vec1, dim=-2)  # torch.sqrt(torch.sum(vec1 ** 2, dim=-2))

        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = vec_dot * self.inv_sqrt_h

        x_vec_h = self.xequi_proj(
            torch.cat(
                [x, scalar], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        dx = xvec1 + xvec2 + vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec2

        return dx, dvec


class aggregate_pos(MessagePassing):

    def __init__(self, aggr='mean'):
        super(aggregate_pos, self).__init__(aggr=aggr)

    def forward(self, vector, edge_index):
        v = self.propagate(edge_index, x=vector)

        return v


class EquiOutput(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                # GatedEquivariantBlock(
                #     hidden_channels,
                #     hidden_channels // 2,
                # ),
                GatedEquivariantBlock(hidden_channels, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
            self,
            hidden_channels,
            out_channels,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = nn.SiLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v


class LEFTNet(torch.nn.Module):
    r"""
        LEFTNet

        Args:
            pos_require_grad (bool, optional): If set to :obj:`True`, will require to take derivative of model output with respect to the atomic positions. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`32`)
            y_mean (float, optional): Mean value of the labels of training data. (default: :obj:`0`)
            y_std (float, optional): Standard deviation of the labels of training data. (default: :obj:`1`)

    """

    def __init__(
            self, pos_require_grad=False, cutoff=5.0, num_layers=4,
            hidden_channels=128, num_radial=32, y_mean=0, y_std=1, **kwargs):
        super(LEFTNet, self).__init__()
        self.y_std = y_std
        self.y_mean = y_mean
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff

        self.z_emb = Embedding(95, hidden_channels)
        self.radial_emb = rbf_emb(num_radial, self.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        self.neighbor_emb = NeighborEmb(hidden_channels)

        self.S_vector = S_vector(hidden_channels)

        self.lin = nn.Sequential(
            nn.Linear(3, hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels // 4, 1))

        self.message_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(
                EquiMessagePassing(hidden_channels, num_radial).jittable()
            )
            self.FTEs.append(FTE(hidden_channels))

        self.last_layer0 = nn.Linear(hidden_channels, 1)
        self.last_layer1 = nn.Linear(hidden_channels, 1)
        self.last_layer2 = nn.Linear(hidden_channels, 1)
        self.last_layer3 = nn.Linear(hidden_channels, 1)

        # for node-wise frame
        self.mean_neighbor_pos = aggregate_pos(aggr='mean')

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()

    def reset_parameters(self):
        self.radial_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        # self.last_layer.reset_parameters()
        for layer in self.radial_lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, batch_data, mask1=None):
        z, pos, batch = batch_data.z, batch_data.posc, batch_data.batch

        # embed z
        z_emb = self.z_emb(z)

        # construct edges based on the cutoff value
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        i, j = edge_index

        # embed pair-wise distance
        dist = torch.norm(pos[i] - pos[j], dim=-1)
        # radial_emb shape: (num_edges, num_radial), radial_hidden shape: (num_edges, hidden_channels)
        radial_emb = self.radial_emb(dist)
        radial_emb=radial_emb.float()
        radial_hidden = self.radial_lin(radial_emb)

        soft_cutoff = 0.5 * (torch.cos(dist * pi / self.cutoff) + 1.0)
        radial_hidden = soft_cutoff.unsqueeze(-1) * radial_hidden

        # init invariant node features
        # shape: (num_nodes, hidden_channels)
        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)

        # init equivariant node features
        # shape: (num_nodes, 3, hidden_channels)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)

        # bulid edge-wise frame
        edge_diff = pos[i] - pos[j]
        edge_diff = _normalize(edge_diff)
        edge_cross = torch.cross(pos[i], pos[j])
        edge_cross = _normalize(edge_cross)
        edge_vertical = torch.cross(edge_diff, edge_cross)
        # edge_frame shape: (num_edges, 3, 3)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)

        # build node-wise frame
        # mean_neighbor_pos = self.mean_neighbor_pos(pos, edge_index)
        # node_diff = pos - mean_neighbor_pos
        # node_diff = _normalize(node_diff)
        # node_cross = torch.cross(pos, mean_neighbor_pos)
        # node_cross = _normalize(node_cross)
        # node_vertical = torch.cross(node_diff, node_cross)
        # # node_frame shape: (num_nodes, 3, 3)
        # node_frame = torch.cat((node_diff.unsqueeze(-1), node_cross.unsqueeze(-1), node_vertical.unsqueeze(-1)), dim=-1)

        # LSE: local 3D substructure encoding
        # S_i_j shape: (num_nodes, 3, hidden_channels)
        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        scalar3 = (self.lin(torch.permute(scalrization1, (0, 2, 1))) + torch.permute(scalrization1, (0, 2, 1))[:, :,
                                                                       0].unsqueeze(2)).squeeze(-1)
        scalar4 = (self.lin(torch.permute(scalrization2, (0, 2, 1))) + torch.permute(scalrization2, (0, 2, 1))[:, :,
0].unsqueeze(2)).squeeze(-1)

        A_i_j = torch.cat((scalar3, scalar4), dim=-1) * soft_cutoff.unsqueeze(-1)
        A_i_j = torch.cat((A_i_j, radial_hidden, radial_emb), dim=-1)

        for i in range(self.num_layers):
            # equivariant message passing
            ds, dvec = self.message_layers[i](
                s, vec, edge_index, radial_emb, A_i_j, edge_diff
            )
            s = s + ds
            vec = vec + dvec
            # FTE: frame transition encoding
            ds, dvec = self.FTEs[i](s, vec)
            s = s + ds
            vec = vec + dvec
        #
        # for t in range(4):
        #     mask = (tag[batch] == t)
        #     # mask1 = (tag == t)
        #     # 找到当前 tag 的样本索引
        #     if mask.any():  # 如果有样本属于当前 tag
        #         subset = s[mask]  # 提取当前 tag 对应的子集
        #         if t == 0:
        #             output[mask] = self.last_layer0(subset)
        #         elif t == 1:
        #             output[mask] = self.last_layer1(subset)
        #         elif t == 2:
        #             output[mask] = self.last_layer2(subset)
        #         elif t == 3:
        #             output[mask] = self.last_layer3(subset)
        # output = scatter(output, batch, dim=0)
        # output = output * self.y_std + self.y_mean
        s = scatter(s, batch, dim=0)

        return s

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
