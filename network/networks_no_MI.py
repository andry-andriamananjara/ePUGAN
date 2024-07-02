import os,sys
sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Conv1d,Conv2d
#from knn_cuda import KNN
#from pointnet2.pointnet2_utils import gather_operation,grouping_operation

import argparse
import torch
import torch.nn as nn
import numpy as np
#from models.utils import get_knn_pts, index_points
from einops import repeat, rearrange
#from models.pointops.functions import pointops
from pytorch3d.ops.knn import knn_points
from network.pu1k_args import parse_pu1k_args


'''
Library for Mamba
'''
import math
from typing import Optional
from mamba_ssm import Mamba

import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor

from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None




######################################### P3DConv

def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res


class Point3DConv(nn.Module):
    def __init__(self, args):
        super(Point3DConv, self).__init__()

        self.k = args.k
        self.args = args
        self.conv_delta = nn.Sequential(
            nn.Conv2d(3, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv_feats = nn.Sequential(
            nn.Conv2d(args.bn_size * args.growth_rate, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(args.growth_rate, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats, pts, knn_idx=None):
        # input: (b, c, n)

        if knn_idx == None:
            # (b, 3, n, k), (b, n, k)
            # knn_pts, knn_idx = get_knn_pts(self.k, pts, pts, return_idx=True)
            _,knn_idx, knn_pts = knn_points( pts,pts,K = self.k+1, return_nn = True,return_sorted = True)
            
        else:
            knn_pts = index_points(pts, knn_idx)
        # (b, 3, n, k)
        knn_delta = knn_pts - pts[..., None]

        # (b, c, n, k)
        knn_delta = self.conv_delta(knn_delta)
        #print('conv delta :: ',knn_delta.shape)
        
        # (b, c, n, k)
        knn_feats = index_points(feats, knn_idx)

        # (b, c, n, k)
        knn_feats = self.conv_feats(knn_feats)
        #print('Conv feats :: ',knn_feats.shape)
        
        # multiply: (b, c, n, k)
        new_feats = knn_delta * knn_feats
        # (b, c, n, k)
        new_feats = self.post_conv(new_feats)
        #print('Post_Conv :: ',new_feats.shape)
        
        # sum: (b, c, n)
        new_feats = new_feats.sum(dim=-1)
        #print(new_feats.shape)

        return new_feats


class DenseLayer(nn.Module):
    def __init__(self, args, input_dim):
        super(DenseLayer, self).__init__()

        self.conv_bottle = nn.Sequential(
            nn.Conv1d(input_dim, args.bn_size * args.growth_rate, 1),
            nn.BatchNorm1d(args.bn_size * args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.point_conv = Point3DConv(args)

    def forward(self, feats, pts, knn_idx=None):
        # input: (b, c, n)

        new_feats = self.conv_bottle(feats)
        # (b, c, n)
        new_feats = self.point_conv(new_feats, pts, knn_idx)
        # concat
        return torch.cat((feats, new_feats), dim=1)


class DenseUnit(nn.Module):
    def __init__(self, args):
        super(DenseUnit, self).__init__()

        self.dense_layers = nn.ModuleList([])
        for i in range(args.layer_num):
            self.dense_layers.append(DenseLayer(args, args.feat_dim + i * args.growth_rate))

    def forward(self, feats, pts, knn_idx=None):
        # input: (b, c, n)

        for dense_layer in self.dense_layers:
            new_feats = dense_layer(feats, pts, knn_idx)
            feats = new_feats
        return feats


class Transition(nn.Module):
    def __init__(self, args):
        super(Transition, self).__init__()

        input_dim = args.feat_dim + args.layer_num * args.growth_rate
        self.trans = nn.Sequential(
            nn.Conv1d(input_dim, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            #nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        # input: (b, c, n)

        new_feats = self.trans(feats)
        return new_feats


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()

        self.k = args.k
        self.conv_init = nn.Sequential(
            nn.Conv1d(3, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            nn.ReLU(inplace=True)
        )

        self.dense_blocks = nn.ModuleList([])
        for i in range(args.block_num):
            self.dense_blocks.append(nn.ModuleList([
                DenseUnit(args),
                Transition(args)
            ]))
        
        # below parameters are the same as the default
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 200, 1),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True)
        )
        
        parser = argparse.ArgumentParser(description='Model Arguments')
        parser.add_argument('--k', default=16, type=int, help='neighbor number')
        parser.add_argument('--layer_num', default=3, type=int, help='dense layer number in each dense block')
        parser.add_argument('--feat_dim', default=200, type=int, help='input(output) feature dimension in each dense block' )
        parser.add_argument('--growth_rate', default=128, type=int, help='input(output) feature dimension in each dense block' )
        parser.add_argument('--bn_size', default=1, type=int, help='the factor used in the bottleneck layer')
        args1,unknown = parser.parse_known_args()

        self.conv3 = nn.Sequential(
            nn.Conv1d(200, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        parser = argparse.ArgumentParser(description='Model Arguments')
        parser.add_argument('--k', default=16, type=int, help='neighbor number')
        parser.add_argument('--layer_num', default=3, type=int, help='dense layer number in each dense block')
        parser.add_argument('--feat_dim', default=128, type=int, help='input(output) feature dimension in each dense block' )
        parser.add_argument('--growth_rate', default=128, type=int, help='input(output) feature dimension in each dense block' )
        parser.add_argument('--bn_size', default=1, type=int, help='the factor used in the bottleneck layer')
        args2,unknown = parser.parse_known_args()

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        parser = argparse.ArgumentParser(description='Model Arguments')
        parser.add_argument('--k', default=16, type=int, help='neighbor number')
        parser.add_argument('--layer_num', default=3, type=int, help='dense layer number in each dense block')
        parser.add_argument('--feat_dim', default=128, type=int, help='input(output) feature dimension in each dense block' )
        parser.add_argument('--growth_rate', default=128, type=int, help='input(output) feature dimension in each dense block' )
        parser.add_argument('--bn_size', default=1, type=int, help='the factor used in the bottleneck layer')
        args3,unknown = parser.parse_known_args()

        self.dense_block1=nn.ModuleList([
                DenseUnit(args),#32
                Transition(args)#32
            ])

        self.dense_block2=nn.ModuleList([
                DenseUnit(args1),#200
                Transition(args1)#200
            ])

        self.dense_block3=nn.ModuleList([
                DenseUnit(args2),#128
                Transition(args2)#128
            ])

        self.dense_block4=nn.ModuleList([
                DenseUnit(args3),#128
                Transition(args3)#128
            ])
            
        self.dense1 = DenseUnit(args)    
        self.trans1 = Transition(args)   #32
        self.dense2 = DenseUnit(args1)   #200
        self.trans2 = Transition(args1)  #200
        self.dense3 = DenseUnit(args2)   #128
        self.trans3 = Transition(args2)  #128
        self.dense4 = DenseUnit(args3)   #128
        self.trans4 = Transition(args3)  #128
        
    def forward(self, pts):
        # input: (b, 3, n)
        
        # get knn_idx: (b, n, 3)
        pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
        # (b, m, k)
        #knn_idx = pointops.knnquery_heap(self.k, pts_trans, pts_trans).long()
        _,knn_idx,_ = knn_points( pts_trans, pts_trans,K = self.k+1, return_nn = True,return_sorted = True)
        
        # Local Features        
        # (b, c, n)
        init_feats = self.conv_init(pts)
        local_feats = []
        local_feats.append(init_feats)
        
        new_feats1=self.dense1(init_feats, pts, knn_idx)
        new_feats1=self.trans1(new_feats1)
        local_feats.append(new_feats1)
        
        new_feats1=self.conv2(new_feats1)
        new_feats2=self.dense2(new_feats1, pts, knn_idx)
        new_feats2=self.trans2(new_feats2)
        local_feats.append(new_feats2)
        
        new_feats3=self.conv3(new_feats2)
        new_feats3=self.dense3(new_feats3, pts, knn_idx)
        new_feats3=self.trans3(new_feats3)
        #print('FEAT 3 :: ',new_feats3.shape)
        local_feats.append(new_feats3)
        
        new_feats4=self.conv4(new_feats3)
        new_feats4=self.dense4(new_feats4, pts, knn_idx)
        new_feats4=self.trans4(new_feats4)
        #print('FEAT 4 :: ',new_feats4.shape)
        local_feats.append(new_feats4)        
        
        new_local_feats=torch.cat(local_feats,dim=1) #b,648,n
        #print('Local Feat :: ',new_local_feats.shape)
        
        # Global Features
        global_feats1 = new_feats4.max(dim=-1)[0]
        global_feats1 = global_feats1.unsqueeze(1).expand(-1, new_feats4.size(2), -1)#Expand global feature
        global_feats1 = global_feats1.permute(0,2,1)#b,128,n
        #print('Global Feat :: ',global_feats1.shape)
        
        # Add globa features in local features
        all_feats=torch.cat([new_local_feats, global_feats1],dim=1) #b,648,n
        #print('Local + Global : ', all_feats.shape)
        
        return all_feats



#########################################
#########################################

class get_edge_feature(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """
    def __init__(self,k=16):
        super(get_edge_feature,self).__init__()
        #self.KNN=KNN(k=k+1,transpose_mode=False)
        #self.KNN=knn_points(k+1)
        self.k=k

    def forward(self,point_cloud):
        '''
        idx is batch_size,k,n_points => batch_size,n_points,k (pytorch3d)
        point_cloud is batch_size,n_dims,n_points => batch_size,n_points,n_dims (pytorch3d) 
        point_cloud_neightbors is batch_size,n_dims,k,n_points
        '''
        
        #dist,idx=self.KNN(point_cloud,point_cloud)

        point_cloud = torch.transpose(point_cloud, 1, 2) #batch_size,n_points,n_dims
        # print('Point Cloud :: ', point_cloud.shape)
        #_,(N, P1, K, D),(N, P1, K, D)
        _,idx,point_cloud_neighbors = knn_points( point_cloud,point_cloud,K = self.k+1, return_nn = True,return_sorted = True)

        point_cloud_neighbors = point_cloud_neighbors.permute(0,3,2,1)
        idx = torch.transpose(idx, 1, 2) #batch_size,k,n_points
        
        #idx=idx[:,1:,:]
        #point_cloud_neighbors = grouping_operation(point_cloud,idx.contiguous().int())
        #point_cloud_neighbors = tp.grouping_operation(point_cloud, idx.contiguous().int())  # (B, 3, npoint, nsample)
        point_cloud_central=point_cloud.unsqueeze(2).repeat(1,1,self.k+1,1).permute(0,3,2,1)
        #print(point_cloud_central.shape,point_cloud_neighbors.shape)
        # print('Shape A :: ', point_cloud_central.shape)
        # print('Shape B :: ', (point_cloud_neighbors).shape)
        edge_feature=torch.cat([point_cloud_central,point_cloud_neighbors-point_cloud_central],dim=1)
        
        return edge_feature,idx



class denseconv(nn.Module):
    def __init__(self,growth_rate=64,k=16,in_channels=6,isTrain=True):
        super(denseconv,self).__init__()
        self.edge_feature_model=get_edge_feature(k=k)
        '''
        input to conv1 is batch_size,2xn_dims,k,n_points
        '''
        self.conv1=nn.Sequential(
            Conv2d(in_channels=in_channels,out_channels=growth_rate,kernel_size=[1,1]),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            Conv2d(in_channels=growth_rate+in_channels,out_channels=growth_rate,kernel_size=[1,1]),
            nn.ReLU()
        )
        self.conv3=nn.Sequential(
            Conv2d(in_channels=2*growth_rate+in_channels,out_channels=growth_rate,kernel_size=[1,1]),
        )
    def forward(self,input):
        '''
        y should be batch_size,in_channel,k,n_points
        '''
        # print('Dense conv Input :: ',input.shape)
        y,idx=self.edge_feature_model(input)
        inter_result=torch.cat([self.conv1(y),y],dim=1) #concat on feature dimension
        inter_result=torch.cat([self.conv2(inter_result),inter_result],dim=1)
        inter_result=torch.cat([self.conv3(inter_result),inter_result],dim=1)
        final_result=torch.max(inter_result,dim=2)[0] #pool the k channel
        return final_result,idx

class ssm_conv(nn.Module):
    def __init__(self, dim):

        super(ssm_conv, self).__init__()
        self.flat  = nn.Flatten(0,1)
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.ln    = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
        self.linear4 = nn.Linear(dim, dim)

    def forward(self, x):

        # x is of dimension batch x N x dim

        x1 = self.flat(x)
        xx = self.ln(x1)
        xx1 = F.sigmoid(self.linear1(xx))
        xx2 = F.sigmoid(self.linear2(xx))
        xx1 = xx1.view(x.shape[0],x.shape[1],x.shape[2])
        #xx1 = xx1.permute(0,2,1)
        xx1 = self.mamba(xx1)
        #xx1 = xx1.permute(0,2,1)
        
        xx1 = self.flat(xx1)
        xx1 = self.ln(xx1)
        xx  = xx1 * xx2
        xx  = self.linear3(xx)
        xx  = xx.view(x.shape[0],x.shape[1],x.shape[2])
        return xx + x

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction,self).__init__()
        self.growth_rate=24
        self.dense_n=3
        self.knn=16
        self.input_channel=3
        comp=self.growth_rate*2
        '''
        make sure to permute the input, the feature dimension is in the second one.
        input of conv1 is batch_size,num_dims,num_points
        '''
        self.conv1=nn.Sequential(
            Conv1d(in_channels=self.input_channel,out_channels=24,kernel_size=1,padding=0),
            nn.ReLU()
        )
        self.denseconv1=denseconv(in_channels=24*2,growth_rate=self.growth_rate)#return batch_size,(3*24+48)=120,num_points
        self.conv2=nn.Sequential(
            Conv1d(in_channels=144,out_channels=comp,kernel_size=1),
            nn.ReLU()
        )
        self.denseconv2=denseconv(in_channels=comp*2,growth_rate=self.growth_rate)
        self.conv3=nn.Sequential(
            Conv1d(in_channels=312,out_channels=comp,kernel_size=1),
            nn.ReLU()
        )
        self.denseconv3=denseconv(in_channels=comp*2,growth_rate=self.growth_rate)
        self.conv4=nn.Sequential(
            Conv1d(in_channels=480,out_channels=comp,kernel_size=1),
            nn.ReLU()
        )
        self.denseconv4=denseconv(in_channels=comp*2,growth_rate=self.growth_rate)
        
    def forward(self,input):
        #print('Input dim :: ',input.shape)
        l0_features=self.conv1(input) #b,24,n
        
        # print('Feaure Extraction l0 :: ',l0_features.shape)
        l1_features,l1_index=self.denseconv1(l0_features) #b,24*2+24*3=120,n
        l1_features=torch.cat([l1_features,l0_features],dim=1) #b,120+24=144,n

        l2_features=self.conv2(l1_features) #b,48,n
        l2_features,l2_index=self.denseconv2(l2_features) #b,48*2+24*3=168,n
        l2_features=torch.cat([l2_features,l1_features],dim=1)#b,168+144=312,n

        l3_features=self.conv3(l2_features)#b,48,n
        l3_features,l3_index=self.denseconv3(l3_features)#b,48*2+24*3=168,n
        l3_features=torch.cat([l3_features,l2_features],dim=1)#b,168+312=480,n

        l4_features=self.conv4(l3_features)#b,48,n
        l4_features,l4_index=self.denseconv4(l4_features)
        l4_features=torch.cat([l4_features,l3_features],dim=1)#b,168+480=648,n
        #print('Feaure Extraction l4 :: ',l4_features.shape)
        
        return l4_features

class Generator(nn.Module):
    def __init__(self,params=None):
        super(Generator,self).__init__()
        self.params=params
        self.pu1kparam=parse_pu1k_args()
        print(self.params)
        
        if self.params['feat_ext']=='P3DConv':
            self.feature_extractor  = FeatureExtractor(self.pu1kparam)
            print("Feature extractor XXXX :: P3DConv")
        else:
            self.feature_extractor = feature_extraction()
            print("Feature extractor XXXX :: Self")
            
        #self.up_ratio=params['up_ratio']
        #self.num_points=params['patch_num_point']
        #self.out_num_point=int(self.num_points*self.up_ratio)
        #self.up_projection_unit=up_projection_unit()
        self.up_projection_unit=up_projection_unit(params)
        
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=1),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=3,kernel_size=1)
        )
        
    def forward(self,input):
        
        features=self.feature_extractor(input) #b,648,n
        #print(features.shape)
        H=self.up_projection_unit(features) #b,128,4*n

        coord=self.conv1(H)
        coord=self.conv2(coord)
        
        return coord

class Generator_recon(nn.Module):
    def __init__(self,params):
        super(Generator_recon,self).__init__()
        self.feature_extractor=feature_extraction()
        self.up_ratio=params['up_ratio']
        self.num_points=params['patch_num_point']

        self.conv0=nn.Sequential(
            nn.Conv1d(in_channels=648,out_channels=128,kernel_size=1),
            nn.ReLU()
        )

        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=1),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=3,kernel_size=1)
        )
    def forward(self,input):
        features=self.feature_extractor(input) #b,648,n
        coord=self.conv0(features)
        coord=self.conv1(coord)
        coord=self.conv2(coord)
        return coord

class attention_unit(nn.Module):
    def __init__(self,in_channels=130):
        super(attention_unit,self).__init__()
        self.convF=nn.Sequential(
            Conv1d(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1),
            nn.ReLU()
        )
        self.convG = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels// 4, kernel_size=1),
            nn.ReLU()
        )
        self.convH = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU()
        )
        self.gamma=nn.Parameter(torch.tensor(torch.zeros([1]))).cuda()
        # self.gamma=nn.Parameter(torch.tensor(torch.zeros([1])))
    def forward(self,inputs):
        f=self.convF(inputs)
        g=self.convG(inputs)#b,32,n
        h=self.convH(inputs)
        s=torch.matmul(g.permute(0,2,1),f)#b,n,n
        beta=F.softmax(s,dim=2)#b,n,n

        o=torch.matmul(h,beta)#b,130,n

        x=self.gamma*o+inputs

        return x


class up_block(nn.Module):
    def __init__(self, params=None,up_ratio=4,in_channels=130):
        super(up_block,self).__init__()
        self.up_ratio=up_ratio
        self.params=params
        self.conv1=nn.Sequential(
            Conv1d(in_channels=in_channels,out_channels=256,kernel_size=1),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            Conv1d(in_channels=256,out_channels=128,kernel_size=1),
            nn.ReLU()
        )

        self.grid=torch.tensor(self.gen_grid(up_ratio)).cuda()
        #self.grid=torch.tensor(self.gen_grid(up_ratio))
        print('Gen XXXX :: ', self.params['gen_attention'])
        if self.params['gen_attention']=='mamba':
            self.mamba=Mamba(
                d_model  = in_channels, 
                d_state  = 16, #24, # SSM state expansion factor
                d_conv   = 4, #32, # Local convolution width
                expand   = 2, #10, 
                ).cuda()
        elif self.params['gen_attention']=='mamba2':
            self.ssm_mamba=ssm_conv(dim  = in_channels).cuda()
        else:
            self.attention_unit=attention_unit(in_channels=in_channels)
            
    def forward(self,inputs):
        net=inputs #b,128,n
        grid=self.grid.clone()
        grid=grid.unsqueeze(0).repeat(net.shape[0],1,net.shape[2])#b,4,2*n
        grid=grid.view([net.shape[0],-1,2])#b,4*n,2

        net=net.permute(0,2,1)#b,n,128
        net=net.repeat(1,self.up_ratio,1)#b,4n,128
        net = torch.cat([net, grid], dim=2)  # b,n*4,130

        if self.params['gen_attention']=='mamba':
            out = self.mamba(net)
            net = out+net
            net = net.permute(0,2,1)#b,130,n*4
        elif self.params['gen_attention']=='mamba2':
            net=self.ssm_mamba(net)
            net=net.permute(0,2,1)#b,130,n*4
        else:
            net=net.permute(0,2,1)#b,130,n*4
            net=self.attention_unit(net)
        
        net=self.conv1(net)
        net=self.conv2(net)
        
        return net


    def gen_grid(self,up_ratio):
        import math
        sqrted=int(math.sqrt(up_ratio))+1
        for i in range(1,sqrted+1).__reversed__():
            if (up_ratio%i)==0:
                num_x=i
                num_y=up_ratio//i
                break
        grid_x=torch.linspace(-0.2,0.2,num_x)
        grid_y=torch.linspace(-0.2,0.2,num_y)

        x,y=torch.meshgrid([grid_x,grid_y])
        grid=torch.stack([x,y],dim=-1)#2,2,2
        grid=grid.view([-1,2])#4,2
        return grid

class down_block(nn.Module):
    def __init__(self,up_ratio=4,in_channels=128):
        super(down_block,self).__init__()
        self.conv1=nn.Sequential(
            Conv2d(in_channels=in_channels,out_channels=256,kernel_size=[up_ratio,1],padding=0),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            Conv1d(in_channels=256,out_channels=128,kernel_size=1),
            nn.ReLU()
        )
        self.up_ratio=up_ratio
    def forward(self,inputs):
        net=inputs#b,128,n*4
        #net = torch.cat(
        #    [net[:, :, 0:1024].unsqueeze(2), net[:, :, 1024:2048].unsqueeze(2), net[:, :, 2048:3072].unsqueeze(2),
        #     net[:, :, 3072:4096].unsqueeze(2)], dim=2)
        net=net.view([inputs.shape[0],inputs.shape[1],self.up_ratio,-1])#b,128,4,n
        #net=torch.cat(torch.unbind(net,dim=2),dim=2)
        net=self.conv1(net)#b,256,1,n
        net=net.squeeze(2)
        net=self.conv2(net)
        return net


class up_projection_unit(nn.Module):
    def __init__(self,params=None,up_ratio=4):
        super(up_projection_unit,self).__init__()
        self.params=params
        self.conv1=nn.Sequential(
            Conv1d(in_channels=648, out_channels=128,kernel_size=1),
            nn.ReLU()
        )

        self.up_block1=up_block(params, up_ratio=4,in_channels=128+2)
        self.up_block2=up_block(params, up_ratio=4,in_channels=128+2)
        self.down_block=down_block(up_ratio=4,in_channels=128)
    def forward(self,input):
        L=self.conv1(input)#b,128,n
        H0=self.up_block1(L)#b,128,n*4
        L0=self.down_block(H0)#b,128,n

        #print(H0.shape,L0.shape,L.shape)
        E0=L0-L #b,128,n
        H1=self.up_block2(E0)#b,128,4*n
        H2=H0+H1 #b,128,4*n
        return H2

class mlp_conv(nn.Module):
    def __init__(self,in_channels,layer_dim):
        super(mlp_conv,self).__init__()
        self.conv_list=nn.ModuleList()
        for i,num_out_channel in enumerate(layer_dim[:-1]):
            if i==0:
                sub_module=nn.Sequential(
                    Conv1d(in_channels=in_channels, out_channels=num_out_channel, kernel_size=1),
                    nn.ReLU()
                )
                self.conv_list.append(sub_module)
            else:
                sub_module=nn.Sequential(
                    Conv1d(in_channels=layer_dim[i-1],out_channels=num_out_channel,kernel_size=1),
                    nn.ReLU()
                )
                self.conv_list.append(sub_module)
        self.conv_list.append(
            Conv1d(in_channels=layer_dim[-2],out_channels=layer_dim[-1],kernel_size=1)
        )
    def forward(self,inputs):
        net=inputs
        for module in self.conv_list:
            net=module(net)
        return net

class mlp(nn.Module):
    def __init__(self,in_channels,layer_dim):
        super(mlp,self).__init__()
        self.mlp_list=nn.ModuleList()
        for i,num_outputs in enumerate(layer_dim[:-1]):
            if i==0:
                sub_module=nn.Sequential(
                    nn.Linear(in_channels, num_outputs),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
            else:
                sub_module=nn.Sequential(
                    nn.Linear(layer_dim[i-1],num_outputs),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
        self.mlp_list.append(
            nn.Linear(layer_dim[-2],layer_dim[-1])
        )
    def forward(self,inputs):
        net=inputs
        for sub_module in self.mlp_list:
            net=sub_module(net)
        return net

class Discriminator(nn.Module):
    def __init__(self,params,in_channels):
        super(Discriminator,self).__init__()
        self.params=params
        self.start_number=32
        self.mlp_conv1=mlp_conv(in_channels=in_channels,layer_dim=[self.start_number, self.start_number * 2])
        self.mlp_conv2=mlp_conv(in_channels=self.start_number*4,layer_dim=[self.start_number*4,self.start_number*8])
        self.mlp=mlp(in_channels=self.start_number*8,layer_dim=[self.start_number * 8, 1])
        
        print('Dis XXXX :: ', self.params['dis_attention'])
        if self.params['dis_attention']=='mamba':
            self.mamba_dis=Mamba(
                d_model  = self.start_number*4, 
                d_state  = 16, # 24, # SSM state expansion factor
                d_conv   = 4, # 32, # Local convolution width
                expand   = 2, #10, 
                ).cuda()
        else:
            self.attention_unit=attention_unit(in_channels=self.start_number*4)
        
    def forward(self,inputs):
        features=self.mlp_conv1(inputs)
        features_global=torch.max(features,dim=2)[0] ##global feature
        features=torch.cat([features,features_global.unsqueeze(2).repeat(1,1,features.shape[2])],dim=1)

        #print('Before net shape :: ',features.shape)
        if self.params['dis_attention']=='mamba':
            features = features.permute(0,2,1)
            out      = self.mamba_dis(features)
            features = out+features
            features = features.permute(0,2,1)
        else:
            features=self.attention_unit(features)
        #print('After Net shape :: ',features.shape)

        features=self.mlp_conv2(features)
        features=torch.max(features,dim=2)[0]

        output=self.mlp(features)

        return output
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


if __name__=="__main__":
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Torch CUDA available :: ',device)
    params={
        "up_ratio":4,
        "patch_num_point":100,
        "feat_ext":"P3DConv",
        "scale":"arbitrary",
        #"feat_ext":"self",
        "gen_attention":"self",
        #"gen_attention":"mamba2",
        "dis_attention":"self",
    }
    generator=Generator(params).cuda()
    point_cloud=torch.rand(4,3,100).cuda()
    output=generator(point_cloud)
    print(output.shape)
    
    #discriminator=Discriminator(params,in_channels=3).cuda()
    #dis_output=discriminator(output)
    #print(dis_output.shape)