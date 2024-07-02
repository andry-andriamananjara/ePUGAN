'''
@InProceedings{He_2023_CVPR,
    author    = {He, Yun and Tang, Danhang and Zhang, Yinda and Xue, Xiangyang and Fu, Yanwei},
    title     = {Grad-PU: Arbitrary-Scale Point Cloud Upsampling via Gradient Descent with Learned Distance Functions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
'''
import os,sys
sys.path.append("../")
import argparse
import torch
import torch.nn as nn
import numpy as np
#from models.utils import get_knn_pts, index_points
from einops import repeat, rearrange
#from models.pointops.functions import pointops
from pytorch3d.ops.knn import knn_points
from network.pu1k_args import parse_pu1k_args

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


        self.dense1 = DenseUnit(args)    #32
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
        
        new_local_feats=torch.cat(local_feats,dim=1) #b,520,n
        print('Local Feat :: ',new_local_feats.shape)
        
        # Global Features
        global_feats1 = new_feats4.max(dim=-1)[0]
        global_feats1 = global_feats1.unsqueeze(1).expand(-1, new_feats4.size(2), -1)#Expand global feature
        global_feats1 = global_feats1.permute(0,2,1)#b,128,n
        print('Global Feat :: ',global_feats1.shape)
        
        # Add globa features in local features
        all_feats=torch.cat([new_local_feats, global_feats1],dim=1) #b,648,n
        #print('Local + Global : ', all_feats.shape)
        
        return all_feats
        
    
if __name__=="__main__":
    
    model_args   = parse_pu1k_args()
    P3Dfeat_extract = FeatureExtractor(model_args).cuda()
    
    # input: (b, 3, n)
    # global_feats: (b, c), local_feats: list (b, c, n)
    
    original_pts = torch.rand(4,3,100).cuda()
    features = P3Dfeat_extract(original_pts)
    #print('Feature shape :: ', features.shape)
    