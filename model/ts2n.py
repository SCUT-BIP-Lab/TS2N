import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F
import torchvision


class TopKChannelPool2d(nn.Module):
    def __init__(self, k):
        super(TopKChannelPool2d, self).__init__()
        self.k = k

    def _top_k_pool(self, scores, k):
        shape = scores.shape  # (N, C, H, W)
        scores = scores.reshape((shape[0], shape[1], -1))
        out = torch.sort(scores, dim=2)[0][:, :, -k:].mean(axis=2)
        out = out.unsqueeze(-1).unsqueeze(-1)
        return out

    def forward(self, input: Tensor) -> Tensor:
        return self._top_k_pool(input, self.k)


class TopKSpatialPool2d(nn.Module):
    def __init__(self, k):
        super(TopKSpatialPool2d, self).__init__()
        self.k = k

    def _top_k_pool(self, scores, k):
        shape = scores.shape  # (N, C, H, W)
        scores = scores.reshape((shape[0], shape[1], -1))
        out = torch.sort(scores, dim=1, descending=True)[0][:, :k, :].mean(axis=1)
        out = out.reshape((shape[0], -1, shape[2], shape[3]))
        return out

    def forward(self, input: Tensor) -> Tensor:
        return self._top_k_pool(input, self.k)
    

class newcSE(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.avgpool = TopKChannelPool2d(225)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)

        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.Conv_Excitation(self.Conv_Squeeze(self.avgpool(U)))  # shape: [bs, c, 1, 1]
        z = self.norm(z)
        return U * z.expand_as(U)
    
        
class CoTsSE(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=8
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
#
        self.ch = newcSE(dim)

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.Conv1x1 = nn.Conv2d(dim, 1, kernel_size=1, bias=False)

        self.topk1 = TopKSpatialPool2d(169)
        self.topk2 = TopKSpatialPool2d(9)
        self.norm = nn.Sigmoid()
        

    def forward(self, x):
        bs,c,h,w=x.shape

        avg_out = self.topk1(x)
        max_out = self.topk2(x)
        q = torch.cat([avg_out, max_out], dim=1)

        q = self.conv1(q)
        k1 = x * self.norm(q) + self.ch(x)


        v=self.value_embed(x).view(bs,c,-1) #   bs,c,h,w
        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)

        return k1+k2


class ResAttentionBlock(nn.Module):
    def __init__(
            self,
            channels: int
    ) -> None:
        super(ResAttentionBlock, self).__init__()

        self.bn = nn.BatchNorm2d(channels)
        self.att = CoTsSE(channels)#top1

        self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.bn.weight, 0)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        identity = x
        out = self.att(x)
        out = self.bn(out)
        out += identity
        out = self.relu(out)

        return out

class Model_Resnet_Att(torch.nn.Module):
    """
    # 模型样板
    """

    def __init__(self, frames_per_video, feature_dim, out_dim):
        super(Model_Resnet_Att, self).__init__()

        # cv模型
        self.feature_dim = feature_dim
        self.frames_per_video = frames_per_video
        self.cv_model = torchvision.models.resnet18(pretrained=True)
        self.cv_model.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)

        self.att = ResAttentionBlock(256)

        # feature extraction
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, data):

        fis = {}  # 字典存结果
        B, N, C, W, H = data.shape
        
        x = data.view((-1, C, W, H))

        x = self.cv_model.conv1(x)
        x = self.cv_model.bn1(x)
        x = self.cv_model.relu(x)
        x = self.cv_model.maxpool(x)

        for i in range(2):
            layer_name = "layer" + str(i + 1)
            layer = getattr(self.cv_model, layer_name)
            x = layer(x)

        x = self.cv_model.layer3[0](x)
        x = self.att(x)
        x = self.cv_model.layer3[1](x)

        x = self.cv_model.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.cv_model.fc(x)
        
        x = x.view(-1, int(self.frames_per_video), int(self.feature_dim))

        x = torch.mean(x, dim=1, keepdim=False)


        physical_feature = torch.div(x, torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # 归一化
        return physical_feature


class Model_Gaitset_PAN(torch.nn.Module):
    """
    # 模型样板
    """
    def __init__(self, frames_per_video, feature_dim, out_dim, block_length=4):
        super(Model_Gaitset_PAN, self).__init__()

        # cv模型
        self.cv_model_backbone = torchvision.models.resnet18(pretrained=True)
        self.cv_model_backbone.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
        self.cv_model_mgp = torchvision.models.resnet18(pretrained=True)
        self.cv_model_mgp.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
    
        # feature extraction
        self.feature_dim = feature_dim
        self.frame_length = frames_per_video
        self.block_length = block_length
        self.frames_block = int(frames_per_video/block_length)

        self.PA = PA(self.block_length)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, data):

        x = self.PA(data)
        # data = data.view((-1,) + data.shape[-3:])
        x = self.cv_model_backbone.conv1(x)
        x = self.cv_model_backbone.bn1(x)
        x = self.cv_model_backbone.relu(x)
        x = self.cv_model_backbone.maxpool(x)
        gl = torch.max(x.view((-1, self.frames_block) + x.size()[-3:]), 1)[0]
        gl = self.cv_model_mgp.layer1(gl)

        x = self.cv_model_backbone.layer1(x)
        gl = gl + torch.max(x.view((-1, self.frames_block) + x.size()[-3:]), 1)[0]
        gl = self.cv_model_mgp.layer2(gl)

        x = self.cv_model_backbone.layer2(x)
        gl = gl + torch.max(x.view((-1, self.frames_block) + x.size()[-3:]), 1)[0]
        gl = self.cv_model_mgp.layer3(gl)

        x = self.cv_model_backbone.layer3(x)
        gl = gl + torch.max(x.view((-1, self.frames_block) + x.size()[-3:]), 1)[0]
        gl = self.cv_model_mgp.layer4(gl)

        x = self.cv_model_backbone.layer4(x)
        gl = gl + torch.max(x.view((-1, self.frames_block) + x.size()[-3:]), 1)[0]

        x = self.avgpool(x) # + self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.cv_model_backbone.fc(x)
        x = x.view(-1, int(self.frames_block), int(self.feature_dim))
        x = torch.mean(x, dim=1, keepdim=False)
        gl = self.avgpool(gl) # + self.maxpool(gl)
        gl = torch.flatten(gl, 1)
        gl = self.cv_model_mgp.fc(gl)

        # 帧间融合()
        x = torch.div(x, torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12))
        gl = torch.div(gl, torch.norm(gl, p=2, dim=1, keepdim=True).clamp(min=1e-12))
        behavior_feature = torch.cat((x, gl), dim=1)
        return behavior_feature
        


class PA(nn.Module):
    def __init__(self, n_length):
        super(PA, self).__init__()
        self.shallow_conv = nn.Conv2d(3,8,7,1,3)
        self.n_length = n_length

    def forward(self, x):
        x = x.view((-1, 3) + x.size()[-2:])
        x = self.shallow_conv(x)
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2), x.size(-1))
        PA = x[:,:self.n_length-1,:,:,:] - x[:,1:self.n_length,:,:,:]
        PA = torch.sqrt(torch.sum(PA ** 2, 2)+1e-12)
        return PA




