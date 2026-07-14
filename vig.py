# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

def resize_pos_embed(pos_embed, target_shape):
    if pos_embed.shape == target_shape:
        return pos_embed

    return F.interpolate(
        pos_embed,
        size=target_shape[-2:],
        mode='bicubic',
        align_corners=False
    )

def load_pretrained_pos_embed(model, state_dict):
    model_state = model.state_dict()

    if 'pos_embed' in state_dict:
        ckpt_pos = state_dict['pos_embed']
        model_pos_shape = model.pos_embed.shape

        if ckpt_pos.shape != model_pos_shape:
            print(f"Resizing pos_embed from {ckpt_pos.shape} to {model_pos_shape}")
            state_dict['pos_embed'] = resize_pos_embed(ckpt_pos, model_pos_shape)

    # Keep only keys that exist and match shape
    filtered_state = {}
    dropped = []

    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            dropped.append(k)

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    print("Loaded keys:", len(filtered_state))
    print("Dropped keys:", len(dropped))
    print("Example dropped keys:", dropped[:20])
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

#curr output is B, C, 14, 14
class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu', grid_size=14):
        super().__init__()
        assert img_size % grid_size == 0, f"img_size {img_size} must be divisible by grid_size {grid_size}"
        total_stride = img_size // grid_size
        assert total_stride in (2, 4, 8, 16, 32), "Supported grid sizes for 224 input: 112, 56, 28, 14, 7"

        n_down = int(math.log2(total_stride))

        c1, c2, c3, c4 = out_dim // 8, out_dim // 4, out_dim // 2, out_dim

        layers = []

        # stage 1
        layers += [
            nn.Conv2d(in_dim, c1, 3, stride=2 if n_down >= 1 else 1, padding=1),
            nn.BatchNorm2d(c1),
            act_layer(act),
        ]

        # stage 2
        layers += [
            nn.Conv2d(c1, c2, 3, stride=2 if n_down >= 2 else 1, padding=1),
            nn.BatchNorm2d(c2),
            act_layer(act),
        ]

        # stage 3
        layers += [
            nn.Conv2d(c2, c3, 3, stride=2 if n_down >= 3 else 1, padding=1),
            nn.BatchNorm2d(c3),
            act_layer(act),
        ]

        # stage 4
        layers += [
            nn.Conv2d(c3, c4, 3, stride=2 if n_down >= 4 else 1, padding=1),
            nn.BatchNorm2d(c4),
            act_layer(act),
        ]

        # stage 5 (optional extra downsample for 7x7)
        layers += [
            nn.Conv2d(c4, c4, 3, stride=2 if n_down >= 5 else 1, padding=1),
            nn.BatchNorm2d(c4),
        ]

        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        x = self.convs(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        self.classifier_mode = opt.classifier_mode
        
        #self.stem = Stem(out_dim=channels, act=act)
        self.stem = Stem(img_size=opt.img_size, in_dim=opt.in_chans, out_dim=channels, act=act, grid_size=opt.grid_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        #the paper has the k increaing across blocks, in my work i set it to k = 9
        #num_knn = [9] * self.n_blocks
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        #this max dialtion matchins the out put of 14x14, should be changed for different grid size
        #max_dilation = 196 // max(num_knn)
        node_count = opt.grid_size * opt.grid_size
        max_dilation = max(1, node_count // max(num_knn))

        #self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, opt.grid_size, opt.grid_size))

        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(
            nn.Conv2d(channels, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(opt.dropout),
            nn.Conv2d(1024, opt.n_classes, 1, bias=True)
        )

        #New reconstruction head for self-supervised training
        self.reconstruction = Seq(
            nn.Conv2d(channels, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Conv2d(1024, opt.in_chans, 1, bias=True)
        )

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        #assumes 14x14 (1, C, 14, 14) and adds pos embed
        x = self.stem(inputs) + self.pos_embed
        #other options is to run the stem, interpolate pos embed, then added
        #that would support non 14x14 feature maps
        B, C, H, W = x.shape
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        if self.classifier_mode == "patch":
            #patch logits (B, n classes, h, w)
            return self.prediction(x)

        if self.classifier_mode == "self_supervised":
            #reconstruct the img from feature map
            #(b, in chans, h, w)
            reconstruct = self.reconstruction(x)

            #upsample to og input size
            reconstruct = F.interpolate(reconstruct, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
            return reconstruct

        #This code ends with pooling and prediction head to return class logits
        #output shape (B, n_class)
        #would need to edit this for patch level vs super pixel vs overall classification vs graph extraction
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

#updated model build:
#model = create_model(
#    "vig_ti_224_gelu",
#    num_classes=2,
#    img_size=224,
#    grid_size=56,   # 112 / 56 / 28 / 14 / 7
#)


@register_model
def vig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 12 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.img_size = kwargs.get("img_size", 224)
            self.grid_size = kwargs.get("grid_size", 14)  # choose 7,14,28,56,112
            self.in_chans = kwargs.get("in_chans", 3)
            self.classifier_mode = kwargs.get("classifier_mode", "image")  # "image" or "patch" or "self_supervised"

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 320 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.img_size = kwargs.get("img_size", 224)
            self.grid_size = kwargs.get("grid_size", 14)  # choose 7,14,28,56,112
            self.in_chans = kwargs.get("in_chans", 3)
            self.classifier_mode = kwargs.get("classifier_mode", "image")  # "image" or "patch" or "self_supervised"

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 640 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.img_size = kwargs.get("img_size", 224)
            self.grid_size = kwargs.get("grid_size", 14)  # choose 7,14,28,56,112
            self.in_chans = kwargs.get("in_chans", 3)
            self.classifier_mode = kwargs.get("classifier_mode", "image")  # "image" or "patch" or "self_supervised"

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model
