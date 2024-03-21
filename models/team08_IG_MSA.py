import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def act(act_type, inplace=True, neg_slope=0.2, n_selu=1):
    # helper selecting activation
    # neg_slope: for selu and init of selu
    # n_selu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(0.2,inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'selu':
        layer = nn.SELU()
    elif act_type == 'elu':
        layer = nn.ELU()
    elif act_type == 'silu':
        layer = nn.SiLU()
    elif act_type == 'rrelu':
        layer = nn.RReLU()
    elif act_type == 'celu':
        layer = nn.CELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='elu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

class EMHA(nn.Module):
    def __init__(self, inChannels, splitfactors=4, heads=8):
        super().__init__()
        dimHead = inChannels // (2*heads)

        self.heads = heads
        self.splitfactors = splitfactors
        self.scale = dimHead ** -0.5

        self.reduction = nn.Conv1d(
            in_channels=inChannels, out_channels=inChannels//2, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.toQKV = nn.Linear(
            inChannels // 2, inChannels // 2 * 3, bias=False)
        self.expansion = nn.Conv1d(
            in_channels=inChannels//2, out_channels=inChannels, kernel_size=1)

    def forward(self, x):
        x = self.reduction(x)
        x = x.transpose(-1, -2)

        qkv = self.toQKV(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        qs, ks, vs = map(lambda t: t.chunk(
            self.splitfactors, dim=2), [q, k, v])

        pool = []
        for qi, ki, vi in zip(qs, ks, vs):
            tmp = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attn = self.attend(tmp)
            out = torch.matmul(attn, vi)
            out = rearrange(out, 'b h n d -> b n (h d)')
            pool.append(out)

        out = torch.cat(tuple(pool), dim=1)
        out = out.transpose(-1, -2)
        out = self.expansion(out)
        return 

class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        #print(illu_fea.shape)
        illu_map = self.conv2(illu_fea)
        return illu_fea


class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=16,
            heads=4,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        #print('dim = ', dim, 'ans =',dim_head*heads)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        #print('x_shape = ', x.shape)
        
        q_inp = self.to_q(x)
        
        k_inp = self.to_k(x)
      
        v_inp = self.to_v(x)
        #print('v_inp = ', v_inp.shape)
        
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        #print('q',q.shape)
        #print('k',k.shape)
        #print('v',v.shape)
        #print('illu_attan',illu_attn.shape)
        v = v * illu_attn
        #print('final_v',v.shape)
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        #print(x.shape)
        x = x.permute(0, 3, 1, 2)    # Transpose
        #print(x.shape)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        #print(x.shape)
        out_c = self.proj(x).view(b, h, w, c)
        #print('out_c = ', out_c.shape)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        #out_p=self.pos_emb(out_c)
        #print('out_p = ',out_p.shape)
        out = out_c + out_p
        out= out.permute(0,3,2,1)
        #print('out',out.shape)

        return out

class CAM(nn.Module):
	def __init__(self, in_channels, reduction_ratio=16):
		super().__init__()
		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
		self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

	def forward(self, x):
		avg_pool = self.global_avg_pool(x)
		avg_pool = avg_pool.view(avg_pool.size(0), -1)
		fc1_output = F.relu(self.fc1(avg_pool))
		channel_attention = torch.sigmoid(self.fc2(fc1_output))
		x = x * channel_attention.unsqueeze(2).unsqueeze(3)
		return x

class BB (nn.Module) :
  def __init__(self, nf, splitfactors=4, heads=8, k=3) :
    super(BB, self).__init__()
    #self.k = k
    self.uk3_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.uk3_2 = conv_block(in_nc=2*nf, out_nc=nf, kernel_size= 3)
    self.uk3_3 = conv_block(in_nc=3*nf, out_nc=nf, kernel_size= 3)
    self.lk3_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.lk3_2 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.lk3_3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.k1 = conv_block(in_nc=4*nf, out_nc=nf, kernel_size= 1)
    self.igmsa = IG_MSA(64)
    
    #self.emha = EMHA(nf*k*k, splitfactors, heads)
    #self.norm = nn.LayerNorm(nf*k*k)
    #self.unFold = nn.Unfold(kernel_size=(k, k), padding=1)
  
  def forward(self,x,xt1):
    _, _, h, w = x.shape

    #upper path
    xu1_1= self.uk3_1(x)
    xu1_2= torch.cat((xu1_1,x),1)
    xu2_1= self.uk3_2(xu1_2)
    xu2_2= torch.cat((xu2_1,xu1_1,x),1)
    xu3_1= self.uk3_3(xu2_2)
    xu3_2= torch.cat((xu3_1,xu2_1,xu1_1,x),1)
    xu3= self.k1(xu3_2)
    # print('xu3_size = ', xu3.shape)
    #lower path
    # xl1= self.lk3_1(x)
    # xl1= self.lk3_2(xl1)
    # xl1= self.lk3_3(xl1)
    # xl1= x+xl1
    #transformer
    xt=self.igmsa(xu3.permute(0,3,2,1), illu_fea_trans=xt1.permute(0, 2, 3, 1)) 
    # print('xt_size = ', xt.shape)

    # xt1 = self.unFold(x)
    # xt2 = xt1.transpose(-2, -1)
    # xt2 = self.norm(xt2)
    # xt2 = xt2.transpose(-2, -1)
    # xt2 = self.emha(xt2)+xt1
    # xt2 = F.fold(xt2, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))
    # xt2 = xt2+x

    return xt

class CB (nn.Module) :
  def __init__(self, nf, no_bb) :
    super(CB, self).__init__()
    self.fwd_bb = nn.ModuleList([BB(nf) for i in range(no_bb)])
    self.fwd_cam = nn.ModuleList([CAM(nf) for i in range (no_bb)])
    self.no_bb = no_bb

  def forward(self,x,xt1) :
    x1 = self.fwd_bb[0](x, xt1)
    x1 = self.fwd_cam[0](x1)
    for i in range(self.no_bb-1):
      x1 = self.fwd_bb[i+1](x1, xt1)
      x1 = self.fwd_cam[i+1](x1)    
    return x1 + x

class OurUpSample(nn.Module):
    def __init__(self,in_nc, kernel_size=3, stride=1, bias=True, pad_type='zero', \
            act_type=None, mode='CNA',upscale_factor=2):
        super(OurUpSample, self).__init__()
        self.U1 = pixelshuffle_block(in_nc, in_nc, upscale_factor=upscale_factor, kernel_size=3, norm_type = None)
        self.co1 = conv_block(in_nc, in_nc, kernel_size=1, norm_type=None, act_type='elu', mode='CNA')
        # self.co2 = conv_block(16, 3, kernel_size=3, norm_type=None, act_type='prelu', mode='CNA')

    def forward(self, x):
        out1 = self.U1(x)
        return self.co1(out1)

def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)

class MyNetwork (nn.Module) :
  def __init__(self, nf, no_cb, no_bb, in_c = 3) :
    super(MyNetwork, self).__init__()
    self.illu_fea = Illumination_Estimator(64)
    self.k5 = conv_block(in_nc=in_c, out_nc=nf, kernel_size= 5)
    self.cb = CB(nf, no_bb)
    self.fwd_cb = nn.ModuleList([CB(nf, no_bb) for i in range (no_cb - 1)])
    self.k6= conv_block(in_nc=nf, out_nc=3, kernel_size= 3)
    self.u = OurUpSample(3, kernel_size=3, act_type='elu',upscale_factor=2)
    self.u1 = OurUpSample(3, kernel_size=3, act_type='elu',upscale_factor=2)
    self.ub = nn.Upsample(scale_factor=4, mode='bicubic')
    self.k7 = conv_block(in_nc=3, out_nc=64, kernel_size=3, norm_type=None, act_type='elu')
    self.uk3_1 = conv_block(in_nc=2*nf, out_nc=nf, kernel_size=3)
    self.k3 = conv_block(in_nc=3, out_nc=nf, kernel_size= 3)
    self.k3_1 = conv_block(in_nc=nf, out_nc=in_c, kernel_size= 3)
      
    #self.k8 = conv_block(in_nc=, out_nc=, kernel_size=3, norm_type=None, act_type='elu')
    #self.ub = nn.Upsample(scale_factor=4, mode='nearest')
    # self.ub = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
    #                         nn.Upsample(scale_factor=4, mode='nearest'))
   # self.u = OurUpSample(nf, kernel_size=3, act_type='elu',upscale_factor=4)
    # self.up1 = pixelshuffle_block(nf, nf, upscale_factor=2,norm_type = 'batch')
    # self.up2 = pixelshuffle_block(nf, nf, upscale_factor=2,norm_type = 'batch')
    #self.conv2 = conv_block(nf,nf,kernel_size=kernel_size,norm_type=norm_type,act_type=act_type)
    #self.conv3 = conv_block(nf,out_nc,kernel_size=kernel_size,norm_type=norm_type,act_type=act_type)
    #self.k3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    #self.k3_1 = conv_block(in_nc=nf, out_nc=in_c, kernel_size= 3)

  def forward (self, x) :
    #upper 
    xt1=self.illu_fea(x)
    #print('xt1_size = ', xt1.shape)
    xu1 = self.k5(x)
    xu2 = self.cb(xu1, xt1)
    for l in self.fwd_cb :
      xu3 = l(xu2,xt1)
    xu2 = torch.cat((xu2,xu3),1)
    xu2 = self.uk3_1(xu2)
    xu2= xu2+xu1

    xu3= self.k6(xu2)
    # xu4= self.up(xu3)
    # xu5= self.k7(xu4)
    #xu6= self.k8(xu5)
    # print('xu3_size = ', xu3.shape)

    xu4=self.u(xu3)
    #print('xu4_size = ', xu4.shape)
    xu41=self.u1(xu4)
    xu5=self.k3(xu41)
    xu6=self.k3_1(xu5)
    xu7=self.ub(x)

    #xu2 = self.u(xu2)
    #xu2 = self.up2(xu2)
    #xu2 = self.k3(xu2)
    #xu2= self.k3_1(xu2)
    #lower
    #xl1= self.ub(x)

    return xu6+xu7

