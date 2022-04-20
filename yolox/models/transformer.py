# 参考链接：https://zhuanlan.zhihu.com/p/411311520
# @Author:Yifx
# @Contact: Xxuyifan1999@163.com
# @Time:2021/9/16 20:02
# @Software: PyCharm

"""
文件说明：基于知乎博客代码进行修改，知乎代码为nlp领域的应用，本代码为cv领域的应用，整合过程参考了Swin transformer
https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

"""

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Positional_Encoding(nn.Module):

    def __init__(self):
        super(Positional_Encoding,self).__init__()
    

    def forward(self, seq_len, embedding_dim):
        positional_encoding = np.zeros((seq_len,embedding_dim))
        for pos in range(positional_encoding.shape[0]):
            for i in range(positional_encoding.shape[1]):
                positional_encoding[pos][i] = math.sin(pos/(10000**(2*i/embedding_dim))) if i % 2 == 0 else math.cos(pos/(10000**(2*i/embedding_dim)))

        return torch.from_numpy(positional_encoding)


class Mutihead_Attention(nn.Module):
    def __init__(self, d_model, dim_k, dim_v, n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)

        self.o = nn.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def forward(self, x, y, requires_mask=False):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
        # 对 x 进行自注意力
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1], self.dim_v // self.n_heads) # n_heads * batch_size * seq_len * dim_v
        # print("Attention V shape : {}".format(V.shape))
        attention_score = torch.matmul(Q,K.permute(0,1,3,2)) * self.norm_fact
        output = torch.matmul(attention_score,V).reshape(y.shape[0],y.shape[1],-1)
        # print("Attention output shape : {}".format(output.shape))

        output = self.o(output)
        return output


class Feed_Forward(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim, input_dim*2)
        self.L2 = nn.Linear(input_dim*2, input_dim)

    def forward(self,x):
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return output

class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(0.1)
       

    def forward(self, x, sub_layer, **kwargs):
        sub_output = sub_layer(x, **kwargs)
        # print("{} output : {}".format(sub_layer,sub_output.size()))
        x = self.dropout(x + sub_output)

        layer_norm = nn.LayerNorm(x.size()[1:])
        if x.device.type == 'cuda':
            layer_norm.cuda()
        out = layer_norm(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.positional_encoding = Positional_Encoding()
        self.muti_atten = Mutihead_Attention(input_dim, input_dim, input_dim, input_dim // 32)
        self.feed_forward = Feed_Forward(input_dim)

        self.add_norm = Add_Norm()
        self.input_dim = input_dim


    def forward(self, x): # batch_size * seq_len 并且 x 的类型不是tensor，是普通list

        positional_encoding = self.positional_encoding(x.shape[1], x.shape[2]) # [1, 64, 128]
        if x.device.type == 'cuda':
            positional_encoding = positional_encoding.cuda()
        x += positional_encoding
        # print("After positional_encoding: {}".format(x.size()))
        output = self.add_norm(x, self.muti_atten ,y = x) # [1, 64, 128]
        output = self.add_norm(output, self.feed_forward)  # [1, 64, 128]

        return output



class Transformer_layer(nn.Module):
    def __init__(self, input_dim):
        super(Transformer_layer, self).__init__()
        self.encoder = Encoder(input_dim)

    def forward(self,x):
        encoder_output = self.encoder(x)
        return encoder_output

class Transformer(nn.Module):
    def __init__(self, input_dim):
        super(Transformer, self).__init__() # 用父类nn.Module的初始化方法，来初始化继承类Transformer的方法
       
        self.output_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.model = nn.Sequential(*[Transformer_layer(input_dim) for _ in range(1)])


    def forward(self,x):
        if x.device.type == 'cuda':
            self.model.cuda()
        output = self.model(x)
        # output = self.linear(output)
        # output = self.softmax(output)

        return output