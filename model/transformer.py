import math
import torch
import numpy as np
import torch.nn as nn
import config
from config import *
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# data_dir = r"./totaldata/"

# output_index={"DoS":Dos,"Fuzzy":Fuzzy ,"gear":Gear ,"RPM":Rpm,"normal_run": Normal_run}
output_size = 2  # one_hot编码
input_size = 16  # 0-15+-999+999
d_model = embedding_size = position_size = config.d_model  # 字嵌入,位置嵌入维度
n_layers = config.n_layers  # number of Encoder of Decoder Layer
n_heads = config.n_heads  # number of heads in Multi-Head Attention
dropout = config.dropout
seq_len = config.seq_len
d_ff = 128  # FeedForward dimension
d_k = d_v = 64  # dimension of K ,Q, V


class Embedding(nn.Module):
    def __init__(self, d_model):
        super(Embedding, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        # self.lrelu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(16, d_model)
        # self.layer_norm = nn.LayerNorm(32)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.to(torch.float32)
        x = self.fc1(x)
        # x = self.lrelu(x)
        # x = self.drop(x)
        x = self.fc(x)
        # x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=dropout, max_len=seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # max_len*512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.pe=pe.cuda()
        self.register_buffer("pe", pe)
        # 最后把pe位置编码矩阵注册成模型的buffer，buffer也就是：对模型效果有帮助，但是不是模型结构的超参数或者参数，不需要随着优化步骤更新
        # 注册之后可以在模型保存后重加载时和模型结构与参数一同被加载
        # 经过嵌入层和位置编码器之后，最终输出为一个加入了位置编码信息的词嵌入张量。

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # print(x.size,self.pe.size)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(16).unsqueeze(1)
    # print(pad_attn_mask, pad_attn_mask.expand(batch_size, len_q, len_k))
    # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k
        )  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(
            attn_mask, -1e9
        )  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        Q = (
            self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        )  # Q: [batch_size, n_heads, len_q, d_k]
        K = (
            self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        )  # K: [batch_size, n_heads, len_k, d_k]
        V = (
            self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        )  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, n_heads, 1, 1
        )  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(
            batch_size, -1, n_heads * d_v
        )  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]#full connected layer
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


# 前馈全连接网路
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(p=0.1),
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(
            output + residual
        )  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(
            enc_outputs
        )  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(input_size, d_model)
        # self.src_emb = Embedding(d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        # print(self.src_emb)
        # print(enc_inputs)
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]

        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(
            0, 1
        )  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(
            enc_inputs, enc_inputs
        )  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """

        enc_outputs = self.encoder(enc_inputs)

        return enc_outputs
