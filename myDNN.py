from distutils.util import strtobool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualFeedFowardBlock(torch.nn.Module):
    '''Block of two feed-forward layer with a reisdual connection:
      
            f(W1^T x + b1)         f(W2^T h1 + b2 )         h2 + x 
        x ------------------> h1 --------------------> h2 ----------> y
        |                                              ^
        |               Residual connection            | 
        +----------------------------------------------+
        
    '''
    
    def __init__(self, dim_in, width, activation_fn=torch.nn.Tanh):
        super().__init__()
        self.layer1 = torch.nn.Linear(dim_in, width)
        self.layer2 = torch.nn.Linear(width, dim_in)
        self.activation_fn = activation_fn()
    
    def forward(self, x):
        h1 = self.activation_fn(self.layer1(x))
        h2 = self.activation_fn(self.layer2(h1))
        return h2 + x
    

class ResidualFeedForwardNet(torch.nn.Module):
    
    def __init__(self, dim_in, nblocks=1, block_width=10):
        super().__init__()
        self._dim_in = dim_in
        self.blocks = torch.nn.Sequential(*[
            ResidualFeedFowardBlock(dim_in, block_width)
            for i in range(nblocks)
        ])
    
    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        # input and output dimension are the same in our residual network.
        return self._dim_in
    
    def forward(self, X):
        return self.blocks(X)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, options, inp_dim, dropout=0.1):
        super().__init__()

        n_head=int(options["selfatt_n_head"])
        d_k=int(options["selfatt_d_k"])
        d_v=int(options["selfatt_d_v"])
        dropout=float(options["selfatt_dropout"])
        d_model=inp_dim

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.out_dim = d_model


    def forward(self, x, mask=None):
        q = x
        k, v = x, x
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class MultiHeadCrossAttention(nn.Module):
    ''' Multi-Head Cross Attention module '''

    def __init__(self, options, inp_dim, dropout=0.1):
        super().__init__()

        n_head=int(options["crossatt_n_head"])
        d_k=int(options["crossatt_d_k"])
        d_v=int(options["crossatt_d_v"])
        dropout=float(options["crossatt_dropout"])
        query_model = inp_dim[0]
        kv_model = inp_dim[1]
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(query_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(kv_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(kv_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, query_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(query_model, eps=1e-6)
        self.out_dim = query_model


    def forward(self, x, mask=None):
        q = x[0]
        k, v = x[1], x[1]

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
