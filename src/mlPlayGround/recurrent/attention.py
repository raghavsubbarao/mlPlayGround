import abc
import torch

class singleSelfAttention(torch.nn.Module):
    """
    Simple self-attention class
    """

    def __init__(self, nin, nout, dropout=0., bias=False):
        super(singleSelfAttention, self).__init__()

        self.n_in = nin
        self.n_out = nout
        self.Wq = torch.nn.Linear(nin, nout, bias=bias)  # torch.nn.Parameter(torch.rand(nin, nout))
        self.Wk = torch.nn.Linear(nin, nout, bias=bias)  # torch.nn.Parameter(torch.rand(nin, nout))
        self.Wv = torch.nn.Linear(nin, nout, bias=bias)  # torch.nn.Parameter(torch.rand(nin, nout))

        if dropout > 0.:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, y):
        # y -> b X n X nin
        que = self.Wq(y)  # y @ self.Wq  # b X n X nout
        key = self.Wk(y)  # y @ self.Wk  # b X n X nout
        val = self.Wv(y)  # y @ self.Wv  # b X n X nout

        att = que @ key.transpose(-2, -1)  # n X n
        att = torch.softmax(att / self.n_out ** 0.5, dim=-1)
        if self.dropout:
            att = self.dropout(att)

        return att @ val


class singleCausalAttention(torch.nn.Module):

    def __init__(self, nin, nout, contextLength, dropout=0., bias=False):
        super(singleCausalAttention, self).__init__()

        self.n_in = nin
        self.n_out = nout
        self.Wq = torch.nn.Linear(nin, nout, bias=bias)  # torch.nn.Parameter(torch.rand(nin, nout))
        self.Wk = torch.nn.Linear(nin, nout, bias=bias)  # torch.nn.Parameter(torch.rand(nin, nout))
        self.Wv = torch.nn.Linear(nin, nout, bias=bias)  # torch.nn.Parameter(torch.rand(nin, nout))

        if dropout > 0.:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.contextLength = contextLength
        self.register_buffer('mask', torch.triu(torch.ones(contextLength, contextLength), diagonal=1))

    def forward(self, y):
        # y -> b X n X nin
        # n should be less than context length
        nTokens = y.shape[1]
        assert (nTokens <= self.contextLength)

        que = self.Wq(y)  # y @ self.Wq  # b X n X nout
        key = self.Wk(y)  # y @ self.Wk  # b X n X nout
        val = self.Wv(y)  # y @ self.Wv  # b X n X nout

        att = que @ key.transpose(-2, -1)  # n X n

        # set masked values to -infinity so softmax sets to 0 for causality
        att.masked_fill_(self.mask.bool()[:nTokens, :nTokens], -torch.inf)

        att = torch.softmax(att / self.n_out ** 0.5, dim=-1)
        if self.dropout:
            att = self.dropout(att)

        return att @ val


class multiHeadAttention(torch.nn.Module):

    def __init__(self, nin, nout, contextLength, nHeads, dropout=0., bias=False):
        super(multiHeadAttention, self).__init__()

        assert (nout % nHeads == 0)

        self.n_in = nin
        self.n_out = nout
        self.nHeads = nHeads
        self.dHead = nout // nHeads

        self.Wqkv = torch.nn.Linear(nin, 3 * nout, bias=bias)
        self.ff = torch.nn.Linear(nout, nout)  # linear layer to combine outputs

        if dropout > 0.:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.contextLength = contextLength
        self.register_buffer("mask", torch.triu(torch.ones(contextLength, contextLength), diagonal=1))

    def forward(self, y):
        b, nTokens, _ = y.shape

        qkv = self.Wqkv(y)  # b X nTokens X 3*nout
        qkv = qkv.view(b, nTokens, 3, self.nHeads, self.dHead)  # b x nTokens x 3 x nHeads x dHead
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x b x nHeads x nTokens x dHead
        que, key, val = qkv  # each is b x nHeads x nTokens x dHead

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        att = que @ key.transpose(-2, -1)  # b x nHeads x nTokens x nTokens

        # Use the mask to fill attention scores
        att.masked_fill_(self.mask.bool()[:nTokens, :nTokens], -torch.inf)

        att = torch.softmax(att / self.dHead ** 0.5, dim=-1)
        if self.dropout:
            att = self.dropout(att)

        context = (att @ val).transpose(1, 2)  # b X nTokens X nHeads X dHead

        # Combine heads as n_out = nHeads * dHead
        context = context.contiguous().view(b, nTokens, self.n_out)
        context = self.ff(context)  # optional projection
        return context


class multiHeadAttentionTorch(torch.nn.Module):

    def __init__(self, nin, nout, contextLength, nHeads, dropout=0., bias=False, needWts=True):
        super(multiHeadAttentionTorch, self).__init__()

        assert (nout % nHeads == 0)

        self.n_in = nin

        self.mha = torch.nn.MultiheadAttention(embed_dim=nout, num_heads=nHeads, dropout=dropout,
                                               bias=bias, add_bias_kv=bias, batch_first=True)
        self.needWeights = needWts
        self.contextLength = contextLength
        self.ff = torch.nn.Linear(nout, nout)  # linear layer to combine outputs
        self.register_buffer("mask", torch.triu(torch.ones(contextLength, contextLength), diagonal=1))

    def forward(self, y):
        b, nTokens, _ = y.shape

        if self.contextLength > nTokens:
            mask = self.mask[:nTokens, :nTokens]
        else:
            mask = self.mask

        context, _ = self.mha(y, y, y, attn_mask=mask, need_weights=self.needWeights)
        context = self.ff(context)  # optional projection
        return context


class multiHeadAttentionTorchSDP(torch.nn.Module):

    def __init__(self, nin, nout, contextLength, nHeads, dropout=0., bias=False):
        super(multiHeadAttentionTorchSDP, self).__init__()

        assert (nout % nHeads == 0)

        self.n_in = nin
        self.n_out = nout
        self.nHeads = nHeads
        self.dHead = nout // nHeads

        self.Wqkv = torch.nn.Linear(nin, 3 * nout, bias=bias)
        self.dropout = dropout
        self.contextLength = contextLength

        self.ff = torch.nn.Linear(nout, nout)  # linear layer to combine outputs

    def forward(self, y):
        b, nTokens, _ = y.shape

        qkv = self.Wqkv(y)  # b X nTokens X 3*nout
        qkv = qkv.view(b, nTokens, 3, self.nHeads, self.dHead)  # b x nTokens x 3 x nHeads x dHead
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x b x nHeads x nTokens x dHead
        que, key, val = qkv  # each is b x nHeads x nTokens x dHead

        dropout = 0. if not self.training else self.dropout

        context = torch.nn.functional.scaled_dot_product_attention(que, key, val,
                                                                   dropout_p=dropout,
                                                                   is_causal=True)  # b x nHeads x nTokens x dHead

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context = context.transpose(1, 2).contiguous().view(b, nTokens, self.n_out)
        context = self.ff(context)  # optional projection
        return context


if __name__ == "__main__":
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running on {device}")

    batch_size = 8
    context_len = 1024
    embed_dim = 768
    n_heads = 12
    dropout = 0.0
    bias = False

    embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

    # mha = multiHeadAttention(embed_dim, embed_dim, context_len,
    #                          n_heads, dropout, bias).to(device)
    # mha = multiHeadAttentionTorch(embed_dim, embed_dim, context_len,
    #                               n_heads, dropout, bias, False).to(device)
    # mha = multiHeadAttentionTorch(embed_dim, embed_dim, context_len,
    #                               n_heads, dropout, bias, True).to(device)
    mha = multiHeadAttentionTorchSDP(embed_dim, embed_dim, context_len,
                                     n_heads, dropout, bias).to(device)

    out = mha(embeddings)
    print(out.shape)