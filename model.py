from torch import Tensor
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class Boom(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, dropout=0, shortcut=False):
        super().__init__()
        self.ff1 = nn.Linear(input_dim, hidden_dim)
        self.ff2 = nn.Linear(hidden_dim, input_dim) if not shortcut else None
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        assert(hidden_dim % input_dim == 0)

    def forward(self, x):
        y = self.dropout(self.activation(self.ff1(x)))
        if self.ff2 is not None:
            return self.ff2(y)
        # fix the dimensions (chunk and sum) if we're taking a shortcut
        input_dim = x.shape[-1]
        z = torch.split(y, input_dim, dim=-1)
        return torch.stack(z).sum(dim=0)

class BlockNorm(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.start = nn.LayerNorm(input_dim)
        self.middle = nn.LayerNorm(input_dim)
        self.mem = nn.LayerNorm(input_dim)
        self.output = nn.LayerNorm(input_dim)
        self.ff = nn.LayerNorm(input_dim)
        self.xff = nn.LayerNorm(input_dim)

class Block(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, heads=1, max_len=5000, dropout=0, rnn=False, attn=False, residual=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(input_dim, heads, dropout=dropout) if attn else None
        self.ln = BlockNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_dim, input_dim) if rnn else None
        self.ff = Boom(input_dim, dropout=dropout, hidden_dim=hidden_dim, shortcut=True)
        self.residual = residual
        self.max_len = max_len

    def forward(self, h: Tensor, attn_mask: Tensor,
            mem: Optional[Tensor]=None, hidden: Optional[Tuple[Tensor, Tensor]]=None
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        new_hidden = (torch.ones(0), torch.ones(0))
        new_mem = torch.ones(0)
        h = self.ln.start(h)
        if self.rnn is not None:
            x, new_hidden = self.rnn(h, hidden)
            x = self.dropout(x)
            # preserve input shape (chunk and sum)
            input_dim = h.shape[-1]
            z = torch.split(x, input_dim, dim=-1)
            x = torch.stack(z).sum(dim=0)
            h = h + x if self.residual else x

        if self.attn is not None:
            h = self.ln.middle(h)
            memh = self.ln.mem(h)
            long_mem = torch.cat([mem, memh], dim=0) if mem is not None else memh
            new_mem = long_mem[-self.max_len:]
            x, _ = self.attn(h, long_mem, long_mem, attn_mask=attn_mask)
            # x, _ = checkpoint(lambda *args: self.attn(*args, attn_mask=attn_mask), h, long_mem, long_mem)
            h = h + self.dropout(x)

        # feed forward
        h = self.ln.ff(h)
        x = self.ln.xff(h)
        x = self.ff(x)
        # x = checkpoint(self.ff, x)
        h = h + self.dropout(x)
        return h, new_mem, new_hidden

class SHARNN(nn.Module):
    def __init__(self, n_token, embed_dim, hidden_dim=2048, n_layers=4, heads=1, max_len=5000, dropout=0, tied=False):
        super().__init__()
        self.encoder = nn.Embedding(n_token, embed_dim)
        self.decoder = nn.Linear(embed_dim, n_token)
        if tied:
            self.decoder.weight = self.encoder.weight
        self.blocks = nn.ModuleList([
            Block(embed_dim, hidden_dim, heads=heads,
                  max_len=max_len, dropout=dropout,
                  rnn=True, attn=(i == n_layers - 2), residual=False)
            for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x,
            mem: Optional[List[Tensor]]=None,
            hidden: Optional[List[Tuple[Tensor, Tensor]]]=None
    ) -> Tuple[Tensor, List[Tensor], List[Tuple[Tensor, Tensor]]]:
        h = self.dropout(self.encoder(x))
        if mem is not None:
            hist = self.max_len - len(h)
            mem = [m[-hist:] for m in mem]
        attn_mask = h.new_full((len(h), len(h)), float('-inf'))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        if mem is not None:
            mem_mask = h.new_zeros((len(x), max([len(m) for m in mem])))
            attn_mask = torch.cat([mem_mask, attn_mask], dim=-1)
        new_mem: List[Tensor] = []
        new_hidden: List[Tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            h, out_mem, out_hidden = block(
                h, attn_mask=attn_mask,
                hidden=hidden[i] if hidden is not None else None,
                mem=mem[i] if mem is not None else None)
            new_mem.append(out_mem)
            new_hidden.append(out_hidden)
        h = self.dropout(h)
        d = self.decoder(h)
        return d, new_mem, new_hidden

    def sample(self, x: Tensor, length: int) -> Tensor:
        out = []
        mem = hidden = None
        for i in range(length):
            y, mem, hidden = self.forward(x, mem, hidden)
            # x = torch.argmax(y[-1], dim=1).unsqueeze(0)
            y = F.softmax(y[-1] / 0.99, dim=1)
            x = torch.multinomial(y, 1)
            out.append(x)
        return torch.stack(out, dim=0)
