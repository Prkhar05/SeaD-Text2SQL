import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

### Input Embeddings ###
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

### Positional Encodings ###
class PositionalEncodings(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

### Layer Normalization ###
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

### Feed Forward Block ###
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

### Multi-Headed Attention Block ###
class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        x = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linear_out(x), attn

### Residual Connection ###
class ResidualConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

### Encoder Layer ###
class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([ResidualConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

### Encoder ###
class Encoder(nn.Module):
    def __init__(self, layer: nn.Module, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNormalization(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

### Decoder Layer ###
class DecoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: nn.Module, src_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([ResidualConnection(size, dropout) for _ in range(3)])
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

### Decoder ###
class Decoder(nn.Module):
    def __init__(self, layer: nn.Module, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNormalization(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

### Generator ###
class Generator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

### Pointer-Generator Network ###
class PointerGeneratorNetwork(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(PointerGeneratorNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.p_gen_linear = nn.Linear(d_model * 3, 1)  # h_t*, s_t, x_t

    def forward(self, decoder_hidden, context_vector, decoder_input, attention_dist):
        p_gen = torch.sigmoid(self.p_gen_linear(torch.cat((decoder_hidden, context_vector, decoder_input), -1)))
        vocab_dist = F.softmax(decoder_hidden, dim=-1)
        final_dist = p_gen * vocab_dist + (1 - p_gen) * attention_dist
        return final_dist

# Complete Transformer Model with Pointer-Generator Network
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, pointer_generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.pointer_generator = pointer_generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(self.src_embed(src), src_mask)
        decoder_output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        context_vector = self.compute_context_vector(decoder_output, memory, src_mask)
        attention_dist = self.compute_attention_distribution(decoder_output, memory, src_mask)
        final_dist = self.pointer_generator(decoder_output, context_vector, tgt, attention_dist)
        return final_dist

    def compute_context_vector(self, decoder_output, memory, src_mask):
        # Compute context vector using attention over the encoder outputs
        # Placeholder for actual implementation
        attn_weights = torch.bmm(decoder_output, memory.transpose(1, 2))
        if src_mask is not None:
            attn_weights = attn_weights.masked_fill(src_mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        context_vector = torch.bmm(attn_weights, memory)
        return context_vector

    def compute_attention_distribution(self, decoder_output, memory, src_mask):
        # Compute attention distribution over the source tokens
        # Placeholder for actual implementation
        attn_weights = torch.bmm(decoder_output, memory.transpose(1, 2))
        if src_mask is not None:
            attn_weights = attn_weights.masked_fill(src_mask == 0, -1e9)
        attention_dist = F.softmax(attn_weights, dim=-1)
        return attention_dist

# Function to build the transformer model
def build_model(src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = FeedForwardBlock(d_model, d_ff, dropout)
    position = PositionalEncodings(d_model, 5000, dropout)
    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
    encoder = Encoder(encoder_layer, N)
    decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    decoder = Decoder(decoder_layer, N)
    src_embed = nn.Sequential(InputEmbeddings(d_model, src_vocab_size), c(position))
    tgt_embed = nn.Sequential(InputEmbeddings(d_model, tgt_vocab_size), c(position))
    generator = Generator(d_model, tgt_vocab_size)
    pointer_generator = PointerGeneratorNetwork(tgt_vocab_size, d_model)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, generator, pointer_generator)
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
