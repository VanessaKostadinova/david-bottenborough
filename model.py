import eng_to_ipa as phoneme
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_char_enc():
    chars = sorted([*'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'])
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode, vocab_size


enc, dec, _ = get_char_enc()

n_enc = 8
n_dec = 6

enc_embed = 512
mel_embed = 128

n_embed = 512
dec_hidden = 512
enc_hidden = 512

enc_kernel = 6

dropout = 0.3
n_heads = 4
n_pre_layers = 3
assert n_embed % n_heads == 0
head_size = int(n_embed / n_heads)
vocab_size = 129
ctx_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionHead(nn.Module):
    def __init__(self, n_embed, head_size, mask=False):
        super().__init__()
        self.mask = mask
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        if self.mask:
            self.register_buffer("tril", torch.tril(torch.ones(ctx_size, ctx_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem=None):
        z = mem if mem is not None else x
        B, T, C = z.shape
        k = self.key(z)
        q = self.query(x)
        wgt = q @ k.transpose(-2, -1) * C ** -0.5

        if self.mask:
            mask = self.tril[:T, :T] == 0
            wgt = wgt.masked_fill(mask, float("-inf"))  # (B, T, T)

        wgt = F.softmax(wgt, dim=-1)
        wgt = self.dropout(wgt)
        v = self.value(z)
        out = wgt @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_heads, dropout, mask=False):
        super().__init__()
        # heads = list of attention heads, the output of each of these should be
        self.heads = nn.ModuleList([AttentionHead(n_embed, head_size, mask) for _ in range(0, n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem=None):
        # print(x.shape)
        out = torch.cat([h(x, mem) for h in self.heads], dim=-1)
        # out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.caus_attn_norm = nn.LayerNorm(n_embed)
        self.caus_attn_heads = MultiHeadAttention(n_embed, n_heads, dropout, mask=True)
        self.attn_heads = MultiHeadAttention(n_embed, n_heads, dropout, mask=False)
        self.attn_norm = nn.LayerNorm(n_embed)
        self.ffw_norm = nn.LayerNorm(n_embed)
        self.ffw = FeedForward(n_embed)

    def forward(self, params):
        (x, mem) = params
        x = self.caus_attn_norm(x)
        x = x + self.caus_attn_heads(x)  # (B, T, C)
        x = x + self.attn_heads(self.attn_norm(x), mem)
        x = x + self.ffw(self.ffw_norm(x))  # (B, T, C)
        return x, mem


class Transpose(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()
        self.d1 = d1
        self.d2 = d2

    def forward(self, x):
        return x.transpose(self.d1, self.d2)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = nn.LayerNorm(n_embed)
        self.attn_heads = MultiHeadAttention(n_embed, n_heads, dropout, mask=False)
        self.ffw_norm = nn.LayerNorm(n_embed)
        self.ffw = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.attn_heads(self.attn_norm(x))
        x = x + self.ffw(self.ffw_norm(x))
        return x


class EncoderPreNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.phoneme_converter = phoneme.convert()
        self.embed = nn.Embedding(vocab_size, n_embed, padding_idx=128)
        # for item in list for index
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels=n_embed, out_channels=n_embed, kernel_size=5, device=device),
            *[fn for fn in [
                nn.Conv1d(in_channels=n_embed, out_channels=n_embed, kernel_size=5, device=device),  # (B, C, T)
                Transpose(1, 2),
                nn.BatchNorm1d(n_embed),  # (B, C, T)
                Transpose(1, 2),
                nn.ReLU(),
                nn.Dropout(dropout)]
              for _ in range(n_pre_layers - 1)]
        )
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        # x = self.phoneme_converter(x)
        x = self.embed(x)  # (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.CNN(x)  # (B, C, L)
        # x = x.view(x.size(0), -1)
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.proj(x)  # (B, L, C)
        return x


class MelLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Sequential(
            nn.Linear(dec_hidden, dec_hidden),
            nn.Linear(dec_hidden, mel_embed)
        )

    def forward(self, x):
        x = self.Linear(x)
        return x


class DecoderPreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            Transpose(1, 2),
            nn.Linear(mel_embed, dec_hidden),  # (B, T, M)
            nn.ReLU(),
            nn.Linear(dec_hidden, dec_hidden),
            nn.ReLU()
        )
        self.proj = nn.Linear(dec_hidden, n_embed)  # (B, T, C)

    def forward(self, x):
        x = self.linear(x)
        x = self.proj(x)
        return x


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_prenet = EncoderPreNet()
        self.decoder_prenet = DecoderPreNet()

        self.encoder = nn.Sequential(*[Encoder() for _ in range(n_enc)])
        self.decoder = nn.Sequential(*[Decoder(n_embed) for _ in range(n_dec)])

        self.position_embedding_table = nn.Embedding(ctx_size, n_embed)
        self.mel_linear = MelLinear()

    def forward(self, enc_idx, dec_idx, targets=None):
        # put our text through the pre net
        dec_idx = dec_idx[:, :, -ctx_size:]
        enc_idx = enc_idx[:, -ctx_size:]
        enc_out = self.encoder_prenet(enc_idx)  # (B, T, C)
        _, C, T = enc_out.shape

        # enc_pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        # enc_out = enc_out + enc_pos_emb
        enc_out = self.encoder(enc_out)

        dec_idx = dec_idx[:, -ctx_size:]

        _, _, T = dec_idx.shape

        # put spectrogram through pre net
        dec_out = self.decoder_prenet(dec_idx)  # (B, T, C)

        # dec_idx = dec_idx.transpose(1, 2)
        # dec_pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # dec_out = dec_out + dec_pos_emb

        dec_out, _ = self.decoder((dec_out, enc_out))
        last_step = dec_out[:, -1, :]
        mel_spectrogram = self.mel_linear(last_step)

        # mel_spectrograms = torch.stack([dec_idx, mel_spectrogram], dim=2)

        # # idx and targets are both (batch, time) tensors of int
        # tok_emb = self.token_embedding_table(idx)  # (batch, time, channel)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        # x = tok_emb + pos_emb  # (B, T, C)
        # x = self.blocks(x)
        # x = self.l_norm(x)
        # logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss_mel = None
        else:
            # need to reshape tensors because pytorch expects (B*T, C) in cross_entropy
            # B, C = mel_spectrogram.shape
            print(targets.shape)
            loss_mel = F.mse_loss(mel_spectrogram, targets)

        return mel_spectrogram, loss_mel

    def generate(self, idx, stop_token):
        stop_linear = torch.ones(128)

        while torch.eq(stop_linear, stop_token):
            idx_ctx = idx[:, -ctx_size:]
            # prediction
            spec, stop, _, _ = self(idx_ctx)
            # get last timestep logits
            spec = spec[:, -1, :]  # (B, C)
            # softmax
            probs = F.softmax(spec, dim=1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
