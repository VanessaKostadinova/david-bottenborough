from random import Random

import datasets
import torch
import model
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_path = 'F:/Code/personal/datasets/text-to-speech/lj_speech_mel.ds'
dataset_text_path = 'F:/Code/personal/datasets/text-to-speech/lj_speech_text.txt'
dataset = datasets.load_from_disk(dataset_path)
random = Random(12345)
batch_size = 64
epochs = 300

# set up simple encoder
def get_char_enc(chars):
    chars = sorted({*chars})
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode, vocab_size


with open(dataset_text_path, 'r') as text_file:
    enc, dec, vocab_size = get_char_enc(text_file.read())

# batch = all time
def get_batch():
    indexes = [random.randint(0, len(dataset) - 1) for _ in range(batch_size)]
    data = [dataset[i] for i in indexes]

    x = [torch.tensor(enc(d['normalized_text'])).to(device) for d in data] # list of unpadded encoded text tensors (B, T)
    x_pad_len = max([t.shape[0] for t in x])
    x = torch.stack([F.pad(t, (0, x_pad_len - t.shape[0]), value=128) for t in x])

    y = [torch.tensor(d['mel']).to(device) for d in data] # list of unpadded mel spectrogram tensors (B, C, T)

    yt = y
    yt = [torch.cat([t, torch.ones((t.shape[0], 1), device=device)], dim=1) for t in yt]

    y_pad_len = max([t.shape[1] for t in y])
    y = torch.stack([F.pad(t, (0, y_pad_len - t.shape[1]), value=-1.0) for t in y])
    yt = torch.stack([F.pad(t, (0, y_pad_len - t.shape[1]), value=-1e9) for t in yt])

    yt = yt[:, :, 1:]

    return x, y, yt
# --------------

model = model.ViT().to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=1.5e-15Haha)

for e in range(epochs):
    x, y, yt = get_batch() # x = (B, T) y = (B, C, T)  x will become BTC when it is embedded
    mel, loss = model(x, y, yt)
    print(loss)
    optimiser.zero_grad()
    loss.backward()
