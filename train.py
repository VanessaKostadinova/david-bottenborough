from random import Random

import datasets
import torch
import model
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_path = '/Users/vanessa/WorkProjects/datasets/lj_speech/lj-mel-full'
dataset = datasets.load_from_disk(dataset_path)
random = Random(12345)
batch_size = 64
epochs = 10

# set up simple encoder
def get_char_enc():
    # chars = sorted(list(set(data)))
    chars = sorted([chr(i) for i in range(128)])
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode, vocab_size

enc, dec, vocab_size = get_char_enc()

def get_batch():
    indexes = [random.randint(0, len(dataset) - 1) for _ in range(batch_size)]
    data = [dataset[i] for i in indexes]

    x = [torch.tensor(enc(d['normalized_text'])).to(device) for d in data] # list of unpadded encoded text tensors
    xt_max = max([t.shape[0] for t in x])
    x = torch.stack([F.pad(t, (0, xt_max - t.shape[0]), value=128) for t in x])

    y = [torch.tensor(d['mel']).to(device) for d in data] # list of unpadded mel spectrogram tensors (C,T)
    y = [torch.cat([t, torch.ones((t.shape[0], 1))], dim=1) for t in y]
    yt_max = max([t.shape[1] for t in y])
    y = torch.stack([F.pad(t, (0, yt_max - t.shape[1]), value=float("-inf")) for t in y])
    # in pytorch how do I pad a tensor to a specific T with the value float(-inf) before I stack it?
    return x, y
# --------------

model = model.ViT().to(device)
optimiser = torch.optim.AdamW(model.parameters())

for e in range(epochs):
    x, y = get_batch()
    for t in range(y.shape[2]):
        x_in = x[:, :]
        y_in = y[:, :, :t+1]
        target = y[:, :, t]
        mel, loss = model(x_in, y_in, target)
        print(loss)
        optimiser.zero_grad()
        loss.backward()
