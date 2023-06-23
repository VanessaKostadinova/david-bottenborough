import librosa.feature
from datasets import load_dataset

dataset_path = '/Users/vanessa/WorkProjects/datasets/lj_speech/'


# This reads an audio file and maps it to a mel spectrogram, which it then stores as mel
def map_to_array(batch):
    y, sr = librosa.load(batch['audio']['path'])
    audio, _ = librosa.effects.trim(y)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    batch["mel"] = mel
    return batch


dataset = load_dataset("lj_speech", split="train")

# this should cache the new mapped dataset
dataset = dataset.map(map_to_array)
dataset.save_to_disk(dataset_path + 'lj_speech_mel.ds')
