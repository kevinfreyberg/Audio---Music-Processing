from torch.utils.data import Dataset
import pandas as pd

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)

    def __len__(self):
        pass

    def __getitem__(self, index):
        