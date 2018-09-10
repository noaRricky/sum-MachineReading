import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from new import constants


class GloveDataSet(Dataset):
    """dataset for containing data of corpus

    Arguments:
        corpus -- list of word index
    """

    def __init__(self, corpus):
        self.data = corpus

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def glove_collate(batch):
    max_len = 0

    # get the max length of batch
    for elem in batch:
        batch_len = len(batch)
        max_len = max_len if max_len > batch_len else batch_len

    # add pad token to each batch
    for i, elem in enumerate(batch):
        batch = np.pad(batch, (0, max_len - len(batch)),
                       mode='constant', constant_value=constants.PAD_TOKEN)
    return default_collate(batch)