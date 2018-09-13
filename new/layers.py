import torch
import torch.nn as nn
import numpy as np

from .sublayers import MultiHeadAttention, PositionWiseFeedForward


def position_encoding_init(n_position, d_pos_vec):
    """ Init the sinusoid position encoding table

    Args:
        n_position (int): number of the postion
        d_pos_vec (int): dimension of the position encoding vector
    """
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec)
         for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)
    ])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[0:, 0::2] = np.cos(position_enc[0:, 0::2])

    return torch.from_numpy(position_enc).type(torch.FloatTensor)


