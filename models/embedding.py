# @source: https://github.com/MishaLaskin/vqvae
# @modified: wujiarong
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.residual import ResidualStack
from models.quantizer import VectorQuantizer

class TimeSeriesCnnEmbedding(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ-VAE, q_theta outputs parameters of a categorical distribution.

    Args:
    - in_channel(int) : the input dimension
    - h_dim(int) : the dimension of hidden conv layers
    - vocab_size(int) : the number of embedding vectors
    - embedding_dim(int) : the dimension of each embedding vector
    - beta(float) : the temperature parameter for softmax
    """

    def __init__(self, 
                in_channel,
                h_dim,
                vocab_size, 
                embedding_dim, 
                beta):
        super(TimeSeriesCnnEmbedding, self).__init__()
        kernel = 64
        stride = 2
        self.cnn_stack = nn.Sequential(
            nn.Conv1d(in_channel, h_dim // 4, kernel_size=kernel, stride=stride),
            nn.BatchNorm1d(h_dim // 4),
            nn.PReLU(),
            nn.Conv1d(h_dim // 4, h_dim // 2, kernel_size=kernel, stride=stride),
            nn.BatchNorm1d(h_dim // 2),
            nn.PReLU(),
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride),
            nn.BatchNorm1d(h_dim),
            nn.PReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=kernel, stride=stride),
            nn.BatchNorm1d(h_dim),
            nn.PReLU(),
        )
        self.quantizer = VectorQuantizer(vocab_size, 
                                        embedding_dim, 
                                        beta)


    def forward(self, x):
        x = self.cnn_stack(x)
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_idx = self.quantizer(x)
        
        return embedding_loss, z_q, perplexity, min_encodings, min_encoding_idx
