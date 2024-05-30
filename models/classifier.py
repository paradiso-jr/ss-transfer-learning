import torch.nn as nn
import torch.nn.functional as F

from models.encoder import TimeSeriesBertEncoder

class TimeSeriesBertClassifier(nn.Module):
    """Transformer based time series classifier.
    Args:
        in_channel (int): The number of input channels.
        h_dim (int): The hidden dimension of the convolutional layer.
        n_res_layers (int): The number of residual layers.
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embedding layer.
        beta (float): The hyperparameter beta.
    """
    def __init__(self,
                in_channel, 
                h_dim, 
                vocab_size, 
                beta, 
                n_labels):

        super().__init__()
        self.encoder = TimeSeriesBertEncoder(in_channel=in_channel, 
                                            h_dim=h_dim, 
                                            vocab_size=vocab_size,
                                            beta=beta,)

        self.fc = nn.Sequential(nn.Linear(768, 384),
                                nn.PReLU(),
                                nn.Linear(384, n_labels),)

    def freeze_cls(self, enable=True):
        for param in self.fc.parameters():
            param.requires_grad = (False if enable else True)
    
    def freeze_encoder(self, enable=True):
        for name, param in self.encoder.named_parameters():
            if ('layer.11' not in name) and ('pooler' not in name):
                param.requires_grad=(False if enable else True)

    def forward(self, x):
        last_hidden_state, pooler_output, embedding_loss, min_encoding_idx  = self.encoder(x)
        # use the last token to predict the label
        logits = self.fc(pooler_output)

        return logits, last_hidden_state, pooler_output, embedding_loss, min_encoding_idx


