import torch.nn as nn

from .models import register, make


@register('encoder-decoder')
class EncoderDecoder(nn.Module):

    def __init__(self, encoder, neck=None, decoder=None):
        super().__init__()
        self.encoder = make(encoder)
        out_dim = self.encoder.out_dim
        self.neck = make(neck, args={'in_dim': out_dim})
        if neck is not None:
            out_dim = self.neck.out_dim
        self.decoder = make(decoder, args={'in_dim': out_dim})

    def forward(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.decoder(x)
        return x
