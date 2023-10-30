import torch
import torch.nn as nn

from StableDiffusion.VAE.encoder import Encoder
from StableDiffusion.VAE.decoder import Decoder


def encoder_test():
    x = torch.randn(1, 3, 224, 224)
    noise = torch.randn(1, 4, 27, 27)

    return Encoder()(x, noise)


def decoder_test(x: torch.Tensor):

    return Decoder()(x)


if __name__ == '__main__':
    enc_out = encoder_test()
    print(enc_out.shape)
    dec_out = decoder_test(enc_out)
    print(dec_out.shape)
