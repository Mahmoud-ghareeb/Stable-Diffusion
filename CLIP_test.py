import torch
from StableDiffusion.CLIP.clip import CLIP, Args


def test_clip():

    x = torch.randint(1, 1000, size=(1, 77))
    args = Args()
    out = CLIP(args)(x)
    print(out.shape)


if __name__ == '__main__':
    test_clip()
