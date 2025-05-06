import numpy as np
import torch
import PIL

# transforma imágenes PIL, numpy arrays o int a tensores de PyTorch
class ToTensor:
    def __call__(self, sample):
        if isinstance(sample, PIL.Image.Image):
            sample = np.array(sample)

        if isinstance(sample, np.ndarray):
            # Si es imagen con canales en último eje (H,W,C), lo convertimos a (C,H,W)
            if sample.ndim == 3:
                sample = sample.transpose((2, 0, 1))
            # Si fuese escala de grises (H,W) → (1,H,W), podrías:
            # elif sample.ndim == 2:
            #     sample = sample[np.newaxis, ...]
            return torch.from_numpy(sample).float() / 255.0

        if isinstance(sample, (int, float)):
            return torch.tensor(sample).float()

        raise TypeError(f"ToTensor espera np.ndarray, PIL.Image o escalar, recibió: {type(sample)}")


def build_transform(config):
    return ToTensor()
