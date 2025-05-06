# This class is used to compose several transforms together.
# It takes a list of transforms and applies them sequentially to the input data.
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    
    def __add__(self, other):
        if isinstance(other, Compose):
            return Compose(self.transforms + other.transforms)
        raise TypeError("Only Compose instances can be added.")