
import numpy as np


class AugmentMethod(object):
    """
    Augmentation method.
    """
    def __init__(self, n_method):
        self.n_method = n_method
        self.data_augment = [self.RandomErasing]

    def RandomErasing(self, xs, value=0, prop=0.2):
        """Randomly selects prop number of element to erase.
        Args:
            xs (tensor): input tensor, (sequence_lenth,) for unbatched data, 
                         (batch_size, sequence_length) for batched data.
            value (int): value to fill erased region.
            prop (float): proportion of erasing.
        """
        if len(xs.shape) == 1:
            sequence_length = xs.size
            idxs = np.random.choice(np.arange(sequence_length),
                                    replace=False,
                                    size=int(sequence_length * prop))
            xs[idxs] = value
            return xs
        else:
            _, sequence_length = xs.shape

            for x in xs:
                idxs = np.random.choice(np.arange(sequence_length),
                                        replace=False,
                                        size=(int(sequence_length * prop)))
                x[idxs] = value
            return xs
            
    def __call__(self, xs):
        """
        Args:
            xs (tensor): input tensor.
        Returns:
            tensor: augmented data.
        """
        augment = np.random.choice(self.data_augment, self.n_method)
        for aug in augment:
            xs = aug(xs)
        return xs

