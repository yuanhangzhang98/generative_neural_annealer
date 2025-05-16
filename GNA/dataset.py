import torch
from torch.utils.data import Dataset
from GNA.model_utils import dec2bin, bin2dec


class BinaryDataset(Dataset):
    def __init__(self, n, problem):
        self.n = n
        self.problem = problem
        self.samples = torch.zeros((n, 0), dtype=torch.bool)
        self.samples_dec = torch.zeros((0,), dtype=torch.int64)
        self.E = torch.zeros((0,), dtype=torch.float32)
        self.E_min = 1e10
        self.train_samples = torch.zeros((n, 0), dtype=torch.bool)
        self.val_samples = torch.zeros((n, 0), dtype=torch.bool)
        self.train_E = torch.zeros((0,), dtype=torch.float32)
        self.val_E = torch.zeros((0,), dtype=torch.float32)

    def add_samples(self, samples):
        """
        Add samples to the dataset.

        Args:
            samples: Tensor of shape (n, batch_size) with binary variables (0 or 1).
        """

        # TODO: duplication check relies on converting to integer, limited by the number of bits
        # Currently all test problems has n < 64
        # change to another hashing method if n > 64

        dupe_flag = False
        samples_dec = bin2dec(samples.T, self.n)
        mask = torch.isin(samples_dec, self.samples_dec, invert=True)
        if mask.sum() == 0:
            dupe_flag = True
            return dupe_flag
        samples = samples[:, mask]
        samples_dec = samples_dec[mask]
        E = self.problem.E(samples.T).float()
        improve_mask = E < self.E_min
        self.samples = torch.cat([self.samples, samples], dim=1)
        self.samples_dec = torch.cat([self.samples_dec, samples_dec])
        train_mask = (torch.rand(samples.shape[1]) < 0.9) | improve_mask
        self.train_samples = torch.cat([self.train_samples, samples[:, train_mask]], dim=1)
        self.val_samples = torch.cat([self.val_samples, samples[:, ~train_mask]], dim=1)
        self.train_E = torch.cat([self.train_E, E[train_mask]], dim=0)
        self.val_E = torch.cat([self.val_E, E[~train_mask]], dim=0)
        self.E = torch.cat([self.E, E], dim=0)
        self.E_min = self.E.min()

        # Ensure that the validation set is not empty
        if len(self.val_E) < 2:
            n_add = 2 - len(self.val_E)
            self.val_samples = torch.cat([self.val_samples, self.train_samples[:, :n_add]], dim=1)
            self.val_E = torch.cat([self.val_E, self.train_E[:n_add]], dim=0)
            self.train_samples = self.train_samples[:, n_add:]
            self.train_E = self.train_E[n_add:]
        return dupe_flag

    def get_train_samples(self, batch):
        train_size = self.train_samples.shape[1]
        if batch > train_size:
            batch = train_size
        idx = torch.randint(0, train_size, (batch,))
        samples = self.train_samples[:, idx]
        E = self.train_E[idx]
        return samples, E

    def get_val_samples(self):
        return self.val_samples, self.val_E
        # return self.samples, self.E

    def s_best(self):
        """
        Get the best sample from the dataset.
        """
        if self.E_min is None:
            raise ValueError("No samples in the dataset.")
        idx = torch.argmin(self.E)
        return self.samples[:, idx], self.E[idx]

