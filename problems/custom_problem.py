import torch


class CustomProblem:
    """
    A template for your custom black-box optimization problem.
    """
    def __init__(self, n):
        # Number of variables in the problem. This is the only required parameter.
        self.n = n

    def E(self, s):
        """
        This is the black-box function that you are trying to minimize.
        Note that this function should be vectorized using Pytorch to handle batch processing.

        Parameters:
        - s: Bool tensor of shape (batch, n) representing the input variables.

        Returns:
        - energy: Tensor of shape (batch,) representing the energy for each configuration.
        """

        energy = s.sum(dim=1)  # Example: simple sum of bits
        return energy
