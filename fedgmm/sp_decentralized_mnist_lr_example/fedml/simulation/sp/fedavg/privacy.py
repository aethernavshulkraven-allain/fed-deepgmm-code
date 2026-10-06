import numpy as np

class GaussianMechanism:
    def __init__(self, sensitivity, epsilon, delta):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta

    def add_noise(self, gradients):
        """
        Add Gaussian noise to gradients based on L2 sensitivity.
        """
        # Compute standard deviation (σ) for the noise
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        # Generate Gaussian noise
        noise = np.random.normal(0, sigma, size=gradients.shape)
        return gradients + noise

    def clip(self, gradients, clipping_threshold):
        """
        Clip the gradients to ensure bounded sensitivity.
        """
        norm = np.linalg.norm(gradients, ord=2)
        if norm > clipping_threshold:
            gradients = (gradients / norm) * clipping_threshold
        return gradients
