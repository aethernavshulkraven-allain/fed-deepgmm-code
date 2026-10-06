import numpy as np
import torch
import random
from collections import defaultdict
from torchvision import datasets, transforms

from scenarios.abstract_scenario import AbstractScenario
from scenarios.toy_scenarios import AGMMZoo, Standardizer

class AbstractCIFAR10Scenario(AbstractScenario):
    def __init__(self, use_x_images, use_z_images, g_function):
        AbstractScenario.__init__(self)
        # Set random seed
        seed = 527
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Data loaders for CIFAR-10
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10("datasets", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                      (0.2023, 0.1994, 0.2010))
                             ])), batch_size=50000)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10("datasets", train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                      (0.2023, 0.1994, 0.2010))
                             ])), batch_size=10000)

        # Load the entire dataset
        train_data, test_data = list(train_loader), list(test_loader)
        images_list = [train_data[0][0].numpy(), test_data[0][0].numpy()]
        labels_list = [train_data[0][1].numpy(), test_data[0][1].numpy()]
        
        # Combine and shuffle data
        self.images = np.concatenate(images_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        idx = list(range(self.images.shape[0]))
        random.shuffle(idx)
        self.images = self.images[idx]
        self.labels = self.labels[idx]
        self.data_i = 0

        self.toy_scenario = AGMMZoo(g_function=g_function, two_gps=False, n_instruments=1, iv_strength=0.5)
        self.use_x_images = use_x_images
        self.use_z_images = use_z_images

    def _sample_images(self, sample_digits, images, labels):
        digit_dict = defaultdict(list)
        for l, image in zip(labels, images):
            digit_dict[int(l)].append(image)
        sampled_images = np.stack([random.choice(digit_dict[int(d)])
                                   for d in sample_digits], axis=0)
        return sampled_images

    def generate_data(self, num_data):
        idx = list(range(self.data_i, self.data_i + num_data))
        images = self.images[idx]
        labels = self.labels[idx]
        self.data_i += num_data

        toy_x, toy_z, toy_y, toy_g, _ = self.toy_scenario.generate_data(num_data)
        if self.use_x_images:
            x_digits = np.clip(1.5 * toy_x[:, 0] + 5.0, 0, 9).round()
            x = self._sample_images(x_digits, images, labels).reshape(-1, 3, 32, 32)
            g = self.toy_scenario._true_g_function_np((x_digits - 5.0) / 1.5).reshape(-1, 1)
            w = x_digits.reshape(-1, 1)
        else:
            x = toy_x.reshape(-1, 1) * 1.5 + 5.0
            g = toy_g.reshape(-1, 1)
            w = toy_x.reshape(-1, 1) * 1.5 + 5.0

        if self.use_z_images:
            z_digits = np.clip(1.5 * toy_z[:, 0] + 5.0, 0, 9).round()
            z = self._sample_images(z_digits, images, labels).reshape(-1, 3, 32, 32)
        else:
            z = toy_z.reshape(-1, 1)

        return x, z, toy_y, g, w

    def true_g_function(self, x):
        raise NotImplementedError()

# Scenario: X Images Only
class CIFAR10ScenarioX(AbstractCIFAR10Scenario):
    def __init__(self, g_function="abs"):
        super().__init__(use_x_images=True, use_z_images=False, g_function=g_function)

    def true_g_function(self, x):
        raise NotImplementedError()

# Scenario: Z Images Only
class CIFAR10ScenarioZ(AbstractCIFAR10Scenario):
    def __init__(self, g_function="abs"):
        super().__init__(use_x_images=False, use_z_images=True, g_function=g_function)

    def true_g_function(self, x):
        raise NotImplementedError()

# Scenario: Both X and Z Images
class CIFAR10ScenarioXZ(AbstractCIFAR10Scenario):
    def __init__(self, g_function="abs"):
        super().__init__(use_x_images=True, use_z_images=True, g_function=g_function)

    def true_g_function(self, x):
        raise NotImplementedError()

# Scenario: None (for testing or baseline)
class CIFAR10ScenarioNone(AbstractCIFAR10Scenario):
    def __init__(self, g_function="abs"):
        super().__init__(use_x_images=False, use_z_images=False, g_function=g_function)

    def true_g_function(self, x):
        raise NotImplementedError()

