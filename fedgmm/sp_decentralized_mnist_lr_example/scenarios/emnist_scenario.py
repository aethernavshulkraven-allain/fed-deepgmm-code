from collections import defaultdict
import random
import numpy as np
import torch
from torchvision import datasets, transforms

# Monkeypatch the broken EMNIST URL (NIST moved the file)
datasets.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

from scenarios.abstract_scenario import AbstractScenario
from scenarios.toy_scenarios import AGMMZoo


class AbstractFederatedEMNISTScenario(AbstractScenario):
    """
    A scenario class analogous to AbstractMNISTScenario, but using
    the EMNIST dataset (with 'digits' split by default).
    You can change the 'split' to 'byclass', 'balanced', etc. if desired.
    """

    def __init__(self, use_x_images, use_z_images, g_function, 
                 emnist_split="digits"):
        super().__init__()

        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(
                root="datasets",
                split="digits",        # e.g. "digits", "byclass", "balanced", etc.
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    # Normalizing similarly to MNIST; feel free to adjust
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=60000  # load entire train set in one batch for concatenation
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(
                root="datasets",
                split="digits",
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=10000  # load entire test set in one batch
        )

  
        train_data = list(train_loader)  # single batch of entire train set
        test_data = list(test_loader)    # single batch of entire test set

        images_list = [train_data[0][0].numpy(), test_data[0][0].numpy()]
        labels_list = [train_data[0][1].numpy(), test_data[0][1].numpy()]

        self.images = np.concatenate(images_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)

        # -------------------------------------------------------
        # 3) Shuffle the combined dataset
        # -------------------------------------------------------
        idx = list(range(self.images.shape[0]))
        random.shuffle(idx)
        self.images = self.images[idx]
        self.labels = self.labels[idx]

        # Keep track of how far we've drawn from the dataset so far
        self.data_i = 0

        # -------------------------------------------------------
        # 4) Construct the internal "toy" scenario to produce
        #    synthetic relationships among x, z, y, etc.
        # -------------------------------------------------------
        self.toy_scenario = AGMMZoo(
            g_function=g_function,
            two_gps=False,
            n_instruments=1,
            iv_strength=0.5
        )

        self.use_x_images = use_x_images
        self.use_z_images = use_z_images

    def _sample_images(self, sample_digits, images, labels):
        """
        Given a list of digit labels (sample_digits), return
        the corresponding EMNIST images at those digits, chosen randomly.
        """
        digit_dict = defaultdict(list)
        for lbl, img in zip(labels, images):
            digit_dict[int(lbl)].append(img)

        # For each digit in sample_digits, pick a random image
        out = np.stack([random.choice(digit_dict[int(d)]) for d in sample_digits], axis=0)
        return out

    def generate_data(self, num_data, **kwargs):
        """
        Generate (x, z, y, g, w) just like in AbstractMNISTScenario.
        We reuse a toy scenario (AGMMZoo) to get numeric x, z, y, g,
        then optionally replace x or z with EMNIST images. 
        """

        # 1) Numeric scenario from toy_scenario
        toy_x, toy_z, toy_y, toy_g, _ = self.toy_scenario.generate_data(num_data)

        # 2) If using EMNIST images for x:
        if self.use_x_images:
            # Convert numeric x to [0..9] for indexing digits
            # (Here, shift and scale to ensure within 0..9 range)
            x_digits = np.clip(1.5 * toy_x[:, 0] + 5.0, 0, 9).round()
            
            # Sample images from EMNIST
            images_x = self._sample_images(x_digits, self.images, self.labels)
            x = images_x.reshape(-1, 1, 28, 28)  # shape consistent with single-channel
            # Evaluate or reuse g from toy scenario
            g = self.toy_scenario._true_g_function_np((x_digits - 5.0) / 1.5).reshape(-1, 1)
            w = x_digits.reshape(-1, 1)
        else:
            # Just use numeric x from toy scenario
            x = toy_x.reshape(-1, 1) * 1.5 + 5.0
            g = toy_g.reshape(-1, 1)
            w = x

        # 3) If using EMNIST images for z:
        if self.use_z_images:
            z_digits = np.clip(1.5 * toy_z[:, 0] + 5.0, 0, 9).round()
            images_z = self._sample_images(z_digits, self.images, self.labels)
            z = images_z.reshape(-1, 1, 28, 28)
        else:
            # Just use numeric z from toy scenario
            z = toy_z.reshape(-1, 1)

        # The labels y come from the toy scenario as well
        y = toy_y.reshape(-1, 1)

        return x, z, y, g, w

    def true_g_function(self, x):
        """
        If you need a closed-form ground-truth g for numeric x,
        you can implement it here. Because this scenario uses
        AGMMZoo's built-in g, we often just rely on the toy_scenario
        for 'true_g_function'.
        """
        raise NotImplementedError("If needed, implement the ground-truth g here.")


# -----------------------------------------------------------------
# Just like we had separate scenario classes for MNIST (X/Z/XZ/None),
# we define small wrappers that specify use_x_images/use_z_images.
# You can adjust or remove these as needed.
# -----------------------------------------------------------------

class FederatedEMNISTScenarioX(AbstractFederatedEMNISTScenario):
    def __init__(self, g_function="abs", emnist_split="digits"):
        super().__init__(
            use_x_images=True,
            use_z_images=False,
            g_function=g_function,
            emnist_split=emnist_split
        )

    def true_g_function(self, x):
        # Typically we'd rely on toy_scenario; override if needed.
        raise NotImplementedError()


class FederatedEMNISTScenarioZ(AbstractFederatedEMNISTScenario):
    def __init__(self, g_function="abs", emnist_split="digits"):
        super().__init__(
            use_x_images=False,
            use_z_images=True,
            g_function=g_function,
            emnist_split=emnist_split
        )

    def true_g_function(self, x):
        raise NotImplementedError()


class FederatedEMNISTScenarioXZ(AbstractFederatedEMNISTScenario):
    def __init__(self, g_function="abs", emnist_split="digits"):
        super().__init__(
            use_x_images=True,
            use_z_images=True,
            g_function=g_function,
            emnist_split=emnist_split
        )

    def true_g_function(self, x):
        raise NotImplementedError()


class FederatedEMNISTScenarioNone(AbstractFederatedEMNISTScenario):
    def __init__(self, g_function="abs", emnist_split="digits"):
        super().__init__(
            use_x_images=False,
            use_z_images=False,
            g_function=g_function,
            emnist_split=emnist_split
        )

    def true_g_function(self, x):
        raise NotImplementedError()
