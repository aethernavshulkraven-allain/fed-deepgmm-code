import numpy as np
import torch

# Import the Federated EMNIST scenario classes (as defined previously)
from scenarios.emnist_scenario import (
    FederatedEMNISTScenarioX,
    FederatedEMNISTScenarioZ,
    FederatedEMNISTScenarioXZ
)

from scenarios.toy_scenarios import Standardizer


def create_dataset(scenario_class, dir):
    # set random seed
    seed = 527
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set up scenario with the "abs" g_function (adjust if needed)
    num_train = 20000
    num_dev = 10000
    num_test = 10000
    scenario = Standardizer(scenario_class(g_function="abs"))

    # Generate and standardize the train/dev/test splits
    scenario.setup(num_train=num_train, num_dev=num_dev, num_test=num_test)

    # Print summary info (if desired)
    scenario.info()

    # Save the generated data to disk
    scenario.to_file(dir)


if __name__ == "__main__":
    # We create three variants: X, Z, and XZ
    for scenario_cls, path in [
        (FederatedEMNISTScenarioX, "femnist_x"),
        (FederatedEMNISTScenarioZ, "femnist_z"),
        (FederatedEMNISTScenarioXZ, "femnist_xz"),
    ]:
        print("Creating " + path + " ...")
        create_dataset(scenario_cls, "data/" + path + "/main")
