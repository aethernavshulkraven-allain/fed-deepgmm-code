from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


curves = [
    ("Actual Causal Effect", [
        (-3.0691, 2.9655),
        (0.0020, 0.2728),
        (3.1146, 3.0054),
    ]),
    ("DeepGMM-OAdam", [
        (-3.0691, 5.3287),
        (-2.2468, 3.3347),
        (-1.0819, 0.7680),
        (-1.0362, 0.6680),
        (-0.0042, 0.6668),
        (0.3442, 0.6707),
        (0.9758, 0.6719),
        (1.1163, 0.9425),
        (3.1146, 4.8263),
    ]),
    ("DeepGMM-SGDA", [
        (-3.0691, 3.4784),
        (-0.3122, 0.3616),
        (-0.2117, 0.2722),
        (-0.0722, 0.2721),
        (-0.0459, 0.2814),
        (-0.0010, 0.2982),
        (0.0452, 0.3133),
        (0.0972, 0.3311),
        (0.1216, 0.3482),
        (0.1758, 0.3833),
        (0.2676, 0.4910),
        (0.9328, 1.1223),
        (1.7752, 1.8635),
        (3.1146, 3.0005),
    ]),
    ("FedDeepGMM-SGDA", [
        (-3.0691, 1.6363),
        (0.0043, 1.0022),
        (3.1146, 2.4732),
    ]),
    ("DeepGMM-GDA", [
        (-2.6582, 3.2844),
        (-0.4867, 0.1546),
        (0.0565, 0.1376),
        (0.5313, 0.1434),
        (0.6522, 0.1463),
        (2.4982, 2.3398),
    ]),
    ("FedDeepGMM-GDA", [
        (-3.0691, 1.9275),
        (-0.2210, 0.8370),
        (-0.0717, 0.8207),
        (-0.0124, 0.8232),
        (0.0566, 0.8292),
        (0.1438, 0.8306),
        (0.2719, 0.8336),
        (3.1146, 1.6850),
    ]),
]


styles = {
    "Actual Causal Effect": {"color": "blue", "linewidth": 2.0, "marker": "o"},
    "DeepGMM-OAdam": {"color": "purple", "linestyle": "-.", "marker": "o"},
    "DeepGMM-SGDA": {"color": "green", "linestyle": ":", "marker": "o"},
    "FedDeepGMM-SGDA": {"color": "orange", "linestyle": "-", "marker": "o"},
    "DeepGMM-GDA": {"color": "gray", "linestyle": "--", "marker": "o"},
    "FedDeepGMM-GDA": {"color": "brown", "linestyle": "--", "marker": "o"},
    "FedDeepGMM-OGDA": {"color": "red", "linestyle": "-", "linewidth": 1.5},
}


def load_ogda_points():
    csv_path = BASE_DIR / "results_abs_ogda_xy.csv"
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return data["x"], data["y"]


def generate_abs_points_plot():
    plt.figure(figsize=(7, 5))

    for label, points in curves:
        values = np.array(points)
        plt.plot(
            values[:, 0],
            values[:, 1],
            label=label,
            markersize=3.5,
            **styles[label],
        )

    ogda_x, ogda_y = load_ogda_points()
    plt.plot(ogda_x, ogda_y, label="FedDeepGMM-OGDA", **styles["FedDeepGMM-OGDA"])

    plt.title("(a) Absolute Scenario")
    plt.xlabel("x")
    plt.ylabel("g(x)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = BASE_DIR / "abs_points_with_ogda.png"
    plt.savefig(output_path, dpi=300)
    print(f"Graph generated successfully as '{output_path.name}'")
    plt.show()


if __name__ == "__main__":
    generate_abs_points_plot()
