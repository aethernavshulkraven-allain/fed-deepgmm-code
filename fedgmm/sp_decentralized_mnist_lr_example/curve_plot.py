# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Load the data saved at the end of the run
# x_values = np.load("results_linear_sgd_x.npy")
# y_pred = np.load("results_linear_sgd_y_pred.npy")
# y_true = np.load("results_linear_sgd_y_true.npy")

# # 2. Create the Plot
# plt.figure(figsize=(6, 4))

# # Plot the Ground Truth (The blue line in your image)
# plt.plot(x_values, y_true, label="Actual Causal Effect", color='blue', linewidth=2)

# # Plot your GDA/SGD Prediction (The purple/brown lines in your image)
# plt.plot(x_values, y_pred, label="FedDeepGMM-GDA", color='brown', linestyle='-')

# # Formatting to match the image
# plt.title("(c) Linear Scenario")
# plt.xlabel("x")
# plt.ylabel("g(x)")
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.legend()

# # 3. Save and Show
# plt.savefig("linear_scenario_results.png")
# plt.show()
###################################################################################################################
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Load the GDA data (SGD)
# x_values = np.load("results_linear_sgd_x.npy")
# y_pred_gda = np.load("results_linear_sgd_y_pred.npy")
# y_true = np.load("results_linear_sgd_y_true.npy")

# # 2. Load the OGDA data (New)
# # Note: x_values are usually identical if using the same seed, but we load them to be safe
# y_pred_ogda = np.load("results_linear_ogda_y_pred.npy")

# # 3. Create the Plot
# plt.figure(figsize=(8, 6))

# # Plot Ground Truth
# plt.plot(x_values, y_true, label="Actual Causal Effect", color='blue', linewidth=2)

# # Plot GDA (Standard SGD)
# plt.plot(x_values, y_pred_gda, label="FedDeepGMM-GDA", color='brown', linestyle='--', alpha=0.8)

# # Plot OGDA (Optimistic GDA)
# plt.plot(x_values, y_pred_ogda, label="FedDeepGMM-OGDA", color='red', linestyle='-', linewidth=1.5)

# # Formatting to match the image aesthetics
# plt.title("(c) Linear Scenario Comparison")
# plt.xlabel("x")
# plt.ylabel("g(x)")
# plt.grid(True, which='both', linestyle=':', alpha=0.6)
# plt.legend()

# # 4. Save and Show
# plt.savefig("linear_gda_vs_ogda_comparison.png", dpi=300)
# plt.show()
####################################################################################
# import numpy as np
# import matplotlib.pyplot as plt

# def create_comparison_plots():
#     # --- PLOT 1: LINEAR (Actual, GDA, OGDA) ---
#     plt.figure(figsize=(7, 5))
    
#     # Load Linear Data
#     x_lin = np.load("results_linear_sgd_x.npy")
#     y_true_lin = np.load("results_linear_sgd_y_true.npy")
#     y_gda_lin = np.load("results_linear_sgd_y_pred.npy")
#     y_ogda_lin = np.load("results_linear_ogda_y_pred.npy")
    
#     plt.plot(x_lin, y_true_lin, label="Actual Causal Effect", color='blue', linewidth=2)
#     plt.plot(x_lin, y_gda_lin, label="FedDeepGMM-GDA", color='brown', linestyle='--')
#     plt.plot(x_lin, y_ogda_lin, label="FedDeepGMM-OGDA", color='red', linestyle='-')
    
#     plt.title("(c) Linear Scenario")
#     plt.xlabel("x")
#     plt.ylabel("g(x)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig("plot_linear_comparison.png")
#     plt.show()

#     # --- PLOT 2: ABSOLUTE (Actual, GDA) ---
#     plt.figure(figsize=(7, 5))
    
#     # Load Absolute Data
#     # Assuming files are named 'abs' based on your YAML setting
#     x_abs = np.load("results_abs_sgd_x.npy")
#     y_true_abs = np.load("results_abs_sgd_y_true.npy")
#     y_gda_abs = np.load("results_abs_sgd_y_pred.npy")
    
#     plt.plot(x_abs, y_true_abs, label="Actual Causal Effect", color='blue', linewidth=2)
#     plt.plot(x_abs, y_gda_abs, label="FedDeepGMM-GDA", color='brown', linestyle='--')
    
#     plt.title("(a) Absolute Scenario")
#     plt.xlabel("x")
#     plt.ylabel("g(x)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig("plot_abs_comparison.png")
#     plt.show()

# if __name__ == "__main__":
#     create_comparison_plots()
##################################################################################
# import numpy as np
# import matplotlib.pyplot as plt

# def generate_final_comparison():
#     # Set up the figure with 2 subplots (1 row, 2 columns)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#     # --- PLOT 1: ABSOLUTE SCENARIO ---
#     x_abs = np.load("results_abs_sgd_x.npy")
#     y_true_abs = np.load("results_abs_sgd_y_true.npy")
#     y_gda_abs = np.load("results_abs_sgd_y_pred.npy")
#     y_ogda_abs = np.load("results_abs_ogda_y_pred.npy")

#     ax1.plot(x_abs, y_true_abs, label="Actual Causal Effect", color='blue', linewidth=2)
#     ax1.plot(x_abs, y_gda_abs, label="FedDeepGMM-GDA", color='brown', linestyle='--')
#     ax1.plot(x_abs, y_ogda_abs, label="FedDeepGMM-OGDA", color='red', linestyle='-')
    
#     ax1.set_title("(a) Absolute Scenario")
#     ax1.set_xlabel("x")
#     ax1.set_ylabel("g(x)")
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()

#     # --- PLOT 2: LINEAR SCENARIO ---
#     x_lin = np.load("results_linear_sgd_x.npy")
#     y_true_lin = np.load("results_linear_sgd_y_true.npy")
#     y_gda_lin = np.load("results_linear_sgd_y_pred.npy")
#     y_ogda_lin = np.load("results_linear_ogda_y_pred.npy")

#     ax2.plot(x_lin, y_true_lin, label="Actual Causal Effect", color='blue', linewidth=2)
#     ax2.plot(x_lin, y_gda_lin, label="FedDeepGMM-GDA", color='brown', linestyle='--')
#     ax2.plot(x_lin, y_ogda_lin, label="FedDeepGMM-OGDA", color='red', linestyle='-')
    
#     ax2.set_title("(c) Linear Scenario")
#     ax2.set_xlabel("x")
#     ax2.set_ylabel("g(x)")
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()

#     # Layout and Save
#     plt.tight_layout()
#     plt.savefig("Final_Federated_Comparison_Results.png", dpi=300)
#     print("Graphs generated successfully as 'Final_Federated_Comparison_Results.png'")
#     plt.show()

# if __name__ == "__main__":
#     generate_final_comparison()
######################################## 3 plots ###########################################################
import numpy as np
import matplotlib.pyplot as plt

def generate_final_comparison():
    # Set up the figure with 3 subplots (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # --- PLOT 1: (a) ABSOLUTE SCENARIO ---
    x_abs = np.load("results_abs_sgd_x.npy")
    y_true_abs = np.load("results_abs_sgd_y_true.npy")
    y_gda_abs = np.load("results_abs_sgd_y_pred.npy")
    y_ogda_abs = np.load("results_abs_ogda_y_pred.npy")
    y_ogda_abs_prev = np.load("results_abs_ogda_y_prednew.npy")
    # y_ogda_abs_minibatch= np.load("results_abs_ogda_y_prednewmb.npy")
    # y_fbgda_abs = np.load("results_abs_ogda_y_prednewfb.npy")


    ax1.plot(x_abs, y_true_abs, label="Actual Causal Effect", color='blue', linewidth=2)
    ax1.plot(x_abs, y_gda_abs, label="FedDeepGMM-GDA", color='brown', linestyle='--')
    ax1.plot(x_abs, y_ogda_abs, label="FedDeepGMM-OGDA", color='red', linestyle='-')
    ax1.plot(x_abs, y_ogda_abs_prev, label="FedDeepGMM-OGDA (Previous)", color='purple', linestyle='-.')
    # ax1.plot(x_abs, y_ogda_abs_minibatch, label="FedDeepGMM-OGDA (Minibatch)", color='green', linestyle=':')
    # ax1.plot(x_abs, y_fbgda_abs, label="FedDeepGMM-OGDA (Full Batch)", color='orange', linestyle='--')

    ax1.set_title("(a) Absolute")
    ax1.set_xlabel("x")
    ax1.set_ylabel("g(x)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- PLOT 2: (b) STEP SCENARIO ---
    x_step = np.load("results_step_sgd_x.npy")
    y_true_step = np.load("results_step_sgd_y_true.npy")
    y_gda_step = np.load("results_step_sgd_y_prednew.npy")
    y_ogda_step = np.load("results_step_ogda_y_prednew.npy")
    y_fbgda_step = np.load("results_step_sgd_y_prednew_fullbatch.npy")
    y_fb_ogda_step = np.load("results_step_ogda_y_prednew_fullbatch.npy")

    ax2.plot(x_step, y_true_step, label="Actual Causal Effect", color='blue', linewidth=2)
    ax2.plot(x_step, y_gda_step, label="FedDeepGMM-SGDA", color='brown', linestyle='--')
    ax2.plot(x_step, y_ogda_step, label="FedDeepGMM-SOGDA", color='red', linestyle='-')
    ax2.plot(x_step, y_fbgda_step, label="FedDeepGMM-GDA (FB)", color='green', linestyle='--')
    ax2.plot(x_step, y_fb_ogda_step, label="FedDeepGMM-OGDA (FB)", color='orange', linestyle='-')
    
    ax2.set_title("(b) Step")
    ax2.set_xlabel("x")
    ax2.set_ylabel("g(x)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- PLOT 3: (c) LINEAR SCENARIO ---
    x_lin = np.load("results_linear_sgd_x.npy")
    y_true_lin = np.load("results_linear_sgd_y_true.npy")
    y_gda_lin = np.load("results_linear_sgd_y_pred.npy")
    y_ogda_lin = np.load("results_linear_ogda_y_pred.npy")

    ax3.plot(x_lin, y_true_lin, label="Actual Causal Effect", color='blue', linewidth=2)
    ax3.plot(x_lin, y_gda_lin, label="FedDeepGMM-GDA", color='brown', linestyle='--')
    ax3.plot(x_lin, y_ogda_lin, label="FedDeepGMM-OGDA", color='red', linestyle='-')
    
    ax3.set_title("(c) Linear")
    ax3.set_xlabel("x")
    ax3.set_ylabel("g(x)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Layout and Save
    plt.tight_layout()
    plt.savefig("All_3.png", dpi=300)
    print("Graphs generated successfully as 'Final_Federated_Comparison_Results_All_3.png'")
    plt.show()

def generate_abs_fedeg_comparison():
    """Plot exact FedEG and zeroth-order FedEG on the absolute scenario."""
    from pathlib import Path

    result_dir = Path(__file__).resolve().parent

    x_fedeg = np.load(result_dir / "results_abs_fed_eg_x.npy").squeeze()
    y_fedeg = np.load(
        result_dir / "results_abs_fed_eg_y_prednewtrial.npy"
    ).squeeze()
    y_true = np.load(result_dir / "results_abs_fed_eg_y_true.npy").squeeze()

    x_zo_q1 = np.load(result_dir / "results_abs_fed_zo_eg_xq1.npy").squeeze()
    y_zo_q1 = np.load(
        result_dir / "results_abs_fed_zo_eg_y_prednewtrialq1.npy"
    ).squeeze()

    x_zo_q4 = np.load(result_dir / "results_abs_fed_zo_eg_x.npy").squeeze()
    y_zo_q4 = np.load(
        result_dir / "results_abs_fed_zo_eg_y_prednewtrial.npy"
    ).squeeze()

    # Sort each run independently in case their saved sample order differs.
    fedeg_order = np.argsort(x_fedeg)
    zo_q1_order = np.argsort(x_zo_q1)
    zo_q4_order = np.argsort(x_zo_q4)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        x_fedeg[fedeg_order],
        y_true[fedeg_order],
        label="Ground truth: absolute function",
        color="black",
        linewidth=2.5,
    )
    ax.plot(
        x_fedeg[fedeg_order],
        y_fedeg[fedeg_order],
        label="FedEG (exact second pass)",
        color="tab:blue",
        linewidth=2,
    )
    ax.plot(
        x_zo_q1[zo_q1_order],
        y_zo_q1[zo_q1_order],
        label="FedEG-ZO (zeroth-order second pass- Q1)",
        color="tab:orange",
        linestyle="--",
        linewidth=2,
    )
    ax.plot(
        x_zo_q4[zo_q4_order],
        y_zo_q4[zo_q4_order],
        label="FedEG-ZO (zeroth-order second pass- Q4)",
        color="tab:red",
        linestyle="--",
        linewidth=2,
    )

    ax.set_title("Absolute Function: FedEG vs. FedEG-ZO")
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    output_path = result_dir / "abs_fedeg_vs_fedzoeg_q1_q4.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph generated successfully: {output_path}")


if __name__ == "__main__":
    generate_abs_fedeg_comparison()
