import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Problem definition
# ------------------------------------------------------------

# Clients are represented by shifts (a, b) in
# f^i(x,y) = gamma * x*y + lambda * (1/3) * (x-a)^3 * (y-b)^3
GAMMA = 1.0
LAMBDA = 0.2

CLIENTS = [
    (0.0, 0.0),   # f1
    (1.0, -1.0),  # f2
    (-1.0, 1.0),  # f3
]

def f_client(x, y, a, b):
    bilinear = GAMMA * x * y
    cubic = ((x - a)**3 * (y - b)**3) / 3.0
    return bilinear + LAMBDA * cubic

def grad_client(x, y, a, b):
    # d/dx = gamma*y + lambda*(x-a)^2*(y-b)^3
    # d/dy = gamma*x + lambda*(x-a)^3*(y-b)^2
    gx = GAMMA * y + LAMBDA * (x - a)**2 * (y - b)**3
    gy = GAMMA * x + LAMBDA * (x - a)**3 * (y - b)**2
    return gx, gy

def f_global(x, y):
    val = 0.0
    for a, b in CLIENTS:
        val += f_client(x, y, a, b)
    return val / len(CLIENTS)

def make_contour_grid(xlim=(-2, 2), ylim=(-2, 2), n=300):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for a, b in CLIENTS:
        bilinear = GAMMA * X * Y
        cubic = ((X - a)**3 * (Y - b)**3) / 3.0
        Z += bilinear + LAMBDA * cubic
    Z /= len(CLIENTS)
    return X, Y, Z

# ------------------------------------------------------------
# 2. FedGDA
# ------------------------------------------------------------

def fed_gda(x0, y0, T=200, R=1, alpha_x=0.02, alpha_y=0.02):
    """
    FedGDA:
      - local GDA for R steps on each client
      - aggregate local displacements
      - plain server averaging
    """
    x, y = float(x0), float(y0)
    traj = [(x, y)]

    for t in range(T):
        delta_x_list = []
        delta_y_list = []

        for a, b in CLIENTS:
            xc, yc = x, y

            for r in range(R):
                gx, gy = grad_client(xc, yc, a, b)
                xc = xc - alpha_x * gx   # descent in x
                yc = yc + alpha_y * gy   # ascent in y

            delta_x_list.append(xc - x)
            delta_y_list.append(yc - y)

        delta_x = np.mean(delta_x_list)
        delta_y = np.mean(delta_y_list)

        x = x + delta_x
        y = y + delta_y

        traj.append((x, y))

        if not np.isfinite(x) or not np.isfinite(y):
            print(f"FedGDA diverged at round {t+1}. Try smaller step sizes.")
            break

    return np.array(traj)

# ------------------------------------------------------------
# 3. FedOGDA (corrected)
# ------------------------------------------------------------

def fed_ogda(
    x0, y0,
    T=200, R=1,
    alpha_x=0.02, alpha_y=0.02,
    beta_x=1.0, beta_y=1.0,
    use_server_ogda=True,
    use_local_ogda=True
):
    """
    FedOGDA with flexible local and server updates:
      - use_local_ogda=True: local OGDA (with warm start)
      - use_local_ogda=False: local GDA only (no OGDA steps)
      - use_server_ogda=True: server OGDA on aggregated deltas
      - use_server_ogda=False: plain server averaging
    """
    x, y = float(x0), float(y0)
    traj = [(x, y)]

    prev_delta_x = None
    prev_delta_y = None

    for t in range(T):
        delta_x_list = []
        delta_y_list = []

        for a, b in CLIENTS:
            # local updates
            x_prev, y_prev = x, y
            x_curr, y_curr = x, y

            if R >= 1:
                # First step: vanilla GDA
                gx, gy = grad_client(x_curr, y_curr, a, b)
                x_next = x_curr - alpha_x * gx
                y_next = y_curr + alpha_y * gy

                x_prev, y_prev = x_curr, y_curr
                x_curr, y_curr = x_next, y_next

            # Remaining steps: OGDA or plain GDA depending on use_local_ogda
            if use_local_ogda:
                # Remaining local steps: OGDA
                for r in range(2, R + 1):
                    gx_curr, gy_curr = grad_client(x_curr, y_curr, a, b)
                    gx_prev, gy_prev = grad_client(x_prev, y_prev, a, b)

                    x_next = x_curr - 2.0 * alpha_x * gx_curr + alpha_x * gx_prev
                    y_next = y_curr + 2.0 * alpha_y * gy_curr - alpha_y * gy_prev

                    x_prev, y_prev = x_curr, y_curr
                    x_curr, y_curr = x_next, y_next
            else:
                # Remaining local steps: plain GDA
                for r in range(2, R + 1):
                    gx, gy = grad_client(x_curr, y_curr, a, b)

                    x_next = x_curr - alpha_x * gx
                    y_next = y_curr + alpha_y * gy

                    x_curr, y_curr = x_next, y_next

            delta_x_list.append(x_curr - x)
            delta_y_list.append(y_curr - y)

        delta_x = np.mean(delta_x_list)
        delta_y = np.mean(delta_y_list)

        # Server update
        if use_server_ogda:
            if prev_delta_x is None:
                # warm start: plain first step
                x = x + beta_x * delta_x
                y = y + beta_y * delta_y
            else:
                x = x + 2.0 * beta_x * delta_x - beta_x * prev_delta_x
                y = y + 2.0 * beta_y * delta_y - beta_y * prev_delta_y
        else:
            # plain server averaging
            x = x + beta_x * delta_x
            y = y + beta_y * delta_y

        prev_delta_x = delta_x
        prev_delta_y = delta_y

        traj.append((x, y))

        if not np.isfinite(x) or not np.isfinite(y):
            print(f"FedOGDA diverged at round {t+1}. Try smaller step sizes.")
            break

    return np.array(traj)

# ------------------------------------------------------------
# 4. Plotting
# ------------------------------------------------------------

def plot_phase_trajectories(trajs, labels=None, xlim=(-2, 2), ylim=(-2, 2)):
    """Plot phase trajectories for multiple algorithms."""
    if not isinstance(trajs, list):
        trajs = [trajs]
    if labels is None:
        labels = [f"traj {i}" for i in range(len(trajs))]
    
    X, Y, Z = make_contour_grid(xlim=xlim, ylim=ylim, n=350)
    lo = np.percentile(Z, 10)
    hi = np.percentile(Z, 90)
    levels = np.linspace(lo, hi, 25)

    n_plots = len(trajs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, traj, label in zip(axes, trajs, labels):
        ax.contour(X, Y, Z, levels=levels, linewidths=0.8)
        ax.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2, linewidth=1.5, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], marker='s', s=60, label='start')
        ax.scatter(traj[-1, 0], traj[-1, 1], marker='*', s=120, label='end')
        ax.set_title(f'{label} trajectory', fontsize=16)
        ax.set_xlabel(r'$\theta_t$', fontsize=14)
        ax.set_ylabel(r'$\tau_t$', fontsize=14)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)

    plt.tight_layout()
    fig.savefig(f'Final_phase_trajectories_lambda{LAMBDA}_alpha{alpha_x}_beta{beta_x}200.png', dpi=200)
    plt.close(fig)


def plot_timeseries(trajs, labels=None):
    """Plot time series for multiple algorithms."""
    if not isinstance(trajs, list):
        trajs = [trajs]
    if labels is None:
        labels = [f"traj {i}" for i in range(len(trajs))]

    n_plots = len(trajs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, traj, label in zip(axes, trajs, labels):
        t = np.arange(len(traj))
        ax.plot(t, traj[:, 0], label=r'$\theta_t$')
        ax.plot(t, traj[:, 1], label=r'$\tau_t$')
        ax.set_title(f'{label} dynamics', fontsize=16)
        ax.set_xlabel('Round', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)

    plt.tight_layout()
    fig.savefig(f'Final_timeseries_lambda{LAMBDA}_alpha{alpha_x}_beta{beta_x}200.png', dpi=200)
    plt.close(fig)


def plot_last_iterate_zoom(trajs, labels=None, last_k=50):
    """Plot last k iterates for multiple algorithms."""
    if not isinstance(trajs, list):
        trajs = [trajs]
    if labels is None:
        labels = [f"traj {i}" for i in range(len(trajs))]

    n_plots = len(trajs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, traj, label in zip(axes, trajs, labels):
        zoomed = traj[-last_k:]
        ax.plot(zoomed[:, 0], zoomed[:, 1], marker='o', markersize=3, linewidth=1.5)
        ax.scatter(zoomed[0, 0], zoomed[0, 1], marker='s', s=60, label='start of zoom')
        ax.scatter(zoomed[-1, 0], zoomed[-1, 1], marker='*', s=120, label='last iterate')
        ax.set_title(f'{label} last {last_k} iterates', fontsize=16)
        ax.set_xlabel(r'$\theta_t$', fontsize=14)
        ax.set_ylabel(r'$\tau_t$', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)

    plt.tight_layout()
    fig.savefig(f'Final_last_iterate_zoom_lambda{LAMBDA}_alpha{alpha_x}_beta{beta_x}200.png', dpi=200)
    plt.close(fig)


def compute_experiment_metrics(traj):
    final_x, final_y = traj[-1]
    norms = np.linalg.norm(traj, axis=1)
    return {
        'final_x': float(final_x),
        'final_y': float(final_y),
        'final_norm': float(norms[-1]),
        'min_norm': float(np.min(norms)),
        'max_norm': float(np.max(norms)),
        'trajectory_length': int(len(traj)),
    }


def save_experiment_results(results, filename='experiment_results.xlsx'):
    if not results:
        print('No results to save.')
        return

    csv_filename = os.path.splitext(filename)[0] + '.csv'

    try:
        import pandas as pd
        df_new = pd.DataFrame(results)
        
        if os.path.exists(csv_filename):
            df_existing = pd.read_csv(csv_filename)
            df = pd.concat([df_existing, df_new], ignore_index=True)
            print(f'Appended {len(results)} new row(s) to {csv_filename}')
        else:
            df = df_new
            print(f'Created new results file {csv_filename}')
        
        df.to_csv(csv_filename, index=False)
        try:
            df.to_excel(filename, index=False)
            print(f'Saved to {filename} ({len(df)} total rows)')
        except Exception as exc:
            print(f'Could not save Excel file {filename}: {exc}. Saved CSV instead.')
        return
    except ImportError:
        pass

    fieldnames = results[0].keys()
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    
    if file_exists:
        print(f"Appended {len(results)} new row(s) to {csv_filename} (pandas not installed).")
    else:
        print(f"Created new results file {csv_filename} (pandas not installed).")


def collect_experiment_result(method, params, traj):
    metrics = compute_experiment_metrics(traj)
    return {
        'method': method,
        **params,
        **metrics,
    }


def plot_radius(*trajs, labels=None, equilibrium=(0.0, 0.0), filename=None):
    """Plot distance from equilibrium r_t = sqrt((theta_t-theta*)^2 + (tau_t-tau*)^2)."""
    plt.figure(figsize=(5 * max(1, len(trajs)), 5))

    if labels is None:
        labels = [f"traj {i}" for i in range(len(trajs))]

    theta_star, tau_star = equilibrium

    for traj, label in zip(trajs, labels):
        r = np.sqrt((traj[:, 0] - theta_star)**2 + (traj[:, 1] - tau_star)**2)
        plt.plot(r, label=label)

    plt.xlabel("Round", fontsize=14)
    plt.ylabel(r"$r_t=\sqrt{(\theta_t-\theta^\star)^2+(\tau_t-\tau^\star)^2}$", fontsize=14)
    plt.title("Distance from equilibrium", fontsize=16)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if filename is None:
        filename = f"Final_radius_plot_gamma{GAMMA}_lambda{LAMBDA}_R{R}_alpha{alpha_x}_beta{beta_x}200.png"

    plt.savefig(filename, dpi=200)
    plt.close()

# ------------------------------------------------------------
# 5. Example run
# ------------------------------------------------------------

if __name__ == "__main__":
    # Good initial settings for probing
    x0, y0 = 0.4, -0.4
    T = 200
    R = 5            # start with R=1 to isolate cycling cleanly
    alpha_x = 0.03
    alpha_y = 0.03
    beta_x = 2.0
    beta_y = 2.0


    # Algorithm 1: FedGDA
    traj_gda = fed_gda(
        x0, y0,
        T=T, R=R,
        alpha_x=alpha_x, alpha_y=alpha_y
    )

    # Algorithm 2: FedOGDA-local (no server OGDA)

    # traj_ogda_local = fed_ogda(
    #     x0, y0,
    #     T=T, R=R,
    #     alpha_x=alpha_x, alpha_y=alpha_y,
    #     beta_x=beta_x, beta_y=beta_y,
    #     use_server_ogda=False
    # )

    # Algorithm 3: FedOGDA-double (with server OGDA)
    traj_ogda_double = fed_ogda(
        x0, y0,
        T=T, R=R,
        alpha_x=alpha_x, alpha_y=alpha_y,
        beta_x=beta_x, beta_y=beta_y,
        use_server_ogda=True,
        use_local_ogda=True
    )

    # Algorithm 4: FedOGDA-server-only (client GDA only + server OGDA)
    # traj_ogda_server_only = fed_ogda(
    #     x0, y0,
    #     T=T, R=R,
    #     alpha_x=alpha_x, alpha_y=alpha_y,
    #     beta_x=beta_x, beta_y=beta_y,
    #     use_server_ogda=True,
    #     use_local_ogda=False
    # )

    # Plot radius to see convergence/cycling/divergence
    plot_radius(
        traj_gda,
        traj_ogda_double,
        labels=["FedGDA", "FedOGDA"],
        equilibrium=(0.0, 0.0)
    )

    results = []
    params_gda = {
        'gamma': GAMMA,
        'lambda': LAMBDA,
        'x0': x0,
        'y0': y0,
        'T': T,
        'R': R,
        'alpha_x': alpha_x,
        'alpha_y': alpha_y,
        'beta_x': None,
        'beta_y': None,
        'use_server_ogda': None,
        'use_local_ogda': None,
    }
    results.append(collect_experiment_result('FedGDA', params_gda, traj_gda))

    # params_ogda_local = {
    #     'gamma': GAMMA,
    #     'lambda': LAMBDA,
    #     'x0': x0,
    #     'y0': y0,
    #     'T': T,
    #     'R': R,
    #     'alpha_x': alpha_x,
    #     'alpha_y': alpha_y,
    #     'beta_x': beta_x,
    #     'beta_y': beta_y,
    #     'use_server_ogda': False,
    #     'use_local_ogda': True,
    # }
    # results.append(collect_experiment_result('FedOGDA-local', params_ogda_local, traj_ogda_local))

    params_ogda_double = {
        'gamma': GAMMA,
        'lambda': LAMBDA,
        'x0': x0,
        'y0': y0,
        'T': T,
        'R': R,
        'alpha_x': alpha_x,
        'alpha_y': alpha_y,
        'beta_x': beta_x,
        'beta_y': beta_y,
        'use_server_ogda': True,
        'use_local_ogda': True,
    }
    results.append(collect_experiment_result('FedOGDA-double', params_ogda_double, traj_ogda_double))

    # params_ogda_server_only = {
    #     'gamma': GAMMA,
    #     'lambda': LAMBDA,
    #     'x0': x0,
    #     'y0': y0,
    #     'T': T,
    #     'R': R,
    #     'alpha_x': alpha_x,
    #     'alpha_y': alpha_y,
    #     'beta_x': beta_x,
    #     'beta_y': beta_y,
    #     'use_server_ogda': True,
    #     'use_local_ogda': False,
    # }
    # results.append(collect_experiment_result('FedOGDA-server-only', params_ogda_server_only, traj_ogda_server_only))

    save_experiment_results(results, filename='fedminimax_experiment_results.xlsx')

    plot_phase_trajectories(
        [traj_gda, traj_ogda_double],
        labels=['FedGDA', 'FedOGDA'],
        xlim=(-2, 2), ylim=(-2, 2)
    )
    plot_timeseries(
        [traj_gda, traj_ogda_double],
        labels=['FedGDA', 'FedOGDA'],
    )
    plot_last_iterate_zoom(
        [traj_gda, traj_ogda_double],
        labels=['FedGDA', 'FedOGDA'],
        last_k=60
    )