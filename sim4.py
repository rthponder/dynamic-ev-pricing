import pandas as pd
import numpy as np
from types import SimpleNamespace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from network_engine import ChargingNetwork, station_throughput

def build_network(roads_csv, stations_csv, ods_csv):
    roads_df = pd.read_csv(roads_csv)
    stations_df = pd.read_csv(stations_csv)
    ods_df = pd.read_csv(ods_csv)
    defaults = dict(
        l0=0.25, L=2.0, a=1.0,
        mu_s=2.0, a_s=0.5, c_s=0.2, phi0=0.1,
        alpha=0.3, gamma=1.0, eta=0.05
    )

    net = ChargingNetwork(classes=["EV", "NEV"], defaults=defaults)

    for _, r in roads_df.iterrows():
        net.add_road(
            u=str(r["u"]), v=str(r["v"]),
            classes=[c.strip() for c in str(r["classes"]).split(",")],
            l0=float(r["l0"]), L=float(r["L"]), a=float(r["a"])
        )

    for _, s in stations_df.iterrows():
        net.add_station(
            u=str(s["u"]), v=str(s["v"]), name=str(s["station_id"]),
            classes=["EV"], mu_s=float(s["mu_s"]), a_s=float(s["a_s"]),
            c_s=float(s["c_s"]), phi0=float(s["phi0"]),
            psi=float(s.get("initial_psi", 0.5))
        )

    for _, od in ods_df.iterrows():
        net.add_od(
            name=str(od["od_name"]), origin=str(od["origin"]), dest=str(od["dest"]),
            lam_fn=lambda t, val=float(od["total_demand"]): val,
            class_shares={"EV": float(od["ev_share"]), "NEV": float(od["nev_share"])}
        )

    net.build(verbose=True)
    return net


def _state_index_groups(net):
    x_idx = np.array(
        [net.IDX[("xr", e, c)] for (e, c) in net.road_state_keys]
        + [net.IDX[("xs", e)] for e in net.station_edges],
        dtype=int
    )
    y_idx = np.array(
        [net.IDX[("y", pid)] for pid in net.path_state_keys],
        dtype=int
    )
    return x_idx, y_idx


def _trivial_converged_solution(y_start):
    y_start = np.asarray(y_start, dtype=float)
    return SimpleNamespace(
        t=np.array([0.0]),
        y=y_start.reshape(-1, 1),
        success=True,
        t_events=[np.array([0.0])],
        message="already converged at t=0 (||x_dot||+||y_dot|| < tol before integrating)",
    )


def solve_equilibrium(net, psi, y0=None, t_max=75000.0, tol=1e-6, return_traj=False,
                       plot_convergence_prefix=None):
    if y0 is None:
        y0 = net.initial_state()

    x_idx, y_idx = _state_index_groups(net)

    def rhs(t, state):
        return net.dynamics(t, state, psi_override=psi)

    def converged(t, state):
        deriv = rhs(t, state)
        x_dot_norm = np.linalg.norm(deriv[x_idx]) if x_idx.size else 0.0
        y_dot_norm = np.linalg.norm(deriv[y_idx]) if y_idx.size else 0.0
        return (x_dot_norm + y_dot_norm) - tol
    converged.terminal = True
    converged.direction = -1

    def _residual_at(state):
        deriv = rhs(0.0, state)
        x_dot_norm = np.linalg.norm(deriv[x_idx]) if x_idx.size else 0.0
        y_dot_norm = np.linalg.norm(deriv[y_idx]) if y_idx.size else 0.0
        return x_dot_norm + y_dot_norm

    def _solve(y_start):
        r0 = _residual_at(y_start)
        if r0 < tol:
            return _trivial_converged_solution(y_start)
        return solve_ivp(rhs, (0.0, t_max), y_start, method="BDF",
                          rtol=1e-8, atol=1e-10, events=converged, max_step=10.0)

    def _converged_flag(sol):
        return sol.success and len(sol.t_events[0]) > 0

    def _finish(sol):
        return (sol.y[:, -1], sol) if return_traj else sol.y[:, -1]

    try:
        sol = _solve(y0)
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            if not _converged_flag(sol):
                print(f"[solve_equilibrium] status: steady-state event not achieved "
                f"before t_max={t_max:.1f} (final ||x_dot||+||y_dot||).")
            if plot_convergence_prefix is not None:
                plot_convergence_trajectory(net, sol, x_idx, y_idx, psi, tol=tol,
                                             save_prefix=plot_convergence_prefix)
            return _finish(sol)
    except Exception:
        pass

    sol = _solve(net.initial_state())
    if not _converged_flag(sol):
        print(f"[solve_equilibrium] WARNING: fallback solve from initial_state() also "
              f"did not converge before t_max={t_max:.1f}.")
    if plot_convergence_prefix is not None:
        plot_convergence_trajectory(net, sol, x_idx, y_idx, psi, tol=tol,
                                     save_prefix=plot_convergence_prefix)
    return _finish(sol)

def plot_state_trajectory(net, sol, x_idx, y_idx, x_labels=None, y_labels=None, save_prefix=None):
    t = sol.t
    y_all = sol.y

    if x_labels is None:
        x_labels = getattr(net, "x_state_labels", None) or [f"x[{i}]" for i in x_idx]
    if y_labels is None:
        y_labels = getattr(net, "y_state_labels", None) or [f"y[{i}]" for i in y_idx]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax = axes[0]
    if x_idx.size:
        for i, lbl in zip(x_idx, x_labels):
            ax.plot(t, y_all[i], label=lbl)
        ax.set_ylabel("x (state)")
        ax.set_title("x-state trajectories")
        if len(x_idx) <= 12:
            ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    if y_idx.size:
        for i, lbl in zip(y_idx, y_labels):
            ax.plot(t, y_all[i], label=lbl)
        ax.set_ylabel("y (state)")
        ax.set_xlabel("time")
        ax.set_title("y-state trajectories")
        if len(y_idx) <= 12:
            ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()

    if save_prefix:
        fig.savefig(f"{save_prefix}_state_trajectories.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig


def compute_derivative_trajectory(net, sol, x_idx, y_idx, psi):
    t = sol.t
    n_t = len(t)
    xdot = np.zeros((len(x_idx), n_t))
    ydot = np.zeros((len(y_idx), n_t))
    for k in range(n_t):
        deriv = net.dynamics(t[k], sol.y[:, k], psi_override=psi)
        if x_idx.size:
            xdot[:, k] = deriv[x_idx]
        if y_idx.size:
            ydot[:, k] = deriv[y_idx]
    return xdot, ydot


def plot_convergence_trajectory(net, sol, x_idx, y_idx, psi, tol=1e-6,
                                 x_labels=None, y_labels=None, save_prefix=None):
    t = sol.t
    xdot, ydot = compute_derivative_trajectory(net, sol, x_idx, y_idx, psi)

    if x_labels is None:
        x_labels = getattr(net, "x_state_labels", None) or [f"x[{i}]" for i in x_idx]
    if y_labels is None:
        y_labels = getattr(net, "y_state_labels", None) or [f"y[{i}]" for i in y_idx]

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    ax = axes[0]
    if x_idx.size:
        for row, lbl in zip(xdot, x_labels):
            ax.plot(t, row, label=lbl)
        ax.set_ylabel(r"$\dot{x}$")
        ax.set_title("x-state derivatives")
        if len(x_idx) <= 12:
            ax.legend(fontsize=7, ncol=2)
    ax.axhline(0.0, color="k", linewidth=0.6, linestyle=":")

    ax = axes[1]
    if y_idx.size:
        for row, lbl in zip(ydot, y_labels):
            ax.plot(t, row, label=lbl)
        ax.set_ylabel(r"$\dot{y}$")
        ax.set_title("y-state derivatives")
        if len(y_idx) <= 12:
            ax.legend(fontsize=7, ncol=2)
    ax.axhline(0.0, color="k", linewidth=0.6, linestyle=":")

    ax = axes[2]
    x_dot_norm = np.linalg.norm(xdot, axis=0) if x_idx.size else np.zeros_like(t)
    y_dot_norm = np.linalg.norm(ydot, axis=0) if y_idx.size else np.zeros_like(t)
    total_norm = x_dot_norm + y_dot_norm
    ax.plot(t, x_dot_norm, label=r"$\|\dot{x}\|$")
    ax.plot(t, y_dot_norm, label=r"$\|\dot{y}\|$")
    ax.plot(t, total_norm, label=r"$\|\dot{x}\|+\|\dot{y}\|$", color="k", linewidth=1.5)
    ax.axhline(tol, color="red", linestyle="--", linewidth=1, label=f"tol={tol:g}")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("derivative norm (log)")
    ax.set_title("Convergence criterion (event trigger)")
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save_prefix:
        fig.savefig(f"{save_prefix}_convergence_trajectory.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig


def _residual_norm(net, psi, state, x_idx, y_idx, t=0.0):
    deriv = net.dynamics(t, state, psi_override=psi)
    x_dot_norm = np.linalg.norm(deriv[x_idx]) if x_idx.size else 0.0
    y_dot_norm = np.linalg.norm(deriv[y_idx]) if y_idx.size else 0.0
    return x_dot_norm + y_dot_norm


def station_metrics(net, psi, state):
    d = net._unpack(state)
    profit, rho, occ = {}, {}, {}
    for name, e in net.stations.items():
        attrs = net.G.edges[e]
        x = d["xs"][e]
        r = station_throughput(x, attrs["a_s"], attrs["mu_s"])
        rho[name] = r
        occ[name] = x
        profit[name] = (psi[name] - attrs["c_s"]) * r
    return profit, rho, occ


def outer_loop(net, n_steps=5, kappa=0.01, delta=0.02, dt_outer=1.0, grad_clip=None,
               save_prefix=None, plot_every_step=False, plot_perturbation_solves=False,
               t_max=75000.0):
    stations = list(net.stations.keys())
    x_idx, y_idx = _state_index_groups(net)

    psi = {s: 0.5 for s in stations}
    hist = {s: {"psi": [], "rho": [], "occ": [], "profit": []} for s in stations}
    hist["residual"] = []
    state = None

    print("\n--- Running Strategic Pricing Outer Loop ---")
    for step in range(n_steps):
        step_prefix = (f"{save_prefix}_step{step:03d}"
                        if (save_prefix is not None and plot_every_step) else None)
        state = solve_equilibrium(net, psi, y0=state, t_max=t_max,
                                   plot_convergence_prefix=step_prefix)
        profit, rho, occ = station_metrics(net, psi, state)
        hist["residual"].append(_residual_norm(net, psi, state, x_idx, y_idx))
        
        for s in stations:
            hist[s]["psi"].append(psi[s])
            hist[s]["rho"].append(rho[s])
            hist[s]["occ"].append(occ[s])
            hist[s]["profit"].append(profit[s])

        pert_prefix = step_prefix if (step_prefix is not None and plot_perturbation_solves) else None

        grad = {}
        for s in stations:
            psi_p = dict(psi); psi_p[s] = psi[s] + delta
            p_tag = f"{pert_prefix}_{s}_plus" if pert_prefix is not None else None
            state_p = solve_equilibrium(net, psi_p, y0=state, t_max=t_max,
                                         plot_convergence_prefix=p_tag)
            profit_p, _, _ = station_metrics(net, psi_p, state_p)

            psi_m = dict(psi); psi_m[s] = max(psi[s] - delta, 0.0)
            m_tag = f"{pert_prefix}_{s}_minus" if pert_prefix is not None else None
            state_m = solve_equilibrium(net, psi_m, y0=state, t_max=t_max,
                                         plot_convergence_prefix=m_tag)
            profit_m, _, _ = station_metrics(net, psi_m, state_m)

            denom = psi_p[s] - psi_m[s]
            g = (profit_p[s] - profit_m[s]) / denom if denom > 1e-9 else 0.0
            grad[s] = float(np.clip(g, -grad_clip, grad_clip)) if grad_clip is not None else float(g)

        for s in stations:
            psi[s] = float(max(psi[s] + dt_outer * kappa * grad[s], 0.0))

        print(f"  Step {step+1:2d}/{n_steps} | Prices: " + ", ".join(f"{s}=${psi[s]:.3f}" for s in stations))

    state, sol = solve_equilibrium(net, psi, y0=state, t_max=t_max, return_traj=True)
    profit, rho, occ = station_metrics(net, psi, state)
    hist["residual"].append(_residual_norm(net, psi, state, x_idx, y_idx))

    if save_prefix is not None:
        x_idx, y_idx = _state_index_groups(net)
        plot_state_trajectory(net, sol, x_idx, y_idx, save_prefix=f"{save_prefix}_final")
        plot_convergence_trajectory(net, sol, x_idx, y_idx, psi, save_prefix=f"{save_prefix}_final")

    return dict(psi=psi, state=state, profit=profit, rho=rho, occ=occ, hist=hist, final_sol=sol)


def plot_convergence(results, stations, save_prefix="client_run", tol=1e-6):
    fig, axs = plt.subplots(3, 2, figsize=(11, 10))
    metrics = [("psi", r"$\psi_s$ ($/veh)", axs[0, 0], "Station Prices"),
               ("rho", r"$\rho_s$ (veh/time)", axs[0, 1], "Station Throughput"),
               ("occ", r"$x_s^{EV}$ (veh)", axs[1, 0], "EV Occupancy"),
               ("profit", r"$\pi_s$ ($/time)", axs[1, 1], "Station Profit")]
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    for key, ylabel, ax, title in metrics:
        for i, s in enumerate(stations):
            ax.plot(results["hist"][s][key], color=colors[i % len(colors)], label=s, marker='o', markersize=3)
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
    ax = axs[2, 0]
    residuals = results["hist"]["residual"]
    ax.plot(residuals, color="k", marker="o", markersize=3)
    ax.axhline(tol, color="red", linestyle="--", linewidth=1, label=f"tol={tol:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel(r"$\|\dot{x}\|+\|\dot{y}\|$ (log)")
    ax.set_title("Equilibrium Residual (below tol = trustworthy)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    axs[2, 1].axis("off")

    fig.tight_layout()
    fig.savefig(f"{save_prefix}_pricing_convergence.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved pricing convergence plot (incl. residual panel) to {save_prefix}_pricing_convergence.png")


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 4:
        print("Usage: python simulator.py <roads.csv> <stations.csv> <ods.csv>")
        sys.exit(1)
        
    roads_csv = sys.argv[1]
    stations_csv = sys.argv[2]
    ods_csv = sys.argv[3]
    
    output_prefix = os.path.splitext(os.path.basename(roads_csv))[0]
    
    print(f"Loading network from CSVs:\n  Roads: {roads_csv}\n  Stations: {stations_csv}\n  ODs: {ods_csv}")
    
    net = build_network(roads_csv, stations_csv, ods_csv)
    
    results = outer_loop(net, n_steps=50, save_prefix=output_prefix, plot_every_step=True)
    
    plot_convergence(results, list(net.stations.keys()), save_prefix=output_prefix)
    
    print("\n--- Generating Final Traffic Simulation Summary ---")
    final_prices = results["psi"]
    sim_result = net.simulate(t_end=50.0, psi_override=final_prices)
    post_processed = net.post_process(sim_result, psi_override=final_prices)
    
    net.make_all_plots(post_processed, save_prefix=output_prefix)
    net.print_final_summary(post_processed)
    
    print("\nSimulation complete. All client outputs generated.")