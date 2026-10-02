import pandas as pd
import numpy as np
from types import SimpleNamespace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor

from network_engine import ChargingNetwork, station_throughput


def _mk(series):
    return dict(marker='o', markersize=3) if len(series) <= 100 else {}


class ConstantDemand:
    def __init__(self, val):
        self.val = val

    def __call__(self, t):
        return self.val


class StepDemand:

    def __init__(self, base_val, switches):
        switches = sorted(switches, key=lambda sw: sw[0])
        self.times = [0.0] + [float(t) for t, _ in switches]
        self.values = [float(base_val)] + [float(v) for _, v in switches]

    def __call__(self, t):
        idx = 0
        for i, st in enumerate(self.times):
            if t >= st:
                idx = i
            else:
                break
        return self.values[idx]

    @classmethod
    def from_spec(cls, base_val, spec):

        switches = []
        for chunk in str(spec).split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            t_str, v_str = chunk.split(":")
            switches.append((float(t_str), float(v_str)))
        return cls(base_val, switches)


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
        base_demand = float(od["total_demand"])
        schedule = od.get("demand_schedule", "")
        if pd.isna(schedule) or str(schedule).strip() == "":
            lam_fn = ConstantDemand(base_demand)
        else:
            lam_fn = StepDemand.from_spec(base_demand, schedule)

        net.add_od(
            name=str(od["od_name"]), origin=str(od["origin"]), dest=str(od["dest"]),
            lam_fn=lam_fn,
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
                       plot_convergence_prefix=None, t0=0.0):
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
        deriv = rhs(t0, state)
        x_dot_norm = np.linalg.norm(deriv[x_idx]) if x_idx.size else 0.0
        y_dot_norm = np.linalg.norm(deriv[y_idx]) if y_idx.size else 0.0
        return x_dot_norm + y_dot_norm

    def _solve(y_start):
        r0 = _residual_at(y_start)
        if r0 < tol:
            return _trivial_converged_solution(y_start)
        return solve_ivp(rhs, (t0, t0 + t_max), y_start, method="BDF",
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
    psi_is_series = isinstance(psi, (list, tuple))
    for k in range(n_t):
        psi_k = psi[k] if psi_is_series else psi
        deriv = net.dynamics(t[k], sol.y[:, k], psi_override=psi_k)
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


def _station_queue(net, name, occ_val):
    e = net.stations[name]
    attrs = net.G.edges[e]
    Ks = attrs["mu_s"] / attrs["a_s"]
    return max(occ_val - Ks, 0.0)


def _wardrop_spreads(net, psi, state, y_tol=1e-6):

    d = net._unpack(state)
    cost = net._path_costs(d, 0.0, psi)
    spreads = {}
    for key, pids in net.game_groups.items():
        used = [pid for pid in pids if d["y"][pid] > y_tol]
        if len(used) < 2:
            spreads[key] = 0.0
            continue
        costs = [cost[pid] for pid in used]
        spreads[key] = max(costs) - min(costs)
    return spreads


_WORKER_NET = None


def _init_worker(net):
    global _WORKER_NET
    _WORKER_NET = net


def _station_profit_series(net, station, psi_val, states):
    e = net.stations[station]
    attrs = net.G.edges[e]
    occ = states[net.IDX[("xs", e)]]
    rho = np.minimum(attrs["a_s"] * occ, attrs["mu_s"])
    return (psi_val - attrs["c_s"]) * rho


def _perturbation_job(psi, station, sign, delta, state, t_now, horizon, h,
                      profit_mode, eq_t_max):
    net = _WORKER_NET
    psi_pert = dict(psi)
    if sign > 0:
        psi_pert[station] = psi[station] + delta
    else:
        psi_pert[station] = max(psi[station] - delta, 0.0)

    if horizon == "eq":
        state_pert = solve_equilibrium(net, psi_pert, y0=state, t_max=eq_t_max, t0=t_now)
        profit_pert, _, _ = station_metrics(net, psi_pert, state_pert)
        profit = profit_pert[station]
    else:
        _, samples = net.advance(state, t_now, psi_pert, h, horizon,
                                 return_samples=True)
        series = _station_profit_series(net, station, psi_pert[station], samples)
        profit = float(series[-1]) if profit_mode == "end" else float(series.mean())
    return station, sign, psi_pert[station], profit


def outer_loop(net, t_end=75000.0, h=10.0, ctrl_horizon=10, kappa=0.1, delta=0.02,
               dt_outer=1.0, grad_clip=None, save_prefix=None,
               profit_mode="mean", eq_t_max=75000.0, n_workers=None, parallel=None,
               psi0=0.5, log_every=None):
    stations = list(net.stations.keys())
    x_idx, y_idx = _state_index_groups(net)

    horizon = "eq" if ctrl_horizon == "eq" else int(ctrl_horizon)
    if horizon != "eq" and horizon < 1:
        raise ValueError("ctrl_horizon must be a positive integer or 'eq'")
    n_steps = int(round(t_end / h))
    if log_every is None:
        log_every = max(n_steps // 50, 1)

    psi = {s: float(psi0) for s in stations}
    hist = {s: {"psi": [], "rho": [], "occ": [], "profit": [], "revenue": [], "queue": []}
            for s in stations}
    hist["total_revenue"] = []
    hist["wardrop"] = {key: [] for key in net.game_groups}
    hist["path_y"] = {pid: [] for pids in net.game_groups.values() for pid in pids}

    state = net.initial_state()
    t_now = 0.0
    times, traj, psi_hist = [t_now], [state.copy()], [dict(psi)]

    use_pool = (horizon == "eq") if parallel is None else bool(parallel)
    workers = n_workers or min(2 * max(len(stations), 1), os.cpu_count() or 1)
    _init_worker(net)
    pool = (ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(net,))
            if use_pool else None)

    ctrl_desc = "equilibrium" if horizon == "eq" else f"{horizon} step(s) = {horizon * h:g} time units"
    print(f"\n--- Running Discrete-Time Strategic Pricing Loop (h={h:g}, t_end={t_end:g}, "
          f"{n_steps} steps, integrator=ode, look-ahead={ctrl_desc}, profit={profit_mode}) ---")
    try:
        for step in range(n_steps):
            args = [(psi, s, sign, delta, state, t_now, horizon, h,
                     profit_mode, eq_t_max)
                    for s in stations for sign in (+1, -1)]
            if pool is not None:
                outs = [f.result() for f in [pool.submit(_perturbation_job, *a) for a in args]]
            else:
                outs = [_perturbation_job(*a) for a in args]
            profit_at = {(s, sign): (psi_val, profit_val) for s, sign, psi_val, profit_val in outs}

            grad = {}
            for s in stations:
                psi_p_val, profit_p_val = profit_at[(s, +1)]
                psi_m_val, profit_m_val = profit_at[(s, -1)]
                denom = psi_p_val - psi_m_val
                g = (profit_p_val - profit_m_val) / denom if denom > 1e-9 else 0.0
                grad[s] = float(np.clip(g, -grad_clip, grad_clip)) if grad_clip is not None else float(g)
            for s in stations:
                psi[s] = float(max(psi[s] + dt_outer * kappa * grad[s], 0.0))

            state = net.advance(state, t_now, psi, h, 1)
            t_now += h

            times.append(t_now)
            traj.append(state.copy())
            psi_hist.append(dict(psi))

            profit, rho, occ = station_metrics(net, psi, state)
            revenue = {s: psi[s] * rho[s] for s in stations}
            wardrop = _wardrop_spreads(net, psi, state)
            d = net._unpack(state)
            for pid in hist["path_y"]:
                hist["path_y"][pid].append(d["y"][pid])
            for s in stations:
                hist[s]["psi"].append(psi[s])
                hist[s]["rho"].append(rho[s])
                hist[s]["occ"].append(occ[s])
                hist[s]["profit"].append(profit[s])
                hist[s]["revenue"].append(revenue[s])
                hist[s]["queue"].append(_station_queue(net, s, occ[s]))
            hist["total_revenue"].append(sum(revenue[s] for s in stations))
            for key in hist["wardrop"]:
                hist["wardrop"][key].append(wardrop[key])

            if (step + 1) % log_every == 0 or step == n_steps - 1:
                print(f"  Step {step+1:5d}/{n_steps} (t={t_now:9.1f}) | Prices: "
                      + ", ".join(f"{s}=${psi[s]:.3f}" for s in stations))
    finally:
        if pool is not None:
            pool.shutdown()

    sol = SimpleNamespace(t=np.array(times), y=np.array(traj).T, psi_hist=psi_hist,
                          success=True, t_events=[np.array([])])
    profit, rho, occ = station_metrics(net, psi, state)

    if save_prefix is not None:
        plot_state_trajectory(net, sol, x_idx, y_idx, save_prefix=f"{save_prefix}_final")
        plot_convergence_trajectory(net, sol, x_idx, y_idx, psi_hist, save_prefix=f"{save_prefix}_final")

    return dict(psi=psi, state=state, profit=profit, rho=rho, occ=occ, hist=hist,
                final_sol=sol, t_end=t_now)



def plot_convergence(results, stations, net=None, save_prefix="client_run", demand_t_end=None):

    metrics = [("psi", r"$\psi_s$ ($/veh)", "Station Prices", "prices"),
               ("rho", r"$\rho_s$ (veh/time)", "Station Throughput", "throughput"),
               ("occ", r"$x_s^{EV}$ (veh)", "EV Occupancy", "occupancy"),
               ("profit", r"$\pi_s$ ($/time)", "Station Profit", "profit"),
               ("revenue", r"$\psi_s \cdot \rho_s$ ($/time)", "Station Revenue", "revenue"),
               ("queue", r"$\max(x_s-K_s,0)$ (veh)", "Station Queue Length", "queue")]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for key, ylabel, title, fname in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, s in enumerate(stations):
            ax.plot(results["hist"][s][key], color=colors[i % len(colors)], label=s, **_mk(results["hist"][s][key]))
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(f"{save_prefix}_{fname}.png", dpi=150)
        plt.close(fig)
        print(f"Saved {title} plot to {save_prefix}_{fname}.png")

    if net is not None and getattr(net, "ods", None):
        if demand_t_end is None:
            switch_times = []
            for od in net.ods:
                times = getattr(od["lam_fn"], "times", None)
                if times:
                    switch_times.extend(times)
            demand_t_end = max(switch_times) * 1.5 if switch_times else 50.0

        fig, ax = plt.subplots(figsize=(7, 5))
        t_grid = np.linspace(0.0, demand_t_end, 300)
        for i, od in enumerate(net.ods):
            lam_vals = [od["lam_fn"](tt) for tt in t_grid]
            ax.step(t_grid, lam_vals, where="post", color=colors[i % len(colors)], label=od["name"])
        ax.set_xlabel("time t")
        ax.set_ylabel(r"$\lambda_{OD}(t)$ (veh/time)")
        ax.set_title("OD Inflow Rate Schedule")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(f"{save_prefix}_od_inflow.png", dpi=150)
        plt.close(fig)
        print(f"Saved OD Inflow Rate Schedule plot to {save_prefix}_od_inflow.png")


def plot_convergence_summary(results, stations, net=None, save_prefix="client_run", demand_t_end=None):

    fig, axs = plt.subplots(4, 2, figsize=(11, 13))
    metrics = [("psi", r"$\psi_s$ ($/veh)", axs[0, 0], "Station Prices"),
               ("rho", r"$\rho_s$ (veh/time)", axs[0, 1], "Station Throughput"),
               ("occ", r"$x_s^{EV}$ (veh)", axs[1, 0], "EV Occupancy"),
               ("profit", r"$\pi_s$ ($/time)", axs[1, 1], "Station Profit"),
               ("revenue", r"$\psi_s \cdot \rho_s$ ($/time)", axs[2, 0], "Station Revenue"),
               ("queue", r"$\max(x_s-K_s,0)$ (veh)", axs[2, 1], "Station Queue Length")]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for key, ylabel, ax, title in metrics:
        for i, s in enumerate(stations):
            ax.plot(results["hist"][s][key], color=colors[i % len(colors)], label=s, **_mk(results["hist"][s][key]))
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    ax = axs[3, 0]
    if net is not None and getattr(net, "ods", None):
        if demand_t_end is None:
            switch_times = []
            for od in net.ods:
                times = getattr(od["lam_fn"], "times", None)
                if times:
                    switch_times.extend(times)
            demand_t_end = max(switch_times) * 1.5 if switch_times else 50.0

        t_grid = np.linspace(0.0, demand_t_end, 300)
        for i, od in enumerate(net.ods):
            lam_vals = [od["lam_fn"](tt) for tt in t_grid]
            ax.step(t_grid, lam_vals, where="post", color=colors[i % len(colors)], label=od["name"])
        ax.set_xlabel("time t")
        ax.set_ylabel(r"$\lambda_{OD}(t)$ (veh/time)")
        ax.set_title("OD Inflow Rate Schedule")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
    else:
        ax.axis("off")

    ax = axs[3, 1]
    wardrop = results["hist"]["wardrop"]
    for i, (key, spreads) in enumerate(wardrop.items()):
        od_name, cls = key
        ax.plot(spreads, color=colors[i % len(colors)], label=f"{od_name}_{cls}", **_mk(spreads))
    ax.set_yscale("log")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel(r"$\max_{\mathrm{pid}}\mathrm{cost} - \min_{\mathrm{pid}}\mathrm{cost}$")
    ax.set_title("Wardrop Convergence (path-cost spread per OD/class)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}_pricing_convergence_summary.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved combined all-in-one convergence plot to {save_prefix}_pricing_convergence_summary.png")


def plot_wardrop_grid(results, save_prefix="client_run", n_cols=2):

    wardrop = results["hist"]["wardrop"]
    keys = list(wardrop.keys())
    n = len(keys)
    if n == 0:
        print("\nNo game_groups with multiple paths -- nothing to plot for Wardrop convergence.")
        return

    n_rows = int(np.ceil(n / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, key in enumerate(keys):
        r, c = divmod(i, n_cols)
        ax = axs[r][c]
        od_name, cls = key
        spreads = wardrop[key]
        ax.plot(spreads, color=colors[i % len(colors)], **_mk(spreads))
        ax.set_yscale("log")
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel(r"$\max_{\mathrm{pid}}\mathrm{cost} - \min_{\mathrm{pid}}\mathrm{cost}$")
        ax.set_title(f"{od_name}_{cls}")
        ax.grid(True, linestyle=":", alpha=0.6)

    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axs[r][c].axis("off")

    fig.suptitle("Wardrop Convergence (path-cost spread per OD/class)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{save_prefix}_wardrop_grid.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved per-OD/class Wardrop convergence grid to {save_prefix}_wardrop_grid.png")


def plot_path_flow_grid(results, net, save_prefix="client_run", n_cols=2):

    path_y = results["hist"]["path_y"]
    keys = list(net.game_groups.keys())
    n = len(keys)
    if n == 0:
        print("\nNo game_groups with multiple paths -- nothing to plot for path flows.")
        return

    n_rows = int(np.ceil(n / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, key in enumerate(keys):
        r, c = divmod(i, n_cols)
        ax = axs[r][c]
        od_name, cls = key
        pids = net.game_groups[key]
        for j, pid in enumerate(pids):
            ax.plot(path_y[pid], color=colors[j % len(colors)], label=pid, **_mk(path_y[pid]))
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel(r"$y_{\mathrm{pid}}$ (flow share)")
        ax.set_title(f"{od_name}_{cls}")
        ax.legend(fontsize=6)
        ax.grid(True, linestyle=":", alpha=0.6)

    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axs[r][c].axis("off")

    fig.suptitle("Path Flow Shares per OD/class", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{save_prefix}_path_flow_grid.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved per-OD/class path flow grid to {save_prefix}_path_flow_grid.png")


def plot_final_revenue(results, stations, save_prefix="client_run"):
    revenue = {s: results["psi"][s] * results["rho"][s] for s in stations}
    total = sum(revenue.values())

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(stations)), 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bars = ax.bar(stations, [revenue[s] for s in stations],
                   color=[colors[i % len(colors)] for i in range(len(stations))])
    for b, s in zip(bars, stations):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"${revenue[s]:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_ylabel(r"$\psi_s \cdot \rho_s$ ($/time)")
    ax.set_title(f"Total Revenue per Station (final)  |  Sum = ${total:.2f}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}_final_revenue_per_station.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved final revenue-per-station plot to {save_prefix}_final_revenue_per_station.png")
    return revenue, total


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Discrete-time strategic pricing simulation")
    parser.add_argument("roads_csv")
    parser.add_argument("stations_csv")
    parser.add_argument("ods_csv")
    parser.add_argument("--t-end", type=float, default=75000.0, help="total simulated time (Tmax)")
    parser.add_argument("--h", type=float, default=1500.0, help="length of one discrete step")
    parser.add_argument("--horizon", default="10",
                        help="look-ahead in steps of h (e.g. 10, 1) or 'eq'")
    parser.add_argument("--profit-mode", choices=["mean", "end"], default="end")
    parser.add_argument("--kappa", type=float, default=0.1, help="price gradient gain")
    parser.add_argument("--dt-outer", type=float, default=1.0, help="price step multiplier")
    parser.add_argument("--delta", type=float, default=0.02, help="price perturbation")
    parser.add_argument("--final-t", type=float, default=500.0, help="length of the final summary run")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=None,
                    help="run the station what-ifs in a process pool "
                         "(default: on only for --horizon eq)")
    parser.add_argument("--workers", type=int, default=None,
                    help="number of worker processes (default: min(2*stations, CPU count))")
    args = parser.parse_args()

    horizon = "eq" if args.horizon == "eq" else int(args.horizon)
    output_prefix = os.path.splitext(os.path.basename(args.roads_csv))[0]

    run_start = time.perf_counter()

    print(f"Loading network from CSVs:\n  Roads: {args.roads_csv}\n  Stations: {args.stations_csv}\n  ODs: {args.ods_csv}")

    net = build_network(args.roads_csv, args.stations_csv, args.ods_csv)

    results = outer_loop(net, t_end=args.t_end, h=args.h, ctrl_horizon=horizon,
                     kappa=args.kappa, delta=args.delta, dt_outer=args.dt_outer,
                     profit_mode=args.profit_mode,
                     parallel=args.parallel, n_workers=args.workers,
                     save_prefix=output_prefix)

    stations = list(net.stations.keys())
    plot_convergence(results, stations, net=net, save_prefix=output_prefix,
                     demand_t_end=results["t_end"])
    plot_convergence_summary(results, stations, net=net, save_prefix=output_prefix,
                             demand_t_end=results["t_end"])
    plot_wardrop_grid(results, save_prefix=output_prefix)
    plot_path_flow_grid(results, net, save_prefix=output_prefix)
    plot_final_revenue(results, stations, save_prefix=output_prefix)

    print("\n--- Generating Final Traffic Simulation Summary (discrete time) ---")
    final_prices = results["psi"]
    sim_result = net.simulate_discrete(t_end=args.final_t, h=args.h, psi_override=final_prices,
                                       y0=results["state"], t0=results["t_end"])
    post_processed = net.post_process(sim_result, psi_override=final_prices)

    net.print_final_summary(post_processed)

    print("\n--- Final Network State at End of Loop ---")
    for name in net.stations:
        print(f"  {name}: occupancy={results['occ'][name]:.3f}  "
              f"throughput={results['rho'][name]:.3f}  "
              f"profit={results['profit'][name]:.3f}")

    elapsed = time.perf_counter() - run_start
    mins, secs = divmod(elapsed, 60)
    print(f"\nTotal simulation wall time: {elapsed:.2f}s ({int(mins)}m {secs:.1f}s)")