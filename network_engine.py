import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.integrate import solve_ivp

def latency(x_total, l0, L):
    ratio = np.clip(x_total / L, 0.0, 0.999999)
    return l0 * ratio / (1.0 - ratio)

def waiting_time(x, Ks, mu_s):
    return max(x - Ks, 0.0) / mu_s

def station_cost(x, psi_s, phi0, alpha, gamma, Ks, mu_s):
    return phi0 + alpha * waiting_time(x, Ks, mu_s) + gamma * psi_s

def station_throughput(x, a_s, mu_s):
    return min(a_s * x, mu_s)

def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return fn(y, x)

class ChargingNetwork:
    def __init__(self, classes, defaults=None):
        self.classes = list(classes)
        self.defaults = dict(
            l0=0.25, L=2.0, a=1.0,
            mu_s=2.0, a_s=0.5, c_s=0.2, phi0=0.1,
            alpha=0.3, gamma=1.0, eta=0.05,
        )
        if defaults:
            self.defaults.update(defaults)

        self.G = nx.MultiDiGraph()
        self.stations = {}
        self.ods = []
        self._built = False

    def add_road(self, u, v, classes=None, l0=None, L=None, a=None, label=None):
        d = self.defaults
        key = self.G.add_edge(
            u, v, kind="road",
            classes=set(classes) if classes else set(self.classes),
            l0=l0 if l0 is not None else d["l0"],
            L=L if L is not None else d["L"],
            a=a if a is not None else d["a"],
            label=label or f"{u}->{v}",
        )
        return (u, v, key)

    def add_station(self, u, v, name, classes=None, mu_s=None, a_s=None,
                     c_s=None, phi0=None, psi=0.0, label=None):
        d = self.defaults
        key = self.G.add_edge(
            u, v, kind="station", name=name,
            classes=set(classes) if classes else {"EV"},
            mu_s=mu_s if mu_s is not None else d["mu_s"],
            a_s=a_s if a_s is not None else d["a_s"],
            c_s=c_s if c_s is not None else d["c_s"],
            phi0=phi0 if phi0 is not None else d["phi0"],
            psi=psi,
            label=label or name,
        )
        e = (u, v, key)
        self.stations[name] = e
        return e

    def add_od(self, name, origin, dest, lam_fn, class_shares):
        self.ods.append(dict(name=name, origin=origin, dest=dest,
                              lam_fn=lam_fn, class_shares=dict(class_shares)))

    def build(self, max_path_length=None, verbose=True):
        G = self.G
        self.path_edges = {}
        self.path_class = {}
        self.path_od = {}
        self.groups = {}

        road_subgraph = nx.DiGraph()
        for u, v, k, attrs in self.G.edges(keys=True, data=True):
            if attrs["kind"] == "road":
                road_subgraph.add_edge(u, v, weight=attrs["l0"], key=k)

        for od in self.ods:
            orig, dest, od_name = od["origin"], od["dest"], od["name"]
            
            try:
                sp_cost = nx.shortest_path_length(road_subgraph, orig, dest, weight="weight")
                max_cost = sp_cost * 2.0
            except nx.NetworkXNoPath:
                continue

            if od["class_shares"].get("NEV", 0) > 0:
                nev_pids = []
                for path in nx.shortest_simple_paths(road_subgraph, orig, dest, weight="weight"):
                    edges = [(path[i], path[i+1], road_subgraph[path[i]][path[i+1]]["key"]) for i in range(len(path)-1)]
                    p_cost = sum(self.G.edges[e]["l0"] for e in edges)
                    
                    if p_cost > max_cost or len(nev_pids) >= 5:
                        break
                    
                    pid = f"{od_name}_NEV_path_{len(nev_pids)+1}"
                    self.path_edges[pid] = edges
                    self.path_class[pid] = "NEV"
                    self.path_od[pid] = od_name
                    nev_pids.append(pid)
                if nev_pids:
                    self.groups[(od_name, "NEV")] = nev_pids

            # EV Paths
            if od["class_shares"].get("EV", 0) > 0:
                ev_pids = []
                for st_name, (st_u, st_v, st_k) in self.stations.items():
                    try:
                        p_in = nx.shortest_path(road_subgraph, orig, st_u, weight="weight")
                        p_out = nx.shortest_path(road_subgraph, st_v, dest, weight="weight")
                        
                        edges_in = [(p_in[i], p_in[i+1], road_subgraph[p_in[i]][p_in[i+1]]["key"]) for i in range(len(p_in)-1)]
                        edges_out = [(p_out[i], p_out[i+1], road_subgraph[p_out[i]][p_out[i+1]]["key"]) for i in range(len(p_out)-1)]
                        st_edge = (st_u, st_v, st_k)
                        
                        total_cost = sum(self.G.edges[e]["l0"] for e in edges_in) + self.G.edges[st_edge]["phi0"] + sum(self.G.edges[e]["l0"] for e in edges_out)
                        
                        if total_cost <= max_cost:
                            pid = f"{od_name}_EV_via_{st_name}"
                            self.path_edges[pid] = edges_in + [st_edge] + edges_out
                            self.path_class[pid] = "EV"
                            self.path_od[pid] = od_name
                            ev_pids.append(pid)
                    except nx.NetworkXNoPath:
                        continue
                if ev_pids:
                    self.groups[(od_name, "EV")] = ev_pids
        
        self.edge_users = defaultdict(list)
        self.edge_injects = defaultdict(list)
        self.transitions = defaultdict(list)
        for pid, edges in self.path_edges.items():
            for e in edges:
                self.edge_users[e].append(pid)
            self.edge_injects[edges[0]].append(pid)
            for i in range(len(edges) - 1):
                self.transitions[edges[i]].append((edges[i + 1], pid))

        road_state_keys = set()
        station_edges = set()
        for pid, edges in self.path_edges.items():
            c = self.path_class[pid]
            for e in edges:
                if G.edges[e]["kind"] == "road":
                    road_state_keys.add((e, c))
                else:
                    station_edges.add(e)
        self.road_state_keys = sorted(road_state_keys, key=lambda k: (k[0], k[1]))
        self.station_edges = sorted(station_edges)
        self.game_groups = {k: v for k, v in self.groups.items() if len(v) > 1}
        self.path_state_keys = [pid for pids in self.game_groups.values() for pid in pids]

        keys = (
            [("xr", e, c) for (e, c) in self.road_state_keys]
            + [("xs", e) for e in self.station_edges]
            + [("y", pid) for pid in self.path_state_keys]
        )
        self.IDX = {k: i for i, k in enumerate(keys)}
        self.N_STATES = len(keys)
        self.STATE_NAMES = [self._name_key(k) for k in keys]
        self._built = True

        if verbose:
            self._print_build_summary()
        return self

    def _name_key(self, k):
        if k[0] == "xr":
            _, e, c = k
            return f"x_{self.G.edges[e]['label']}_{c}"
        if k[0] == "xs":
            _, e = k
            return f"x_{self.G.edges[e]['name']}"
        _, pid = k
        return f"y_{pid}"

    def _print_build_summary(self):
        print(f"[network_engine] {len(self.ods)} OD pair(s), "
              f"{len(self.stations)} station(s), "
              f"{sum(1 for _, _, a in self.G.edges(data=True) if a['kind'] == 'road')} road link(s)")
        for (od, c), pids in self.groups.items():
            tag = "route-choice game" if len(pids) > 1 else "single feasible path"
            print(f"  OD='{od}' class='{c}': {len(pids)} path(s) [{tag}] -> {pids}")
        print(f"  Total ODE states: {self.N_STATES} "
              f"({len(self.road_state_keys)} link, {len(self.station_edges)} station, "
              f"{len(self.path_state_keys)} route-choice)")

    def _unpack(self, state):
        xr = {(e, c): state[self.IDX[("xr", e, c)]] for (e, c) in self.road_state_keys}
        xs = {e: state[self.IDX[("xs", e)]] for e in self.station_edges}
        y = {pid: state[self.IDX[("y", pid)]] for pid in self.path_state_keys}
        return dict(xr=xr, xs=xs, y=y)

    def _get_psi(self, e, t, psi_override):
        name = self.G.edges[e]["name"]
        if psi_override and name in psi_override:
            val = psi_override[name]
        else:
            val = self.G.edges[e]["psi"]
        return val(t) if callable(val) else val

    def initial_state(self):
        state = np.zeros(self.N_STATES)
        for (od_name, c), pids in self.game_groups.items():
            od = next(o for o in self.ods if o["name"] == od_name)
            lam0 = od["class_shares"][c] * od["lam_fn"](0.0)
            for pid in pids:
                state[self.IDX[("y", pid)]] = lam0 / len(pids)
        return state

    def dynamics(self, t, state, psi_override=None):
        if not self._built:
            raise RuntimeError("Call .build() before simulating.")
        G = self.G
        d = self._unpack(state)
        p = self.defaults

        lam = {}
        for od in self.ods:
            L = od["lam_fn"](t)
            for c, share in od["class_shares"].items():
                lam[(od["name"], c)] = share * L

        q = {}
        for key, pids in self.groups.items():
            total_lam = lam[key]
            if len(pids) == 1:
                q[pids[0]] = total_lam
            else:
                ys = np.array([d["y"][pid] for pid in pids])
                tot_y = ys.sum()
                fracs = ys / tot_y if tot_y > 1e-12 else np.full(len(pids), 1.0 / len(pids))
                for pid, fr in zip(pids, fracs):
                    q[pid] = fr * total_lam

        outflow_road = {}
        for (e, c) in self.road_state_keys:
            outflow_road[(e, c)] = G.edges[e]["a"] * d["xr"][(e, c)]
        outflow_station = {}
        for e in self.station_edges:
            a_s, mu_s = G.edges[e]["a_s"], G.edges[e]["mu_s"]
            outflow_station[e] = station_throughput(d["xs"][e], a_s, mu_s)

        arrival_road = defaultdict(float)
        arrival_station = defaultdict(float)
        for e, pids in self.edge_injects.items():
            kind = G.edges[e]["kind"]
            for pid in pids:
                if kind == "road":
                    arrival_road[(e, self.path_class[pid])] += q[pid]
                else:
                    arrival_station[e] += q[pid]

        for e_from, targets in self.transitions.items():
            kind_from = G.edges[e_from]["kind"]
            if kind_from == "road":
                by_class = defaultdict(list)
                for e_to, pid in targets:
                    by_class[self.path_class[pid]].append((e_to, pid))
                for c, lst in by_class.items():
                    denom = sum(q[pid] for pid in self.edge_users[e_from]
                                if self.path_class[pid] == c)
                    outf = outflow_road[(e_from, c)]
                    if denom < 1e-12:
                        continue
                    for e_to, pid in lst:
                        frac = q[pid] / denom
                        if G.edges[e_to]["kind"] == "road":
                            arrival_road[(e_to, c)] += outf * frac
                        else:
                            arrival_station[e_to] += outf * frac
            else:
                denom = sum(q[pid] for pid in self.edge_users[e_from])
                outf = outflow_station[e_from]
                if denom < 1e-12:
                    continue
                for e_to, pid in targets:
                    frac = q[pid] / denom
                    c = self.path_class[pid]
                    if G.edges[e_to]["kind"] == "road":
                        arrival_road[(e_to, c)] += outf * frac
                    else:
                        arrival_station[e_to] += outf * frac

        out = np.zeros(self.N_STATES)
        for (e, c) in self.road_state_keys:
            out[self.IDX[("xr", e, c)]] = arrival_road[(e, c)] - outflow_road[(e, c)]
        for e in self.station_edges:
            out[self.IDX[("xs", e)]] = arrival_station[e] - outflow_station[e]

        if self.game_groups:
            cost = self._path_costs(d, t, psi_override)
            eta = p["eta"]
            for key, pids in self.game_groups.items():
                total_lam = lam[key]
                ys = np.array([d["y"][pid] for pid in pids])
                costs = np.array([cost[pid] for pid in pids])
                # avg = np.dot(ys, costs) / total_lam if total_lam > 1e-12 else 0.0
                S = ys.sum()
                avg = np.dot(ys, costs) / S if S > 1e-12 else 0.0
                for pid, y_val, c_val in zip(pids, ys, costs):
                    out[self.IDX[("y", pid)]] = eta * y_val * (avg - c_val)
        return out

    def _path_costs(self, d, t, psi_override):
        G, p = self.G, self.defaults
        cost = {}
        for pid, edges in self.path_edges.items():
            total = 0.0
            for e in edges:
                attrs = G.edges[e]
                if attrs["kind"] == "road":
                    xtot = sum(d["xr"].get((e, cc), 0.0) for cc in self.classes)
                    total += latency(xtot, attrs["l0"], attrs["L"])
                else:
                    psi_val = self._get_psi(e, t, psi_override)
                    Ks = attrs["mu_s"] / attrs["a_s"]
                    total += station_cost(d["xs"][e], psi_val, attrs["phi0"],
                                           p["alpha"], p["gamma"], Ks, attrs["mu_s"])
            cost[pid] = total
        return cost

    def simulate(self, t_end, pts_per_unit=4.0, psi_override=None,
                 method="BDF", rtol=1e-9, atol=1e-11, y0=None):
        if not self._built:
            self.build()
        state0 = self.initial_state() if y0 is None else np.asarray(y0, dtype=float)
        n_eval = max(int(t_end * pts_per_unit), 10)
        t_eval = np.linspace(0.0, t_end, n_eval)
        sol = solve_ivp(
            self.dynamics, (0.0, t_end), state0, args=(psi_override,),
            method=method, t_eval=t_eval, rtol=rtol, atol=atol, dense_output=False,
        )
        if not sol.success:
            print(f"[network_engine] WARNING: solver stopped early at t={sol.t[-1]:.3f} "
                  f"(requested t_end={t_end}): {sol.message}. "
                  f"Results after this point are NOT available.")
        return dict(t=sol.t, y=sol.y, psi_override=psi_override, success=sol.success)


    def _state_floor(self):
        floor = getattr(self, "_floor", None)
        if floor is None or len(floor) != self.N_STATES:
            floor = np.zeros(self.N_STATES)
            for pid in self.path_state_keys:
                floor[self.IDX[("y", pid)]] = 1e-12
            self._floor = floor
        return floor

    def advance(self, state, t0, psi_override, h, n_steps=1, return_samples=False):

        if not self._built:
            raise RuntimeError("Call .build() before simulating.")
        n_steps = int(n_steps)
        s0 = np.asarray(state, dtype=float)
        t1 = t0 + n_steps * h
        t_eval = t0 + h * np.arange(1, n_steps + 1)
        t_eval[-1] = t1
        sol = solve_ivp(self.dynamics, (t0, t1), s0, args=(psi_override,), method="BDF",
                        t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=min(h, 10.0))
        if (not sol.success) or sol.y.shape[1] != n_steps or not np.all(np.isfinite(sol.y)):
            raise FloatingPointError(
                f"ODE window [{t0:.2f}, {t1:.2f}] failed: {sol.message}")
        samples = np.maximum(sol.y, self._state_floor()[:, None])
        final = samples[:, -1].copy()
        return (final, samples) if return_samples else final

    def simulate_discrete(self, t_end, h=10.0, psi_override=None, y0=None, t0=0.0):
        if not self._built:
            self.build()
        state = self.initial_state() if y0 is None else np.asarray(y0, dtype=float)
        n = int(np.ceil(t_end / h - 1e-9))
        T = t0 + h * np.arange(n + 1)
        Y = np.empty((self.N_STATES, n + 1))
        Y[:, 0] = state
        success = True
        for k in range(n):
            try:
                state = self.advance(state, T[k], psi_override, h, 1)
            except FloatingPointError as err:
                print(f"[network_engine] WARNING: {err} Results truncated at t={T[k]:.3f}.")
                T, Y, success = T[:k + 1], Y[:, :k + 1], False
                break
            Y[:, k + 1] = state
        return dict(t=T, y=Y, psi_override=psi_override, success=success)

    def post_process(self, res, psi_override=None):
        psi_override = psi_override if psi_override is not None else res.get("psi_override")
        T, Y = res["t"], res["y"]
        out = {"t": T}

        def row(key):
            return Y[self.IDX[key], :]

        combined_edge = defaultdict(lambda: np.zeros_like(T))
        for (e, c) in self.road_state_keys:
            lbl = self.G.edges[e]["label"]
            series = row(("xr", e, c))
            out[f"x_{lbl}_{c}"] = series
            combined_edge[lbl] += series
        for lbl, series in combined_edge.items():
            out[f"x_{lbl}"] = series

        lam_od = {}
        for od in self.ods:
            lam_od[od["name"]] = np.array([od["lam_fn"](tt) for tt in T])
        out["lam_od"] = lam_od

        for name, e in self.stations.items():
            occ = row(("xs", e))
            attrs = self.G.edges[e]
            rho = np.array([station_throughput(v, attrs["a_s"], attrs["mu_s"]) for v in occ])
            psi_series = np.array([self._get_psi(e, tt, psi_override) for tt in T])
            Ks = attrs["mu_s"] / attrs["a_s"]
            out[f"occ_{name}"] = occ
            out[f"rho_{name}"] = rho
            out[f"psi_{name}"] = psi_series
            out[f"profit_{name}"] = (psi_series - attrs["c_s"]) * rho
            out[f"revenue_{name}"] = psi_series * rho
            out[f"queue_{name}"] = np.maximum(occ - Ks, 0.0)

        cost_series = {pid: np.zeros_like(T) for pid in self.path_edges}
        flow_series = {pid: np.zeros_like(T) for pid in self.path_edges}
        for k in range(len(T)):
            state_k = Y[:, k]
            d = self._unpack(state_k)
            lam_k = {}
            for od in self.ods:
                Lval = od["lam_fn"](T[k])
                for c, share in od["class_shares"].items():
                    lam_k[(od["name"], c)] = share * Lval
            cost_k = self._path_costs(d, T[k], psi_override)
            for key, pids in self.groups.items():
                total_lam = lam_k[key]
                if len(pids) == 1:
                    flow_series[pids[0]][k] = total_lam
                else:
                    ys = np.array([d["y"][pid] for pid in pids])
                    tot_y = ys.sum()
                    fracs = ys / tot_y if tot_y > 1e-12 else np.full(len(pids), 1.0 / len(pids))
                    for pid, fr in zip(pids, fracs):
                        flow_series[pid][k] = fr * total_lam
            for pid, cval in cost_k.items():
                cost_series[pid][k] = cval
        for pid in self.path_edges:
            out[f"y_{pid}"] = flow_series[pid]
            out[f"cost_{pid}"] = cost_series[pid]

        out["total_profit"] = sum(out[f"profit_{s}"] for s in self.stations) \
            if self.stations else np.zeros_like(T)
        out["total_user_cost"] = sum(
            out[f"y_{pid}"] * out[f"cost_{pid}"] for pid in self.path_edges
        )
        return out

    def print_final_summary(self, res_pp):
        T = res_pp["t"]
        print("\n=== Final-time summary ===")
        for name in self.stations:
            print(f"  {name}: occupancy={res_pp[f'occ_{name}'][-1]:.3f}  "
                  f"throughput={res_pp[f'rho_{name}'][-1]:.3f}  "
                  f"profit={res_pp[f'profit_{name}'][-1]:.3f}")
        for (od, c), pids in self.groups.items():
            shares = ", ".join(f"{pid}={res_pp[f'y_{pid}'][-1]:.3f}" for pid in pids)
            print(f"  {od}/{c}: {shares}")
        print(f"  Total user cost (final): {res_pp['total_user_cost'][-1]:.3f}")
        print(f"  Total station profit (final): {res_pp['total_profit'][-1]:.3f}")