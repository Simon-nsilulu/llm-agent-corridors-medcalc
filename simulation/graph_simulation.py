"""Численная проверка утверждения 1 и следствий 1–2 на случайных графах (раздел 6 статьи).
Numerical check of Statement 1 and Corollaries 1–2 on random agent graphs (section 6).

Модель / model
  * ациклический граф из n = 3..8 агентов, ребро i -> j (i < j) с вероятностью 0,45;
  * выход агента — вектор размерности 1..2, все масштабы σ = 1, равномерная норма;
  * g_k(u) = A_k u + b_k (аффинная); ширина коридора w_k ~ Exp(среднее 0,5), у 10 % агентов
    коридор не ограничен (w = ∞). Два вида коридоров, на одних и тех же графах и ошибках:
      «фиксированный»: границы не зависят от входа (L_k = 0) — интервал ширины w_k вокруг
        эталонного предложения g_k(u*_k), сдвинутый на U[-0,25; 0,25] (аналог диапазонов правдоподобия);
      «зависящий от входа»: l_k(u) = D_k u + d_k - w_k/2, h_k = l_k + w_k,
        D_k = A_k + U[-0,2; 0,2], d_k = b_k + U[-0,25; 0,25];
  * M_k = ‖A_k‖∞, L_k = ‖D_k‖∞ (максимальная сумма модулей по строке; 0 для фиксированного), K_k = M_k + L_k;
  * в половине графов все коэффициенты A_k, D_k неотрицательны (монотонный случай).

Сценарии ошибок, по 5000 графов (seed 2026):
  S1 грубая ошибка одного случайного агента, e ~ U[-5; 5] по каждой координате;
  S2 независимый шум у всех агентов, N(0; 0,1²);
  S3 общее смещение 0,3 плюс шум N(0; 0,02²);
  S4 шум N(0; 0,1²) у всех и грубая ошибка U[-5; 5] у каждого агента с вероятностью 0,2.

Проверяется / checked
  (3.1) δ_k ≤ min(‖e_k‖, ℓ_k) + K_k Δ_k ;  (3.2) δ_k ≤ max(‖e_k‖ + M_k Δ_k, L_k Δ_k) ;
  (3.3) δ_k ≤ B_k, B_k = c_k + K_k max_{j∈P(k)} B_j ;
  (3.4) ‖x_k − x_k^(−j)‖ ≤ min(‖e_j‖, ℓ_j) Λ_{j→k} для каждого агента j с e_j ≠ 0 и каждого k ⪰ j ;
  монотонность: в монотонных графах при |e| ≤ ε отклонение 50 случайных ошибок не превышает
  наибольшего из отклонений при e = +ε и e = −ε.
Консервативность на стоках: среднее δ, среднее B, средняя оценка без коридоров
(c_k = ‖e_k‖, K_k = M_k) и доля стоков, где оценка с коридором строго меньше.

python simulation/graph_simulation.py   ->  simulation/results.json, simulation/results.txt
"""
import json
import os

import numpy as np

SEED = 2026
N_GRAPHS = 5000
TOL = 1e-9
HERE = os.path.dirname(os.path.abspath(__file__))


def make_graph(rng):
    n = int(rng.integers(3, 9))
    dim = [int(rng.integers(1, 3)) for _ in range(n)]
    preds = [[i for i in range(k) if rng.random() < 0.45] for k in range(n)]
    monotone = rng.random() < 0.5
    agents = []
    for k in range(n):
        din = sum(dim[j] for j in preds[k])
        if monotone:
            A = rng.uniform(0, 1, (dim[k], din))
            D = np.clip(A + rng.uniform(-0.2, 0.2, (dim[k], din)), 0, None)
        else:
            A = rng.uniform(-1, 1, (dim[k], din))
            D = A + rng.uniform(-0.2, 0.2, (dim[k], din))
        b = rng.uniform(-1, 1, dim[k])
        d = b + rng.uniform(-0.25, 0.25, dim[k])
        w = np.full(dim[k], np.inf) if rng.random() < 0.10 else rng.exponential(0.5, dim[k])
        M = np.abs(A).sum(axis=1).max() if din else 0.0
        L = np.abs(D).sum(axis=1).max() if din else 0.0
        agents.append(dict(A=A, D=D, b=b, d=d, w=w, M=M, L=L, K=M + L))
    return dict(n=n, dim=dim, preds=preds, monotone=monotone, agents=agents)


def run(G, errors, mode="input"):
    """Выходы агентов / outputs. mode: "input" (зависящий от входа), "fixed", "none" (без коридоров)."""
    xs, ells = [], []
    for k, ag in enumerate(G["agents"]):
        u = np.concatenate([xs[j] for j in G["preds"][k]]) if G["preds"][k] else np.zeros(0)
        s = ag["A"] @ u + ag["b"] + errors[k]
        if mode == "none":
            xs.append(s)
        else:
            centre = (ag["D"] @ u + ag["d"]) if mode == "input" else ag["fixed_centre"]
            half = ag["w"] / 2
            xs.append(np.minimum(np.maximum(s, centre - half), centre + half))
        ells.append(float(np.max(ag["w"])))
    return xs, ells


def set_fixed_centres(G, rng):
    """Центр фиксированного коридора: эталонное предложение g_k(u*) + сдвиг (L_k = 0)."""
    xs = []
    for k, ag in enumerate(G["agents"]):
        u = np.concatenate([xs[j] for j in G["preds"][k]]) if G["preds"][k] else np.zeros(0)
        g = ag["A"] @ u + ag["b"]
        ag["fixed_centre"] = g + rng.uniform(-0.25, 0.25, g.shape)
        xs.append(np.minimum(np.maximum(g, ag["fixed_centre"] - ag["w"] / 2), ag["fixed_centre"] + ag["w"] / 2))


def draw_errors(G, scen, rng):
    n, dim = G["n"], G["dim"]
    if scen == "S1":
        e = [np.zeros(dim[k]) for k in range(n)]
        j = int(rng.integers(n))
        e[j] = rng.uniform(-5, 5, dim[j])
    elif scen == "S2":
        e = [rng.normal(0, 0.1, dim[k]) for k in range(n)]
    elif scen == "S3":
        e = [0.3 + rng.normal(0, 0.02, dim[k]) for k in range(n)]
    else:
        e = [rng.normal(0, 0.1, dim[k]) + (rng.uniform(-5, 5, dim[k]) if rng.random() < 0.2 else 0)
             for k in range(n)]
    return e


def lambdas(G, K):
    """Λ[j][k] — наибольшее произведение K вдоль пути j -> k без K_j; Λ[j][j] = 1."""
    n = G["n"]
    lam = np.zeros((n, n))
    for j in range(n):
        lam[j, j] = 1.0
        for k in range(j + 1, n):
            cand = [lam[j, p] for p in G["preds"][k] if lam[j, p] > 0]
            if cand:
                lam[j, k] = K[k] * max(cand)
    return lam


MODES = ("fixed", "input")


def main():
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "graphs_per_scenario": N_GRAPHS, "checks": {}, "sinks": {}}
    tot = {m: dict(nodes=0, viol_a=0, viol_b=0, viol_min=0, viol_B=0, attr=0, viol_attr=0, mono=0, viol_mono=0)
           for m in MODES}
    for scen in ("S1", "S2", "S3", "S4"):
        sk = {m: dict(delta=[], B=[], better=[]) for m in MODES}
        sk_nc = dict(delta=[], B=[])
        for _ in range(N_GRAPHS):
            G = make_graph(rng)
            set_fixed_centres(G, rng)
            n, ags = G["n"], G["agents"]
            zero = [np.zeros(G["dim"][k]) for k in range(n)]
            e = draw_errors(G, scen, rng)
            enorm = [float(np.max(np.abs(e[k]))) for k in range(n)]
            eps = [rng.uniform(0, 1, G["dim"][k]) for k in range(n)] if G["monotone"] else None
            mono_err = [[rng.uniform(-1, 1, G["dim"][k]) * eps[k] for k in range(n)] for _r in range(50)] if eps else []
            succ = {j for k in range(n) for j in G["preds"][k]}
            sinks = [k for k in range(n) if k not in succ]
            # без коридоров: K = M, собственная составляющая ‖e‖
            ref_nc, _ = run(G, zero, "none")
            x_nc, _ = run(G, e, "none")
            Bnc = [0.0] * n
            for k in range(n):
                Bnc[k] = enorm[k] + ags[k]["M"] * max((Bnc[j] for j in G["preds"][k]), default=0.0)
            for k in sinks:
                sk_nc["delta"].append(float(np.max(np.abs(x_nc[k] - ref_nc[k]))))
                sk_nc["B"].append(Bnc[k])
            for mode in MODES:
                Lk = [0.0 if mode == "fixed" else ag["L"] for ag in ags]
                K = [ags[k]["M"] + Lk[k] for k in range(n)]
                t = tot[mode]
                ref, _ = run(G, zero, mode)
                x, ell = run(G, e, mode)
                delta = [float(np.max(np.abs(x[k] - ref[k]))) for k in range(n)]
                B = [0.0] * n
                for k in range(n):
                    P = G["preds"][k]
                    Dk = max((delta[j] for j in P), default=0.0)
                    a_ = min(enorm[k], ell[k]) + K[k] * Dk
                    b_ = max(enorm[k] + ags[k]["M"] * Dk, Lk[k] * Dk)
                    B[k] = min(enorm[k], ell[k]) + K[k] * max((B[j] for j in P), default=0.0)
                    t["nodes"] += 1
                    t["viol_a"] += delta[k] > a_ + TOL
                    t["viol_b"] += delta[k] > b_ + TOL
                    t["viol_min"] += delta[k] > min(a_, b_) + TOL
                    t["viol_B"] += delta[k] > B[k] + TOL
                lam = lambdas(G, K)
                for j in range(n):
                    if enorm[j] == 0:
                        continue
                    e2 = list(e)
                    e2[j] = np.zeros(G["dim"][j])
                    x2, _ = run(G, e2, mode)
                    cj = min(enorm[j], ell[j])
                    for k in range(j, n):
                        if lam[j, k] == 0:
                            continue
                        t["attr"] += 1
                        t["viol_attr"] += np.max(np.abs(x[k] - x2[k])) > cj * lam[j, k] + TOL
                if eps:
                    xp, _ = run(G, eps, mode)
                    xm, _ = run(G, [-v for v in eps], mode)
                    worst = [np.maximum(np.abs(xp[k] - ref[k]), np.abs(xm[k] - ref[k])) for k in range(n)]
                    for er in mono_err:
                        xr, _ = run(G, er, mode)
                        for k in range(n):
                            t["mono"] += 1
                            t["viol_mono"] += bool(np.any(np.abs(xr[k] - ref[k]) > worst[k] + TOL))
                for k in sinks:
                    sk[mode]["delta"].append(delta[k])
                    sk[mode]["B"].append(B[k])
                    sk[mode]["better"].append(B[k] < Bnc[k] - TOL)
        out["sinks"][scen] = {"n_sinks": len(sk_nc["B"]),
                              "no_corridor": {"mean_delta": float(np.mean(sk_nc["delta"])), "mean_B": float(np.mean(sk_nc["B"]))}}
        for mode in MODES:
            out["sinks"][scen][mode] = {"mean_delta": float(np.mean(sk[mode]["delta"])),
                                        "mean_B": float(np.mean(sk[mode]["B"])),
                                        "share_tighter_than_no_corridor": float(np.mean(sk[mode]["better"]))}
    out["checks"] = {m: {k: int(v) for k, v in tot[m].items()} for m in MODES}
    json.dump(out, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    L = [f"seed {SEED}, {N_GRAPHS} графов на сценарий, 4 сценария", ""]
    for m in MODES:
        c = out["checks"][m]
        L.append(f"[{m}] (3.1), (3.2), минимум, (3.3): {c['nodes']} проверок; нарушений "
                 f"{c['viol_a']}, {c['viol_b']}, {c['viol_min']}, {c['viol_B']}; "
                 f"(3.4): {c['attr']} проверок, нарушений {c['viol_attr']}; "
                 f"монотонность: {c['mono']} проверок, нарушений {c['viol_mono']}")
    L += ["", "сценарий | стоков | δ без кор. | B без кор. | δ фикс. | B фикс. | уточняет | δ вход | B вход | уточняет"]
    for sc, r in out["sinks"].items():
        f, i_, nc = r["fixed"], r["input"], r["no_corridor"]
        L.append(f"{sc} | {r['n_sinks']} | {nc['mean_delta']:.3f} | {nc['mean_B']:.2f} | {f['mean_delta']:.3f} | "
                 f"{f['mean_B']:.2f} | {100 * f['share_tighter_than_no_corridor']:.1f} % | {i_['mean_delta']:.3f} | "
                 f"{i_['mean_B']:.2f} | {100 * i_['share_tighter_than_no_corridor']:.1f} %")
    txt = "\n".join(L)
    open(os.path.join(HERE, "results.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
