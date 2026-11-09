#!/usr/bin/env python3
"""Roblox Top Trending forecast — make.fun roblox-trending-forecast-01.

Predicts the top 10 of Roblox's Top Trending chart for every hour of the
24-hour window after the deadline. One forecast serves all 24 hourly slots,
so the honest forecast is the average of the per-hour rank distributions.

Method (calibrated on the public chart archive, 30 Oct - 9 Nov 2026):
  * Per-game rank distribution diffused over horizons h = 1..24h
    (Gaussian, sigma growing ~ sqrt(h); the top of the chart is stickier,
    so ranks 1-2 diffuse much slower than the rest).
  * Top-10 exit / rank 11-30 entry mass transfer, rank-dependent, scaled
    from 24h rates to each horizon.
  * #1 reign-age exit risk: a long reign at #1 is fragile (DONT SPLASH
    vanished after ~6.5 days at #1); a dethroned #1 usually falls to
    ranks 2-5 (70%) or leaves the chart (30%).
  * Momentum is NOT used: backtests showed rank momentum hurts
    (mean-reversion dominates this chart).
  * Playing counts, votes, visits and favorites are NOT used: they do
    not predict rank changes (corr ~ -0.02).
  * Per place, named probabilities sum to 0.95; 0.05 is left unnamed to
    cover off-chart entrants, and places 5-10 are slightly flattened
    toward uniform (churn is high there).

Inputs: $MAKEFUN_INPUT_DIR/history.ndjson + task.json (window read from
task.json, never hardcoded). Output: $MAKEFUN_OUTPUT_DIR/forecast.json.
Python 3.13, standard library only, deterministic, no network.
"""
import json
import math
import os
from datetime import datetime

# ---------------------------------------------------------------- parameters
# Tuned by backtest over 29 archive cutoffs (mean log score -18.87 vs the
# house baseline's -24.43; practice window -29.68 vs baseline -32.68).
PARAMS = {
    "VEL_WINDOW_H": 48.0,     # lookback (h) for velocity estimation
    "VEL_HALF_LIFE_H": 24.0,  # exponential recency half-life for velocity fit
    "SHRINK": 0.0,            # momentum shrink (0 = no momentum; tuned: hurts)
    "DRIFT_TAU_H": 12.0,      # momentum saturation timescale (h)
    "DRIFT_MAX": 3.0,         # max total drift (ranks) over the window
    "SIGMA_H": 0.40,          # diffusion: ranks per sqrt(hour)
    "SIGMA_CAP": 2.0,         # max sigma (ranks)
    "SIGMA_TOP_MULT": 0.3,    # sigma multiplier for ranks 1-2 (sticky top)
    "SIGMA_MID_MULT": 0.8,    # sigma multiplier for ranks 3-4
    "SIGMA_MIN_TOP": 0.3,     # sigma floor for ranks 1-2
    "SIGMA_MIN": 0.6,         # sigma floor otherwise
    "EXIT_BASE_24H": 0.05,    # 24h P(leave top 10) at rank 1
    "EXIT_SLOPE_24H": 0.025,  # + per rank (rank 10 -> 0.275)
    "RANK1_EXIT_24H": 0.10,   # 24h P(#1 loses #1); reign-age adjusted below
    "ENTRY_TOTAL_11_20": 1.44,  # expected top-10 entrants/24h from ranks 11-20
    "ENTRY_TOTAL_21_30": 0.12,  # expected top-10 entrants/24h from ranks 21-30
    "ALPHA_TOP": 0.0,         # flat-mix weight, places 1-4
    "ALPHA_LOW": 0.05,        # flat-mix weight, places 5-10
    "UNNAMED": 0.05,          # per-place share left unnamed
    "CANDIDATE_TOP_N": 40,    # games considered, by adjusted last rank
    "MAXNAMED": 30,           # max named games per place in the output
    "MIN_P": 0.001,           # drop output rows below this p
    "RANK_GRID": 45,          # ranks modelled explicitly (1..45); rest = out
}
RANK_GRID = PARAMS["RANK_GRID"]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda r: r["at"])
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    return reads, task


def norm_pdf(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def velocity(times, ranks, t0, window_h, half_life_h):
    """Recency-weighted linear fit of rank vs time. Returns ranks/hour
    (negative = improving). None if fewer than 2 points in the window.
    Dormant while SHRINK == 0, kept for the tuned parameter set."""
    pts = [
        (t, r)
        for t, r in zip(times, ranks)
        if 0 <= (t0 - t).total_seconds() / 3600 <= window_h
    ]
    if len(pts) < 2:
        return None
    w = [
        0.5 ** (((t0 - t).total_seconds() / 3600) / half_life_h) for t, _ in pts
    ]
    x = [-(t0 - t).total_seconds() / 3600 for t, _ in pts]  # hours ago (<= 0)
    y = [r for _, r in pts]
    sw = sum(w)
    mx = sum(wi * xi for wi, xi in zip(w, x)) / sw
    my = sum(wi * yi for wi, yi in zip(w, y)) / sw
    den = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, x))
    if den <= 0:
        return None
    return sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y)) / den


def build_forecast(reads, task):
    P = PARAMS
    t_end = parse_time(task["window"]["from"])
    last = reads[-1]
    last_chart = sorted(last["top_trending"], key=lambda g: g["rank"])
    last_rank = {g["universe_id"]: g["rank"] for g in last_chart}

    # Per-game rank history (time, rank), oldest first.
    hist = {}
    for r in reads:
        t = parse_time(r["at"])
        for g in r["top_trending"]:
            hist.setdefault(g["universe_id"], []).append((t, g["rank"]))
    for u in hist:
        hist[u].sort()

    # Reign age of the #1 game: hours it has held #1 consecutively, measured
    # back from the window start. A fresh #1 is safer than a stale one.
    top1 = last_chart[0]["universe_id"]
    reign_h = 0.0
    for r in reversed(reads):
        first = sorted(r["top_trending"], key=lambda g: g["rank"])[0]
        if first["universe_id"] == top1:
            reign_h = (t_end - parse_time(r["at"])).total_seconds() / 3600
        else:
            break
    rank1_exit_24h = min(
        0.22, P["RANK1_EXIT_24H"] + 0.018 * max(0.0, reign_h - 24) / 24
    )

    # Candidate set: last rank <= 40, or top-20 in the last 48h
    # (recent re-entrants). Ordered by adjusted last rank (velocity dormant).
    cand = {u for u, r in last_rank.items() if r <= P["CANDIDATE_TOP_N"]}
    for u, pts in hist.items():
        if any(
            r <= 20 and (t_end - t).total_seconds() / 3600 <= 48
            for t, r in pts
        ):
            cand.add(u)
    vel = {}
    adj = {}
    for u in cand:
        pts = hist.get(u, [])
        v = velocity(
            [t for t, _ in pts],
            [r for _, r in pts],
            t_end,
            P["VEL_WINDOW_H"],
            P["VEL_HALF_LIFE_H"],
        )
        vel[u] = v if v is not None else 0.0
        r0 = last_rank.get(u, 40)
        adj[u] = r0 + vel[u] * P["SHRINK"] * P["DRIFT_TAU_H"]
    cand = sorted(cand, key=lambda u: adj[u])[: P["CANDIDATE_TOP_N"]]

    # Per-game rank distribution, averaged over horizons h = 1..slots.
    avg = {}
    horizons = list(range(1, task["window"]["slots"] + 1))
    for u in cand:
        r0 = last_rank.get(u, 40)
        v = vel[u]
        drift_total = max(
            -P["DRIFT_MAX"],
            min(P["DRIFT_MAX"], v * P["SHRINK"] * P["DRIFT_TAU_H"]),
        )
        acc = [0.0] * (RANK_GRID + 1)  # index 0 unused
        out_mass = 0.0
        for h in horizons:
            frac = 1 - math.exp(-h / P["DRIFT_TAU_H"])
            mu = max(1.0, min(float(RANK_GRID), r0 + drift_total * frac))
            mult = (
                P["SIGMA_TOP_MULT"]
                if r0 <= 2
                else P["SIGMA_MID_MULT"]
                if r0 <= 4
                else 1.0
            )
            smin = P["SIGMA_MIN_TOP"] if r0 <= 2 else P["SIGMA_MIN"]
            sigma = min(P["SIGMA_H"] * math.sqrt(h) * mult, P["SIGMA_CAP"])
            sigma = max(sigma, smin)
            dist = [0.0] * (RANK_GRID + 1)
            tot = 0.0
            for k in range(1, RANK_GRID + 1):
                dist[k] = norm_pdf(float(k), mu, sigma)
                tot += dist[k]
            # Mass beyond the grid (rank > 45) counts as "out".
            p_out = max(0.0, 1.0 - tot)
            # Exit/entry transfer for this horizon (24h rates -> h hours).
            f = h / 24.0
            if r0 <= 10:
                p_exit = (
                    rank1_exit_24h
                    if r0 == 1
                    else min(
                        0.6,
                        P["EXIT_BASE_24H"] + P["EXIT_SLOPE_24H"] * (r0 - 1),
                    )
                )
                e = 1 - (1 - p_exit) ** f
                moved = sum(dist[1:11]) * e
                for k in range(1, 11):
                    dist[k] *= 1 - e
                if r0 == 1:
                    # Dethroned #1 falls to #2-5 (70%) or vanishes (30%).
                    for k, w in ((2, 0.28), (3, 0.18), (4, 0.14), (5, 0.10)):
                        dist[k] += moved * w
                    p_out += moved * 0.30
                else:
                    # Falls to ranks 11-30, decaying weights.
                    ws = {k: 0.5 ** ((k - 11) / 4.0) for k in range(11, 31)}
                    sw = sum(ws.values())
                    for k, w in ws.items():
                        dist[k] += moved * w / sw
            else:
                # Entry: per-game 24h entry probability decays with origin
                # rank; expected entrants from 11-20 ~= 1.44/24h, from
                # 21-30 ~= 0.12/24h (archive 24h pairs).
                if r0 <= 20:
                    w = 1.0 / (r0 - 9)
                    norm = sum(1.0 / (r - 9) for r in range(11, 21))
                    p_entry = P["ENTRY_TOTAL_11_20"] * w / norm
                else:
                    w = 1.0 / (r0 - 9)
                    norm = sum(1.0 / (r - 9) for r in range(21, 31))
                    p_entry = P["ENTRY_TOTAL_21_30"] * w / norm
                e = 1 - (1 - p_entry) ** f
                moved = sum(dist[11:]) * e + p_out * e
                for k in range(11, RANK_GRID + 1):
                    dist[k] *= 1 - e
                p_out *= 1 - e
                # Entrants land near origin rank - 6 (spread sigma 2).
                mu_land = max(2.0, r0 - 6)
                lw = {}
                ltot = 0.0
                for k in range(1, 11):
                    lw[k] = norm_pdf(float(k), mu_land, 2.0)
                    ltot += lw[k]
                for k in range(1, 11):
                    dist[k] += moved * lw[k] / ltot
            # Renormalise (pdf approximation + transfers).
            s = sum(dist[1:]) + p_out
            if s > 0:
                for k in range(1, RANK_GRID + 1):
                    acc[k] += dist[k] / s
                out_mass += p_out / s
        n_h = len(horizons)
        avg[u] = ([a / n_h for a in acc], out_mass / n_h)

    # Per-place probabilities from the horizon-averaged distributions.
    places = task["places"]
    probs = {k: {} for k in range(1, places + 1)}
    for u in cand:
        dist, _ = avg[u]
        for k in range(1, places + 1):
            probs[k][u] = dist[k]
    out_rows = []
    for k in range(1, places + 1):
        alpha = P["ALPHA_TOP"] if k <= 4 else P["ALPHA_LOW"]
        cands = sorted(probs[k], key=lambda u: -probs[k][u])[: P["MAXNAMED"]]
        named_sum = sum(probs[k][u] for u in cands)
        target = 1.0 - P["UNNAMED"]
        uni = target / len(cands) if cands else 0.0
        kept = []
        for u in cands:
            base = probs[k][u] / named_sum * target if named_sum > 0 else uni
            p = (1 - alpha) * base + alpha * uni
            if p >= P["MIN_P"]:
                kept.append([int(u), p])
        # Renormalise after the MIN_P cut so the named rows at this place
        # sum to exactly 1 - UNNAMED (the cut otherwise leaves places 1-4
        # a touch below target, inflating the unnamed share).
        kept_sum = sum(p for _, p in kept)
        if kept_sum > 0:
            scale = target / kept_sum
            for row in kept:
                row[1] = round(row[1] * scale, 6)
        for u, p in kept:
            out_rows.append({"universe_id": u, "place": k, "p": p})

    # top10 by expected rank (lowest mean rank; "out" counts as rank 50).
    # top10 does not change the log score (hits + tie-break only).
    def mean_rank(u):
        dist, out = avg[u]
        return sum(k * dist[k] for k in range(1, RANK_GRID + 1)) + out * 50.0

    top10 = [int(u) for u in sorted(cand, key=mean_rank)[:places]]
    return {"probabilities": out_rows, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    reads, task = load_history(input_dir)
    fc = build_forecast(reads, task)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task["output"]), "w") as f:
        f.write(json.dumps(fc) + "\n")
    print(f"wrote {len(fc['probabilities'])} rows, top10={fc['top10']}")


if __name__ == "__main__":
    main()
