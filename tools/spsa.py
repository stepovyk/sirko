#!/usr/bin/env python3
"""Підбір чисел пошуку методом SPSA: Сірко грає пари партій «параметри + c·δ» проти «− c·δ»
(δ – випадкові ±1 для кожного числа), і кожне число зсувається туди, де вигравали.

  spsa.py ШЛЯХ_ДО_SIRKO --minutes 280 --workers 4 --tc 5+0.05 --out spsa.json

Кілька пар грається одночасно (асинхронний SPSA). Підсумок – середні значення за другу
половину підбору (так менше шуму), у JSON: {"назва": значення, ...}.
"""
import argparse
import json
import os
import random
import sys
import threading
import time

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import OPENINGS, play  # noqa: E402

R = 0.02   # крок відносно c: за одну пару число зсувається щонайбільше на R·c·2


def read_params(cmd):
    e = chess.engine.SimpleEngine.popen_uci([cmd], env={**os.environ, "SIRKO_TUNE": "1"})
    names = {"Hash", "Threads", "Move Overhead", "Ponder", "UCI_Chess960", "Clear Hash", "EvalFile"}
    params = {}
    for name, opt in e.options.items():
        if name in names or opt.type != "spin" or name.startswith("UCI_"):
            continue
        params[name] = {"default": opt.default, "min": opt.min, "max": opt.max}
    e.quit()
    return params


def opening(rnd):
    b = chess.Board()
    line = rnd.choice(OPENINGS).split()
    for u in line:
        b.push_uci(u)
    for _ in range(2):
        moves = sorted(m.uci() for m in b.legal_moves)
        u = moves[rnd.randrange(len(moves))]
        b.push_uci(u)
        line.append(u)
    return " ".join(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--tc", default="5+0.05")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="spsa.json")
    args = ap.parse_args()
    base, _, inc = args.tc.partition("+")
    base, inc = float(base), float(inc or 0)
    params = read_params(args.engine)
    names = sorted(params)
    theta = {n: float(params[n]["default"]) for n in names}
    c = {n: max(1.0, (params[n]["max"] - params[n]["min"]) / 20) for n in names}
    lock = threading.Lock()
    history = []
    pairs = [0]
    deadline = time.monotonic() + args.minutes * 60

    def worker(wid):
        rnd = random.Random(args.seed * 1000 + wid)
        env = {**os.environ, "SIRKO_TUNE": "1"}
        a = chess.engine.SimpleEngine.popen_uci([args.engine], env=env)
        b = chess.engine.SimpleEngine.popen_uci([args.engine], env=env)
        for e in (a, b):
            e.configure({"Hash": 16, "Threads": 1})
        try:
            while time.monotonic() < deadline:
                with lock:
                    delta = {n: rnd.choice((-1, 1)) for n in names}
                    plus = {n: theta[n] + c[n] * delta[n] for n in names}
                    minus = {n: theta[n] - c[n] * delta[n] for n in names}
                clamp = lambda v, n: int(round(min(params[n]["max"], max(params[n]["min"], v))))
                a.configure({n: clamp(plus[n], n) for n in names})
                b.configure({n: clamp(minus[n], n) for n in names})
                line = opening(rnd)
                score = 0.0
                for a_white in (True, False):
                    white, black = (a, b) if a_white else (b, a)
                    result, _, _ = play(white, black, line, base, inc, rnd.randrange(1 << 30))
                    if result == "1/2-1/2":
                        continue
                    score += 1.0 if (result == "1-0") == a_white else -1.0
                with lock:
                    for n in names:
                        lo, hi = params[n]["min"], params[n]["max"]
                        theta[n] = min(hi, max(lo, theta[n] + R * c[n] * score * delta[n]))
                    pairs[0] += 1
                    history.append(dict(theta))
        finally:
            a.quit()
            b.quit()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.workers)]
    for t in threads:
        t.start()
    last = time.monotonic()
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        if time.monotonic() - last > 600:
            last = time.monotonic()
            with lock:
                print(f"пар {pairs[0]}: " + ", ".join(f"{n}={theta[n]:.1f}" for n in names), flush=True)
    for t in threads:
        t.join()
    tail = history[len(history) // 2:] or [theta]
    final = {n: round(sum(h[n] for h in tail) / len(tail), 2) for n in names}
    json.dump({"pairs": pairs[0], "params": final}, open(args.out, "w"), indent=1)
    print(f"готово: пар {pairs[0]}")
    for n in names:
        print(f"  {n}: {params[n]['default']} → {final[n]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
