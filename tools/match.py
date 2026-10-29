#!/usr/bin/env python3
"""Матч двох UCI-рушіїв: різні дебюти, кожен дебют – двічі зі зміною кольорів.

  match.py --a "./sirko" --b "stockfish" --b-opt UCI_LimitStrength=true --b-opt UCI_Elo=2000 \
           --games 40 --tc 10+0.1

Підсумок: перемоги, нічиї й поразки першого рушія, очки й оцінка різниці в Elo з похибкою.
"""
import argparse
import math
import random
import shlex
import sys

import chess
import chess.engine
import chess.pgn

# Збалансовані початки (перші ходи), щоб партії не повторювалися.
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4", "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7", "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6", "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6", "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5", "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5", "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6", "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6", "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5", "c2c4 c7c5 g1f3 g8f6 b1c3 b8c6",
    "e2e4 e7e5 f2f4 e5f4 g1f3 g7g5", "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6", "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6",
    "g1f3 g8f6 c2c4 b7b6 g2g3 c8b7", "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4", "d2d4 e7e6 c2c4 f8b4 b1c3 g8f6",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6", "b2b3 e7e5 c1b2 b8c6 e2e3 g8f6", "e2e4 b8c6 d2d4 d7d5 b1c3 d5e4",
]


def parse_tc(tc):
    base, _, inc = tc.partition("+")
    return float(base), float(inc or 0)


def open_engine(cmd, opts):
    e = chess.engine.SimpleEngine.popen_uci(shlex.split(cmd))
    conf = {}
    for o in opts:
        k, _, v = o.partition("=")
        conf[k] = int(v) if v.lstrip("-").isdigit() else (v.lower() == "true" if v.lower() in ("true", "false") else v)
    if conf:
        e.configure(conf)
    return e


def play(white, black, opening, base, inc, game_no):
    board = chess.Board()
    for u in opening.split():
        board.push_uci(u)
    clock = [base, base]
    engines = [white, black]
    for e in engines:
        e.protocol.send_line("ucinewgame") if hasattr(e.protocol, "send_line") else None
    while not board.is_game_over(claim_draw=True):
        side = 0 if board.turn == chess.WHITE else 1
        limit = chess.engine.Limit(white_clock=clock[0], black_clock=clock[1], white_inc=inc, black_inc=inc)
        import time
        t = time.monotonic()
        try:
            res = engines[side].play(board, limit, game=game_no)
        except chess.engine.EngineError as err:
            return ("0-1" if side == 0 else "1-0"), f"рушій зламався: {err}", board
        used = time.monotonic() - t
        clock[side] -= used
        if clock[side] < 0:
            return ("0-1" if side == 0 else "1-0"), "час", board
        clock[side] += inc
        if res.move is None or res.move not in board.legal_moves:
            return ("0-1" if side == 0 else "1-0"), f"нелегальний хід {res.move}", board
        board.push(res.move)
        if board.ply() > 600:
            return "1/2-1/2", "довжина", board
    outcome = board.outcome(claim_draw=True)
    return outcome.result(), outcome.termination.name.lower(), board


def elo(score, n):
    if n == 0 or score <= 0 or score >= n:
        return float("nan"), float("nan")
    p = score / n
    d = -400 * math.log10(1 / p - 1)
    # Похибка 95%: з дисперсії результатів.
    return d, 400 / math.log(10) * 1.96 * math.sqrt(p * (1 - p) / n) / (p * (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-opt", action="append", default=[])
    ap.add_argument("--b-opt", action="append", default=[])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--tc", default="10+0.1")
    ap.add_argument("--pgn", default="")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    base, inc = parse_tc(args.tc)
    a = open_engine(args.a, args.a_opt)
    b = open_engine(args.b, args.b_opt)
    rnd = random.Random(args.seed)
    openings = OPENINGS[:]
    rnd.shuffle(openings)
    w = d = l = 0
    reasons = {}
    pgn_out = open(args.pgn, "w") if args.pgn else None
    for g in range(args.games):
        opening = openings[(g // 2) % len(openings)]
        a_white = g % 2 == 0
        white, black = (a, b) if a_white else (b, a)
        result, why, board = play(white, black, opening, base, inc, g)
        if result == "1/2-1/2":
            d += 1
        elif (result == "1-0") == a_white:
            w += 1
        else:
            l += 1
            reasons[why] = reasons.get(why, 0) + 1
        if pgn_out:
            game = chess.pgn.Game.from_board(board)
            game.headers["White"] = "A" if a_white else "B"
            game.headers["Black"] = "B" if a_white else "A"
            game.headers["Result"] = result
            game.headers["Termination"] = why
            print(game, file=pgn_out, end="\n\n", flush=True)
        n = w + d + l
        diff, err = elo(w + d / 2, n)
        print(f"[{n}/{args.games}] +{w} ={d} -{l}  очки {w + d / 2}/{n}  Elo {diff:+.0f} ± {err:.0f}  ({result} {why})", flush=True)
    a.quit()
    b.quit()
    if reasons:
        print("поразки:", reasons)


if __name__ == "__main__":
    sys.exit(main())
