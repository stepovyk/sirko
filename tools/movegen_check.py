#!/usr/bin/env python3
"""Генерація ходів Сірка проти python-chess на тисячах випадкових позицій (класика й Фішера):
множина легальних ходів і позиція (FEN) після кожного з них мають збігатися.

  movegen_check.py ШЛЯХ_ДО_SIRKO [ПАРТІЙ]
"""
import random
import subprocess
import sys

import chess


def positions(games, rnd):
    out = []
    for g in range(games):
        fischer = g % 3 == 0
        board = chess.Board.from_chess960_pos(rnd.randrange(960)) if fischer else chess.Board()
        board.chess960 = fischer
        for _ in range(rnd.randrange(20, 160)):
            moves = list(board.legal_moves)
            if not moves:
                break
            # Частіше взяття й промоції: там найбільше тонкощів.
            sharp = [m for m in moves if board.is_capture(m) or m.promotion]
            board.push(rnd.choice(sharp if sharp and rnd.random() < 0.4 else moves))
            out.append((board.fen(en_passant="legal", shredder=fischer), fischer))
    return out


def expected(fen, fischer):
    board = chess.Board(fen, chess960=fischer)
    lines = []
    for m in board.legal_moves:
        uci = m.uci() if fischer else board.uci(m)
        board.push(m)
        lines.append(f"{uci} {board.fen(en_passant='legal', shredder=fischer)}")
        board.pop()
    return sorted(lines)


def main():
    engine = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    cases = positions(games, random.Random(7))
    text = "\n".join(f for f, _ in cases) + "\n"
    got = subprocess.run([engine, "moves"], input=text, capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(got) == len(cases), f"Сірко відповів на {len(got)} позицій з {len(cases)}"
    bad = 0
    for (fen, fischer), line in zip(cases, got):
        mine = line.split("|") if line else []
        want = expected(fen, fischer)
        if mine != want:
            bad += 1
            if bad <= 5:
                print("розбіжність у", fen)
                print("  лише в Сірка:", sorted(set(mine) - set(want))[:4])
                print("  лише в python-chess:", sorted(set(want) - set(mine))[:4])
    print(f"ok: генерація ходів збігається з python-chess у {len(cases) - bad} з {len(cases)} позицій")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
