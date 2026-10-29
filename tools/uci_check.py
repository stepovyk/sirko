#!/usr/bin/env python3
"""Сірко очима python-chess (так само його бачить lichess-bot): опції, мати, обдумування на час
суперника (ponder і ponderhit), шахи Фішера, повна партія сам із собою без жодного нелегального ходу.

  uci_check.py ШЛЯХ_ДО_SIRKO
"""
import sys
import time

import chess
import chess.engine


def main():
    cmd = sys.argv[1]
    e = chess.engine.SimpleEngine.popen_uci(cmd)
    e.configure({"Hash": 32, "Threads": 2, "Move Overhead": 50})
    assert e.id["name"].startswith("Sirko"), e.id

    # Мати в один хід (дитячий і по останній горизонталі) і задача на мат у три ходи.
    for fen, mate_in in (("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 1),
                         ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 1),
                         ("r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1", 3)):
        info = e.analyse(chess.Board(fen), chess.engine.Limit(time=3))
        score = info["score"].relative
        assert score.is_mate() and score.mate() == mate_in, f"{fen}: {score}"
    print("ok: мати знаходить")

    # Обдумування на час суперника: python-chess сам надсилає go ponder, далі ponderhit або stop.
    board = chess.Board()
    for _ in range(6):
        r = e.play(board, chess.engine.Limit(white_clock=20, black_clock=20, white_inc=0.2, black_inc=0.2), ponder=True)
        assert r.move in board.legal_moves, r
        board.push(r.move)
        if r.ponder and r.ponder in board.legal_moves:
            board.push(r.ponder)  # суперник зіграв очікуваний хід: ponderhit
        else:
            board.push(next(iter(board.legal_moves)))
    print("ok: обдумування на час суперника")

    # Шахи Фішера: python-chess сам вмикає UCI_Chess960, рокіровка – записом «король бере туру».
    board = chess.Board("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9", chess960=True)
    for _ in range(20):
        if board.is_game_over():
            break
        r = e.play(board, chess.engine.Limit(time=0.05))
        assert r.move in board.legal_moves, (board.fen(), r.move)
        board.push(r.move)
    print("ok: шахи Фішера")

    # Партія сам із собою до кінця: жодного нелегального ходу, правильний кінець.
    board = chess.Board()
    t = time.monotonic()
    while not board.is_game_over(claim_draw=True) and board.ply() < 400:
        r = e.play(board, chess.engine.Limit(time=0.03), game=1)
        assert r.move in board.legal_moves, (board.fen(), r.move)
        board.push(r.move)
    print(f"ok: партія сам із собою – {board.result(claim_draw=True)} за {board.ply()} півходів, {time.monotonic() - t:.0f} с")
    e.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
