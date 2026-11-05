"""Перевірка lichess-bot: партія за листуванням, що чекає, не валить бота.

Старий lichess-bot падав у check_in_on_correspondence_games на події correspondence_ping
(KeyError: 'game'), і служба перезапускалася, кидаючи партії, що йшли. Версію з такою помилкою
sirko-bot не ставить. Тест підлаштовується під старий і новий вигляд цієї функції.
Запуск з теки lichess-bot його ж Python: python tools/lichess_bot_fix.py
"""
import inspect
import os
import queue
import sys

sys.path.insert(0, os.getcwd())
from lib import lichess_bot  # noqa: E402


class Pool:
    def __init__(self):
        self.started = []

    def apply_async(self, func, kwds, error_callback):
        self.started.append(kwds["game_id"])


def main():
    check = lichess_bot.check_in_on_correspondence_games
    source = inspect.getsource(check)
    params = inspect.signature(check).parameters
    waiting = queue.Queue()
    # Новий lichess-bot тримає в черзі пару (гра, суперник), старий – лише гру.
    waiting.put_nowait(("corrGame", "") if "opponent_name = correspondence_queue" in source else "corrGame")
    active = {}
    pool = Pool()
    args = [pool, {"type": "correspondence_ping"}, waiting, [], {}, active, 2]
    if "games_in_progress" in params:
        args.append(set())
    check(*args)
    assert pool.started == ["corrGame"], pool.started
    assert active == {"corrGame": ""}, active
    print("ok: партія за листуванням стартує без падіння lichess-bot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
