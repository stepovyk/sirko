#!/usr/bin/env bash
# Встановлення бота на чистій Ubuntu 24.04, як на сервері, справжньою командою sirko-bot install:
# системні пакети, свій lichess-bot з master (з перевіркою на партіях за листуванням), збирання
# системним cargo від безправного користувача, perft і перевірка UCI, налаштування очима
# lichess-bot, служби systemd. Ключа Lichess тут немає: бот має все підготувати й чекати на нього.
set -Eeuo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
install -d -m 755 /opt/sirko
rm -rf /opt/sirko/src
cp -r "${here}" /opt/sirko/src
chown -R root:root /opt/sirko/src

/opt/sirko/src/deploy/sirko-bot install

fail() { echo "::error::$*"; exit 1; }
test -x /opt/sirko/engine/sirko || fail "рушія немає"
test "$(cat /opt/sirko/engine/VERSION)" = "$(git -C /opt/sirko/src rev-parse HEAD)" || fail "рушій не з цього коду"
test -x /opt/sirko/lichess-bot/venv/bin/python || fail "lichess-bot не встановлено"
test ! -e /opt/sirko/lichess-bot.new -a ! -e /opt/sirko/build || fail "лишилося недоготоване"
test "$(stat -c %U /opt/sirko/lichess-bot/lichess-bot.py)" = root || fail "lichess-bot має належати root"
grep -qE "Threads: $(nproc)$" /opt/sirko/config.yml || fail "рушій бере не всі ядра"
grep -E 'Threads|Hash|pgn_directory' /opt/sirko/config.yml
systemctl is-enabled -q sirko.service sirko-update.timer || fail "служби не ввімкнено"
systemctl is-active -q sirko.service && fail "без ключа бот не має стартувати"
systemd-analyze verify /etc/systemd/system/sirko.service /etc/systemd/system/sirko-update.service /etc/systemd/system/sirko-update.timer
/usr/local/bin/sirko-bot status

# Повторне встановлення нічого не перезбирає й не ламає.
out=$(/opt/sirko/src/deploy/sirko-bot install 2>&1)
printf '%s\n' "${out}"
grep -q "lichess-bot – найновіший" <<<"${out}" || fail "lichess-bot ставиться вдруге"
grep -q "вже зібраний" <<<"${out}" || fail "рушій збирається вдруге"
echo "ok: встановлення Сірка на чистій Ubuntu"
