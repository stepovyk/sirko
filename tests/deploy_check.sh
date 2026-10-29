#!/usr/bin/env bash
# Встановлення бота на чистій Ubuntu 24.04, як на сервері: системний cargo (1.75), збирання від
# безправного користувача, perft і перевірка UCI, налаштування очима lichess-bot, служби systemd.
# lichess-bot ставиться туди, де його тримає Характерник.
set -Eeuo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
lb=/opt/kharakternyk/lichess-bot
install -d -m 755 /opt/kharakternyk
git clone -q --depth 1 https://github.com/lichess-bot-devs/lichess-bot "${lb}"
python3 -m venv "${lb}/venv"
"${lb}/venv/bin/python" -m pip install -q -r "${lb}/requirements.txt"
chmod -R a+rX "${lb}"

install -d -m 755 /opt/sirko
rm -rf /opt/sirko/src
cp -r "${here}" /opt/sirko/src

# shellcheck source=deploy/sirko-bot
source /opt/sirko/src/deploy/sirko-bot
trap - EXIT
ensure_packages
ensure_user
rustc --version
install -d -m 755 "${ETC}"
printf 'LICHESS_BOT_TOKEN=lip_check\n' >"${TOKEN_FILE}"
build_engine
((ENGINE_NEW)) || { echo "::error::рушій не зібрано"; exit 1; }
render_config "${CONF}.new"
validate_config "${CONF}.new" sirko.new
grep -E 'Threads|Hash|pgn_directory' "${CONF}.new"
for unit in "${UNITS[@]}"; do install -m 644 "${SRC}/deploy/systemd/${unit}" "/etc/systemd/system/${unit}"; done
mv -f "${ENGINE}/sirko.new" "${ENGINE}/sirko"
mv -f "${CONF}.new" "${CONF}"
systemd-analyze verify "/etc/systemd/system/${NAME}.service" "/etc/systemd/system/${NAME}-update.service" "/etc/systemd/system/${NAME}-update.timer"
echo "ok: встановлення Сірка на чистій Ubuntu"
