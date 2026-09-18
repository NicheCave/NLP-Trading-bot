#!/usr/bin/env bash
# Provision a fresh Ubuntu VM to run the bot 24/7.
#
#   REPO_URL=https://github.com/you/repo.git BRANCH=bybit-futures \
#     sudo -E bash deploy/setup.sh
#
# Tested on Ubuntu 22.04/24.04, Oracle Cloud Always Free and GCP e2-micro,
# on both ARM (Ampere) and x86 — pandas, numpy and pyarrow all publish
# aarch64 wheels.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/bybit-bot}
APP_USER=${APP_USER:-botuser}
REPO_URL=${REPO_URL:-}
BRANCH=${BRANCH:-bybit-futures}

echo "==> System packages"
apt-get update -qq
apt-get install -y python3 python3-venv python3-pip git chrony

echo "==> Clock"
# The bot corrects request timestamps in software (brokers/clock_sync.py),
# so this is belt-and-braces — but a correct clock helps everything else,
# and UTC makes log timestamps line up with funding settlements.
# chrony, not systemd-timesyncd: cloud images point chrony at the provider's
# own time source, and installing timesyncd would remove it.
systemctl enable --now chrony
timedatectl set-timezone UTC

echo "==> Service user"
id -u "$APP_USER" &>/dev/null || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"

echo "==> Application"
if [ -n "$REPO_URL" ]; then
  if [ -d "$APP_DIR/.git" ]; then
    sudo -u "$APP_USER" git -C "$APP_DIR" fetch --all
    sudo -u "$APP_USER" git -C "$APP_DIR" checkout "$BRANCH"
    sudo -u "$APP_USER" git -C "$APP_DIR" pull
  else
    mkdir -p "$APP_DIR"; chown "$APP_USER:$APP_USER" "$APP_DIR"
    sudo -u "$APP_USER" git clone -b "$BRANCH" "$REPO_URL" "$APP_DIR"
  fi
else
  echo "    REPO_URL not set — copy the working tree to $APP_DIR yourself"
  mkdir -p "$APP_DIR"; chown -R "$APP_USER:$APP_USER" "$APP_DIR"
fi

echo "==> Virtualenv"
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/venv"
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install -q --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "==> Swap"
# Steady state is ~110 MB and peaks near 150 MB during an analysis cycle,
# so 1 GB is ample — but swap is cheap insurance against the OOM killer if
# a data refetch spikes.
if [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap -q /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> Credentials"
if [ -f "$APP_DIR/.env" ]; then
  chown "$APP_USER:$APP_USER" "$APP_DIR/.env"; chmod 600 "$APP_DIR/.env"
  echo "    .env locked to $APP_USER (0600)"
else
  echo "    !! No .env — it is gitignored, so it never arrives via git."
  echo "       Copy it up separately, e.g. from Windows PowerShell:"
  echo "         scp C:\Users\famou\files\bit\.env user@server:/tmp/.env"
  echo "       then:  sudo mv /tmp/.env $APP_DIR/.env"
fi

echo "==> Priming the market-data cache"
# Without this the first cycle downloads ~2 years of klines for 10 symbols
# over the VM's link, and the first few analysis cycles time out.
sudo -u "$APP_USER" "$APP_DIR/venv/bin/python" - <<'PY' || echo "    (skipped — run it after .env exists)"
import sys; sys.path.insert(0, '/opt/bybit-bot')
import config
from data import get_klines
for s in config.SYMBOLS:
    get_klines(s, '60', days=30)
    print('  cached', s)
PY

echo "==> Runtime directories"
# logs/ and cache/ are gitignored, so a fresh checkout has neither. The unit
# lists both in ReadWritePaths=, and systemd refuses to start a service whose
# ReadWritePaths= don't exist (status=226/NAMESPACE).
sudo -u "$APP_USER" mkdir -p "$APP_DIR/logs" "$APP_DIR/cache"

echo "==> systemd"
cp "$APP_DIR/deploy/bybit-bot.service" /etc/systemd/system/
systemctl daemon-reload

cat <<EOF

Done. Verify the exchange path once before starting:

    sudo -u $APP_USER $APP_DIR/venv/bin/python $APP_DIR/verify_demo.py

Then:

    sudo systemctl enable --now bybit-bot
    journalctl -u bybit-bot -f

EOF
