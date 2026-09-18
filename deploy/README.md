# Hosting the bot 24/7

The bot is a **long-running, stateful process** — not a web service. That rules
out most free tiers:

| | why it doesn't work |
|---|---|
| Render / Railway free | web services only, or sleeps on inactivity |
| Replit | "Always On" is paid |
| PythonAnywhere free | outbound whitelist blocks Bybit; no always-on tasks |
| AWS free tier | 12 months, then billed |
| GitHub Actions | 5-min minimum cron, no persistent state, against ToS |

Two things make ephemeral hosting actively unsafe here: `logs/executor_state.json`
holds open positions, their entry times and trailing-stop state, and a container
that resets on redeploy would lose that while real positions stayed open at the
exchange.

## What it actually needs

Measured on the live loop with 10 symbols:

| resource | usage |
|---|---|
| RAM | **~110 MB** steady, ~150 MB peak mid-cycle |
| CPU | ~10 s per 4-minute cycle — idle the rest of the time |
| Disk | ~28 MB (23 MB market-data cache + logs) |
| Network | outbound HTTPS only; no inbound, no public port |

Any always-on Linux VM with 1 GB RAM covers it with room to spare.

## Free options that work

**Oracle Cloud Always Free** — free permanently, not a trial. Either 4 ARM Ampere
cores + 24 GB RAM, or 2× AMD micro (1 GB each). Card needed for identity check
only. Common frustration: ARM shows "out of capacity" in popular regions — retry,
change region, or take the AMD micro, which is plenty.

**Google Cloud Free Tier** — `e2-micro` (1 vCPU, 1 GB) in `us-west1`,
`us-central1` or `us-east1`, always free, 30 GB disk. Usually easier to get than
Oracle ARM.

**Not free but worth considering:** a Raspberry Pi draws ~3 W, never expires, and
needs no cloud account. If the goal is only "stop leaving my laptop on", it is the
least hassle.

## Deploying

The code ships via git; the secrets do not.

```bash
REPO_URL=https://github.com/NicheCave/NLP-Trading-bot.git BRANCH=bybit-futures \
  sudo -E bash deploy/setup.sh
```

`.env` is gitignored (deliberately — it holds your API keys), so copy it up
separately from Windows PowerShell:

```powershell
scp C:\Users\famou\files\bit\.env user@your-server:/tmp/.env
```

```bash
sudo mv /tmp/.env /opt/bybit-bot/.env && sudo chmod 600 /opt/bybit-bot/.env
```

Prove the exchange path works before trusting it unattended:

```bash
sudo -u botuser /opt/bybit-bot/venv/bin/python /opt/bybit-bot/verify_demo.py
```

```bash
sudo systemctl enable --now bybit-bot && journalctl -u bybit-bot -f
```

## Things worth knowing

**Don't run the tuner on a free VM.** `run_walkforward.py` spawns one worker per
core and takes ~50 min on a 22-core desktop. On a single shared vCPU it would take
most of a day and likely trip fair-use limits. Tune locally, commit
`tuned_params.json`, pull on the server. Same for `run_backtest.py`.

**`setup.sh` primes the market-data cache** (30 days × 10 symbols). Without it the
first cycle tries to pull two years of klines over the VM's link.

**The clock stops being a problem.** A Linux VM runs NTP properly, so
`brokers/clock_sync.py` becomes a safety net rather than load-bearing — unlike on
Windows, where the time service had never once synced.

**You can finally whitelist the IP.** A VM has a stable address, so the API key can
move off "No IP restriction" — more secure, and it removes the 3-month auto-expiry
that unrestricted keys carry. Do it after confirming the bot runs, and note IP
whitelisting doesn't apply to private WebSocket requests.

**Everything is UTC**, so log timestamps line up with funding settlements at 00:00,
08:00 and 16:00.

## Operating it

```bash
journalctl -u bybit-bot -f              # live logs
```

```bash
systemctl status bybit-bot              # is it alive?
```

```bash
cd /opt/bybit-bot && sudo -u botuser git pull && sudo systemctl restart bybit-bot
```

The service restarts on crash and on reboot. On startup the bot reconciles
`logs/executor_state.json` against the exchange, so an interrupted session recovers
rather than double-counting — and because stop-losses live at Bybit, open positions
stay protected even while the bot is down.
