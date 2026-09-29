#!/usr/bin/env bash
# Runs ON the Oracle server (Ubuntu). Safe to re-run: every step is idempotent.
# Installs Docker, opens ports 80/443 in the server firewall, and starts the app.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n==> %s\n' "$*"; }

# --- 1. config ---------------------------------------------------------------
if [ ! -f .env ]; then
  echo "deploy/oracle/.env is missing. Fill in env.example locally and re-run upload.ps1," >&2
  echo "or here: cp env.example .env && nano .env" >&2
  exit 1
fi
provider=$(sed -n 's/^LLM_PROVIDER=//p' .env)
case "$provider" in
  groq) key_var=GROQ_API_KEY ;;
  gemini) key_var=GEMINI_API_KEY ;;
  anthropic) key_var=ANTHROPIC_API_KEY ;;
  *) key_var="" ;;
esac
for var in APP_PASSWORD $key_var; do
  if ! grep -Eq "^${var}=.+" .env; then
    echo "Set ${var} in deploy/oracle/.env first." >&2
    exit 1
  fi
done

if ! grep -Eq '^DOMAIN=.+' .env; then
  ip=$(curl -fsS https://api.ipify.org || curl -fsS https://ifconfig.me)
  domain="${ip//./-}.sslip.io"   # free hostname that resolves to this IP, so HTTPS works without buying a domain
  sed -i '/^DOMAIN=/d' .env
  echo "DOMAIN=${domain}" >> .env
  say "Using free hostname ${domain}"
fi
domain=$(grep -E '^DOMAIN=' .env | cut -d= -f2)

# --- 2. Docker -----------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  say "Installing Docker"
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker >/dev/null

# --- 3. firewall inside the server ---------------------------------------------
# Oracle's Ubuntu images ship iptables rules that reject everything except SSH.
# (Ports must ALSO be opened in the VCN security list in the Oracle console.)
if command -v iptables >/dev/null 2>&1; then
  for port in 80 443; do
    if ! sudo iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null; then
      reject_line=$(sudo iptables -L INPUT --line-numbers | awk '/REJECT/ {print $1; exit}')
      if [ -n "$reject_line" ]; then
        sudo iptables -I INPUT "$reject_line" -p tcp --dport "$port" -m state --state NEW -j ACCEPT
      else
        sudo iptables -A INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT
      fi
    fi
  done
  if command -v netfilter-persistent >/dev/null 2>&1; then
    sudo netfilter-persistent save >/dev/null
  fi
  say "Firewall: ports 80 and 443 open"
fi

# --- 4. start --------------------------------------------------------------------
mkdir -p data
say "Building and starting (first build takes several minutes: it downloads the models)"
sudo docker compose up -d --build

say "Waiting for the app to become healthy"
for _ in $(seq 1 60); do
  if sudo docker compose exec -T app python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" >/dev/null 2>&1; then
    say "Running at https://${domain}   (log in with any username + APP_PASSWORD)"
    echo "    The HTTPS certificate is issued on the first visit; if the browser warns, wait a minute and reload."
    exit 0
  fi
  sleep 5
done
echo "The app did not become healthy in time. Logs: sudo docker compose logs app" >&2
exit 1
