#!/bin/bash
# KBet Daily Cron — run via Fly.io scheduled machine
# Set up with: flyctl machine run . --schedule daily --command "bash fly-cron.sh"
# Or just hit the admin endpoint daily:
#   curl -X POST "https://kbet-api.fly.dev/admin/run-cron?simulate=false" \
#        -H "X-Admin-Token: $ADMIN_TOKEN"

set -e
echo "[$(date -u)] KBet daily cron starting..."
cd /app
python3 kbet/cron_runner.py
echo "[$(date -u)] KBet daily cron complete."
