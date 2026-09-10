# KBet — Football Betting Intelligence (Dixon-Coles + ELO)

> **We don't gamble. We exploit market inefficiencies.** Value betting with no guessing — 10 leagues, Monte Carlo Poisson, CLV tracking. Live at `kbet-7fyx.onrender.com`

## Live Demo
- **Site:** `https://kbet-7fyx.onrender.com` → Daily Bet Card `10` ranked value bets, `SP1 D1 E0...` `10 leagues`
- **API:** `/card/today` `/card/{date}` `/api/mode` `LIVE/SIMULATED` `/performance` `/clv`
- **Admin:** `POST /admin/run-cron` `X-Admin-Token: kbet-admin-Kaks753-2026`

## Methodology
1. **Dixon-Coles** Bivariate Poisson `xi 0.0065` half-life 107d `home_adv 0.25` `rho [-0.5,0]` `kbet/engine/models/dixon_coles.py:34` tau correction, vectorized `scipy.optimize L-BFGS-B`
2. **Isotonic Calibration** `kbet/daily_card.py:219` last 25% tail, `Shin de-vig` `kbet/engine/utils/devig.py:119` (not flat 5%)
3. **Blend** `20% DC + 80% Pinnacle` `kbet/daily_card.py:504` must beat Pinnacle `gap 0.012` `kbet/daily_card.py:96` else skip
4. **Trends** 60-day rolling `2.72g` `kbet/daily_card.py:592` `+0.008/0.1g` above `2.75`
5. **Value** `EV=(prob*odds-1)*(1-0.20 slippage)` `kbet/daily_card.py:539` cap `0.20` hard `0.12` warn, overround `0.98-1.15` reject, `1x2 EV>0.025` `O/U 0.04` `BTTS 0.05` `kbet/daily_card.py:83`
6. **Confidence** `0.4*EV/0.15 +0.2*prob +0.2*gap -0.1 rain` `kbet/daily_card.py:976` → `★` `>0.80=5`
7. **Ranking** `score=EV*conf*(1+clv)` `kbet/daily_card.py:802` `max 2/match` `2/team` `kbet/daily_card.py:106` adaptive `1-day` → thin `<5` or `<8` matches merges `+1 day` top `10` `kbet/daily_card.py:392` horizon `kbet/api/app.py:220` `2-day pool`
8. **Staking** Quarter Kelly `0.25` `max 2%/bet 5%/day` `kbet/engine/utils/kelly.py:43` `kbet/engine/utils/kelly.py:31`

## Data Sources (10 leagues, free start)
| Layer | Source | What | Tier |
|-------|--------|------|------|
| Stats | `football-data.co.uk` | 24294 matches 2018-2025 + closing `Pinnacle` `kbet/data/processed/all_matches.parquet` | Free CSV |
| Live | `The Odds API` `kbet/engine/data/odds_client.py:39` `soccer_epl…g1` `10` `h2h/totals/btts` | Live `10cr/day` `<500/mo free` `beceff...0651` |
| Elo | `ClubElo` `kbet/engine/data/clubelo_client.py:39` `65` adv | Fallback DC |
| Weather | `Open-Meteo` `kbet/engine/data/weather_client.py:33` `wind>35 -0.08` | Free |
| Glue | `kbet/data/processed/entity_registry.json` 250+ teams | Variants map |

## Backtest (Sacred Gate)
- **Walk-forward** `kbet/backtester/backtest_engine.py:1` 3 rounds `2022-23/23-24/24-25` `kbet/config/settings.py:110` `brier_gate 0.62` `roi_gate 0.03` `min 300` bets 20% slippage
- **Results** `kbet/data/backtest_results/week1_findings.md:46` `V3 Brier 0.576 ROI -1.83%` `V4 0.579 -4.24%` `V5 -22%` synthetic `CLV 18%` artifact → **structural: close vs close `CLV 0`** `week1_findings.md:109` needs live `T-48h` `kbet/ingest_odds_snapshots.py:257`
- **Run:** `python kbet/run_backtest_v5.py --markets 1x2 ou --rounds 1 2 3`

## Setup (Local)
```bash
git clone https://github.com/Kaks753/betting.git && cd betting
pip install -r requirements.txt  # numpy pandas scipy sklearn pyarrow requests fastapi uvicorn
python kbet/engine/scrapers/download_data.py  # or use committed parquet
ODDS_API_KEY=beceff... python kbet/daily_card.py --date 2026-09-10 --simulate --max-bets 12
ODDS_API_KEY=beceff... KBET_ENV=production python kbet/cron_runner.py --date 2026-09-10 --card-only
uvicorn kbet.api.app:app --host 0.0.0.0 --port 8000
pytest kbet/tests/test_week3.py -q  # 57 tests incl. EV<20% sum=1 odds sanity
```

## Deploy (Render Free $0 until earning)
- **Render** `render.yaml:21` `plan: free` `docker` `kbet-7fyx.onrender.com` `DATA_DIR=/tmp/kbet_data` ephemeral, `daily_cards/*.json` committed fallback `kbet/api/app.py:223`
- **Env** `ADMIN_TOKEN=kbet-admin-Kaks753-2026` `ODDS_API_KEY=beceff...0651` `KBET_ENV=production` `kbet/api/app.py:135` guards `SIMULATED` banner `kbet/api/templates/dashboard.html:758`
- **Cron** `10:00 UTC` `kbet/cron_runner.py --card-only` freshness `<12h` skip `kbet/api/app.py:103`
- **Upgrade** `Starter $7 +1GB /data` `DATA_DIR=/data` only after `50` live bets `CLV>0.3%`

## Limitations
- `Corners/Cards` `kbet/engine/models/corners_model.py:58` `cards_model.py:66` fitted but `ODDS_API` `totals` no corners market on free → advisory `OBTAIN ODDS` `kbet/daily_card.py:650`
- `xG` `Understat/FBref` scrape `TODO` — DC uses raw goals; `BTTS` `kbet/daily_card.py:740` now live via `btts` market
- `Brier 0.62` lenient vs `0.667` random; tighten to `0.60` after live `50` bets

## Legal
`18+` `BeGambleAware.org` `kbet/api/templates/dashboard.html:758` footer. Not a bookmaker — analytics only. No `guaranteed` language, `EV` + `hash` `kbet/data/public_picks/hash_*.txt` `kbet/daily_card.py:1139` proves no cherry-pick.
