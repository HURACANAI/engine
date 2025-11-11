#!/bin/bash
# Download Top 250 Coins, Train Top 3
# Quick script to download 250 coins and train on BTC, ETH, SOL

set -e

# Set Dropbox token
export DROPBOX_ACCESS_TOKEN="sl.u.AGFGiCwLVyXLDRrc0RKgYtd7uZu_IeqNtSVpJqG6NbY0yz6TkUIqns1hsXi8PtW8_tBNRYMgPRvju_zii5BddOlZdZS_9F-c81TGvG5wu1LkspNaNHnGEWB1Djxv-Oe4o4shFAC-JutB5utpaSoVkFaJlX6iaKMJTif3DJajVzESj7NumwYIMmLqn_4y1LQw3pah-6DdqgtDNSfUAfh5WqWiEC8DOzTTP2-zFvBYnbaoOH0f6ZjLXg_-BolLqeOXnDI-3Ee5aj05jyBXrBwmYITZpECMqFIPlTXo_tc8j4LdMNnHgaojzXvb6_NWy6Lmsbeh1tSFWRpwI-JtJSTQgfDhh59S7jWsqWU-mj3F8Z_uXBrYDtbynAec-H2c1mF0hm09Zgk3d6N4l2WjWLiNskbTmKHGK7Qeyqoq5dkS00zdrgpJfCvo8-RS0zPQ44b2ck9tADF_N7Q9uwmsdh_h2fc4axWQbO_w4ZQBeYH_9HHz5mB0M6u1W_Z1PCW4Tu5pj4l_JA3WLAIKOGVs1LD8kPPAEcSA63gDc_KMZItd5NdP2ePzUMJIdp-CeRGBOqRhjLCkJufWa7doy1fsWexbBitt_A7I96GW-1T8FeZ2ltZCLCATffRDAYh2LHp0Iigi5LLIVVr5MCIlioHYR7hu0xLdd-G9KEfA-PUyN3K2e4Vy7BPLe0Z1a7g1XYfy9yYLDX5hsPQsSGCuljFfhvhNud-F_zFrkmJb94vE3vpmK16afZe06ZQelZm84_aXkK4pzgmJzVLUX2kWWfkmApXlZC39jKZRXFbdZ3VIUrGo1ZWXS8kqNKxguwNX-RQ7d3H2fw5cZZuWiKUB443IdVmVF_eaDbPQFZqa2YaICQk6oMonhMvDOR8iOc6AFgMwPaMRk93gyXeipbPrYYI5DSdafoWWsI4u143m1-apCJpVbDJRvBUznOzlb66SKBsTq471b-ImsX9vlmCGJ6ugdaPrXaHAzB1UsnwPjWovCi6Wc_nSSTVsXvGYveiCAwpyKCVuK4Ybuqmfon6Dt_zRGGWKISi4OyrpEbo3tGld0FJB9BXEVixprM8blrBN-hHN26ZyS1q_ad8hmPmiwqvlZ3InoTj7SFcS2gf_JQJfqIa7A-_djO49D_4oGdvESCciIAqjb1CtEbzn1uXyNIOlC4xAh5xf6ScISIANh_eR7UhqXsA5eXDSd5LRbL7O-89y-arY_ymxUOvCyXN2SB9Q2vZ6FXv76ud9y8Uljw0IkeYnrgikMSJnBgMz2ZDYDovTLtto4Zij736hratutNcWXKcSoL5k"

cd "$(dirname "$0")/.."

echo "=============================================================================="
echo "🚀 DOWNLOAD TOP 250 COINS, TRAIN TOP 3"
echo "=============================================================================="
echo ""

# Step 1: Download top 250 coins
echo "STEP 1: Downloading top 250 coins..."
echo "----------------------------------------------------------------------"
python scripts/simple_download_candles.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h \
  --no-adaptive

echo ""
echo "STEP 2: Training top 3 coins (BTC, ETH, SOL)..."
echo "----------------------------------------------------------------------"
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h

echo ""
echo "=============================================================================="
echo "✅ COMPLETE!"
echo "=============================================================================="

