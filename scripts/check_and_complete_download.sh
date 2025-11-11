#!/bin/bash
# Check download status and complete missing coins

DROPBOX_TOKEN="sl.u.AGFGiCwLVyXLDRrc0RKgYtd7uZu_IeqNtSVpJqG6NbY0yz6TkUIqns1hsXi8PtW8_tBNRYMgPRvju_zii5BddOlZdZS_9F-c81TGvG5wu1LkspNaNHnGEWB1Djxv-Oe4o4shFAC-JutB5utpaSoVkFaJlX6iaKMJTif3DJajVzESj7NumwYIMmLqn_4y1LQw3pah-6DdqgtDNSfUAfh5WqWiEC8DOzTTP2-zFvBYnbaoOH0f6ZjLXg_-BolLqeOXnDI-3Ee5aj05jyBXrBwmYITZpECMqFIPlTXo_tc8j4LdMNnHgaojzXvb6_NWy6Lmsbeh1tSFWRpwI-JtJSTQgfDhh59S7jWsqWU-mj3F8Z_uXBrYDtbynAec-H2c1mF0hm09Zgk3d6N4l2WjWLiNskbTmKHGK7Qeyqoq5dkS00zdrgpJfCvo8-RS0zPQ44b2ck9tADF_N7Q9uwmsdh_h2fc4axWQbO_w4ZQBeYH_9HHz5mB0M6u1W_Z1PCW4Tu5pj4l_JA3WLAIKOGVs1LD8kPPAEcSA63gDc_KMZItd5NdP2ePzUMJIdp-CeRGBOqRhjLCkJufWa7doy1fsWexbBitt_A7I96GW-1T8FeZ2ltZCLCATffRDAYh2LHp0Iigi5LLIVVr5MCIlioHYR7hu0xLdd-G9KEfA-PUyN3K2e4Vy7BPLe0Z1a7g1XYfy9yYLDX5hsPQsSGCuljFfhvhNud-F_zFrkmJb94vE3vpmK16afZe06ZQelZm84_aXkK4pzgmJzVLUX2kWWfkmApXlZC39jKZRXFbdZ3VIUrGo1ZWXS8kqNKxguwNX-RQ7d3H2fw5cZZuWiKUB443IdVmVF_eaDbPQFZqa2YaICQk6oMonhMvDOR8iOc6AFgMwPaMRk93gyXeipbPrYYI5DSdafoWWsI4u143m1-apCJpVbDJRvBUznOzlb66SKBsTq471b-ImsX9vlmCGJ6ugdaPrXaHAzB1UsnwPjWovCi6Wc_nSSTVsXvGYveiCAwpyKCVuK4Ybuqmfon6Dt_zRGGWKISi4OyrpEbo3tGld0FJB9BXEVixprM8blrBN-hHN26ZyS1q_ad8hmPmiwqvlZ3InoTj7SFcS2gf_JQJfqIa7A-_djO49D_4oGdvESCciIAqjb1CtEbzn1uXyNIOlC4xAh5xf6ScISIANh_eR7UhqXsA5eXDSd5LRbL7O-89y-arY_ymxUOvCyXN2SB9Q2vZ6FXv76ud9y8Uljw0IkeYnrgikMSJnBgMz2ZDYDovTLtto4Zij736hratutNcWXKcSoL5k"

echo "=========================================="
echo "🔍 CHECKING DOWNLOAD STATUS"
echo "=========================================="
echo ""

# Check progress file
if [ -f "data/download_progress.json" ]; then
    COMPLETED=$(python3 -c "import json; p=json.load(open('data/download_progress.json')); print(len(p.get('completed', [])))" 2>/dev/null || echo "0")
    FAILED=$(python3 -c "import json; p=json.load(open('data/download_progress.json')); print(len(p.get('failed', [])))" 2>/dev/null || echo "0")
    echo "Completed: $COMPLETED coins"
    echo "Failed: $FAILED coins"
    echo ""
    
    if [ "$COMPLETED" -lt "249" ]; then
        MISSING=$((249 - COMPLETED))
        echo "⚠️  Missing: $MISSING coins"
        echo ""
        echo "🚀 Starting download of missing coins..."
        echo ""
        
        # Run robust downloader with resume
        python3 scripts/robust_download_top250.py \
            --dropbox-token "$DROPBOX_TOKEN" \
            --top 250 \
            --days 1095 \
            --timeframe 1d \
            --resume \
            --delay 0.5
    else
        echo "✅ All 249 coins downloaded!"
    fi
else
    echo "⚠️  No progress file found. Starting fresh download..."
    echo ""
    
    # Run robust downloader
    python3 scripts/robust_download_top250.py \
        --dropbox-token "$DROPBOX_TOKEN" \
        --top 250 \
        --days 1095 \
        --timeframe 1d \
        --delay 0.5
fi

echo ""
echo "=========================================="
echo "✅ CHECK COMPLETE"
echo "=========================================="

