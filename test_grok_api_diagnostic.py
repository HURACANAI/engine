#!/usr/bin/env python3
"""Comprehensive diagnostic test for Grok API."""

import requests
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cloud.training.config.settings import EngineSettings

def test_api_key():
    """Test Grok API key with various methods."""
    print("🔍 Grok API Diagnostic Test")
    print("=" * 60)
    print()
    
    # Load settings
    settings = EngineSettings.load()
    api_key = settings.notifications.grok_api_key
    
    if not api_key:
        print("❌ No API key found in settings")
        return
    
    api_key = api_key.strip()
    print(f"🔑 API Key: {api_key[:15]}...{api_key[-5:]}")
    print(f"📏 Length: {len(api_key)}")
    print(f"✅ Format check: {'Valid (starts with gsk_)' if api_key.startswith('gsk_') else 'Invalid'}")
    print()
    
    # Test 1: Check if we can list models
    print("1️⃣  Testing: List available models...")
    try:
        response = requests.get(
            "https://api.x.ai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.ok:
            models = response.json()
            print(f"   ✅ Success! Available models:")
            if 'data' in models:
                for model in models['data'][:5]:
                    print(f"      - {model.get('id', 'unknown')}")
            else:
                print(f"   Response: {json.dumps(models, indent=2)[:300]}")
        else:
            print(f"   ❌ Failed: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Exception: {str(e)[:100]}")
    print()
    
    # Test 2: Try different model names
    models_to_test = [
        "grok-2-latest",
        "grok-beta",
        "grok-2",
        "grok-2-1212",
        "grok-vision-beta",
    ]
    
    print("2️⃣  Testing: Different model names...")
    for model in models_to_test:
        try:
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "Say hello"}
                    ],
                    "max_tokens": 10
                },
                timeout=10
            )
            
            if response.ok:
                print(f"   ✅ {model}: WORKS!")
                result = response.json()
                if 'choices' in result:
                    print(f"      Response: {result['choices'][0]['message']['content']}")
                break
            else:
                error_text = response.text[:150]
                print(f"   ❌ {model}: {response.status_code} - {error_text}")
        except Exception as e:
            print(f"   ❌ {model}: Exception - {str(e)[:50]}")
    print()
    
    # Test 3: Try minimal request
    print("3️⃣  Testing: Minimal request (no system message)...")
    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-2-latest",
                "messages": [
                    {"role": "user", "content": "Hi"}
                ]
            },
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.ok:
            print(f"   ✅ Success!")
            result = response.json()
            print(f"   Response: {json.dumps(result, indent=2)[:300]}")
        else:
            print(f"   ❌ Failed:")
            print(f"   {response.text[:500]}")
            try:
                error_json = response.json()
                print(f"   Error details: {json.dumps(error_json, indent=2)}")
            except:
                pass
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test 4: Check API key format variations
    print("4️⃣  Testing: API key format variations...")
    variations = [
        api_key,
        api_key.strip(),
        api_key.replace(" ", ""),
    ]
    
    for i, key_var in enumerate(variations):
        if key_var == api_key:
            continue
        print(f"   Testing variation {i+1}...")
        try:
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key_var}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-2-latest",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10
                },
                timeout=10
            )
            if response.ok:
                print(f"   ✅ Variation {i+1} WORKS!")
                break
            else:
                print(f"   ❌ Variation {i+1}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Variation {i+1}: {str(e)[:50]}")
    print()
    
    # Test 5: Check if free plan has different endpoint
    print("5️⃣  Testing: Alternative endpoints...")
    endpoints = [
        "https://api.x.ai/v1/chat/completions",
        "https://api.x.ai/v1/completions",
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-2-latest",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10
                },
                timeout=10
            )
            print(f"   {endpoint}: {response.status_code}")
            if response.ok:
                print(f"   ✅ This endpoint works!")
        except Exception as e:
            print(f"   {endpoint}: Exception - {str(e)[:50]}")
    print()
    
    print("=" * 60)
    print("💡 Recommendations:")
    print("   1. Verify the API key at https://console.x.ai")
    print("   2. Check if free plan supports API access")
    print("   3. Ensure the key is activated and not expired")
    print("   4. Check xAI documentation for free plan limitations")
    print("=" * 60)

if __name__ == "__main__":
    test_api_key()

