#!/usr/bin/env python3
"""Test Grok API with exact request format."""

import requests
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cloud.training.config.settings import EngineSettings

def test_exact():
    """Test with exact API key from config."""
    settings = EngineSettings.load()
    api_key = settings.notifications.grok_api_key
    
    if not api_key:
        print("❌ No API key found")
        return
    
    # Get exact key value
    api_key = api_key.strip()
    print(f"🔑 API Key (exact): {repr(api_key)}")
    print(f"📏 Length: {len(api_key)}")
    print(f"🔤 First 20 chars: {api_key[:20]}")
    print(f"🔤 Last 10 chars: {api_key[-10:]}")
    print()
    
    # Test with exact format from xAI docs
    print("Testing with exact xAI API format...")
    
    url = "https://api.x.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "grok-2-latest",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "max_tokens": 10
    }
    
    print(f"URL: {url}")
    print(f"Headers: {json.dumps({k: v[:20] + '...' if len(v) > 20 else v for k, v in headers.items()}, indent=2)}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print()
        
        if response.ok:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print("❌ FAILED")
            print(f"Response Text: {response.text}")
            print()
            
            # Try to parse error
            try:
                error_json = response.json()
                print(f"Error JSON: {json.dumps(error_json, indent=2)}")
            except:
                print("Could not parse error as JSON")
            
            # Check if it's an authentication issue
            if response.status_code == 401:
                print("\n⚠️ 401 Unauthorized - Authentication issue")
            elif response.status_code == 400:
                print("\n⚠️ 400 Bad Request - Check request format or API key")
                
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exact()

