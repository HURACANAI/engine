#!/usr/bin/env python3
"""Test script to verify Dropbox token works independently."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import dropbox
    from dropbox.exceptions import AuthError, ApiError
except ImportError:
    print("❌ dropbox package not installed. Install with: pip install dropbox")
    sys.exit(1)

# Token from settings (same as in code)
TOKEN = "sl.u.AGFdM18mLSy6vc0ckgqc5MazOFDG4z0HH1QqjzfAmZ1HuSTmrgwT3HKVSVzEkUTPWsOrUBSL6kSK7_eUpXzk1HPUuYdhQpXDk30gN13UZSofXzKT017zGuB2aivpoGjzhKKenguBOWrglgWIyHnKnBT2LVwR-JNlTl7KvcSFEJ3rB1uwOosWoZ9GloNKNqlz6z_yxAW2v-Lrnedjcm28H1M8YC5wjORLdRcvC0X9dOsAevDfSaC3jCiKNFOL2XLFfdQiYLVX0r8OQDe3CczWL69revaZuZpyNbiVbBkkggqMDvgM51er81aNNf5-zJ2DQQkLp35VopHi4FIWNKVI8cbVEXFHDYhXnlMLrKqhQOKyZmcwhiYC1Y8Gli2clv6xTpEkva47t8wEPhqHTdkRnf9JdoHcweAEj_HuibsQV8wQjZMXf4C8Qy3hKqjSxBhJdVkjCPgte77RcTYiw6NrV3ETZzQ6Qhnsul_rpSmorfv7lmX4AM5FmCAVeOYr977dTHuS3faNIpRrJhL62syD-4autbmSBeyHRSus3Ppw4Qh3SHazaz6tEU76LCpucvZmmrRbnWFtKjGAg5wLx4CsYrJXqBlpq4wkBJMIU-cBmZU5yfxEn0pLOJYvcNq7YbsyJK9X8l_RPFMHM2Q_TErtXHW7QTdv5ZGCLzVJAqilpumooBpaBRQcp69ExWJl-mWbkwrpIVukOVs0p_yX_s15dj6HqRSAbYOaYa10lSHTMFGfMI_n_ditd9-DxeZJveOxlKWtcZb-5P5E9s1fO_hOk-zUqe9VCdNdzls_2jErn1OebGqYaMFggqOkcCRfXSjB25RAsU3W0pulIenS7nMnP8RRQ_-V-mmpzj9yjOQNHfXspdCvALjvnKf8uYVD1Y3WlFojpmxtE-Ia5ERWOZdWeupNu6JOPsWsSUnha4zWsS38GiAqSPAsWJipMonn7zdux35RlD3MTlDaG3c3OUEegHC6N8Oi6y4ocdyvQ7bvK74URENKrAG3hXMFJUX8Vx5mTdTvxWkVv9rNe9_f6oMfAoUmNWfHMXnaGHukLXI_fQiwCFU828q26grlOJ09abvg0OPvt20L1E-N8ITNypUbS4UzKTgORDODTGHnZ0_SijOH8IzI9eZ6iIve7cyOHG-gAbaV869euDjHfhC3SPY_dVzmwmUrGtip1iB0ZQNhbJPe6Af2X_huDQqC6UU1JZdH9zrKUrpTWD9njJSuXL0xVwbZP4gCINHrWKAa4vqJLTkMqB1KNA7EbIJBJ1j2za--NM"

def test_token():
    """Test Dropbox token authentication."""
    print("🔍 Testing Dropbox token...")
    print(f"Token length: {len(TOKEN)}")
    print(f"Token prefix: {TOKEN[:30]}...")
    print()
    
    # Clean token
    token = TOKEN.strip().strip('"').strip("'").replace("\n", "").replace("\r", "")
    
    if not token.startswith("sl."):
        print(f"❌ Token format invalid - should start with 'sl.' (got: {token[:20]})")
        return False
    
    print("✅ Token format valid")
    print()
    
    # Initialize Dropbox client
    try:
        print("🔌 Initializing Dropbox client...")
        dbx = dropbox.Dropbox(token)
        print("✅ Dropbox client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Dropbox client: {e}")
        return False
    
    # Test authentication
    try:
        print("🔐 Testing authentication...")
        account_info = dbx.users_get_current_account()
        print(f"✅ Authentication successful!")
        print(f"   Account: {account_info.email}")
        print(f"   Name: {account_info.name.display_name}")
        print(f"   Account ID: {account_info.account_id}")
    except AuthError as e:
        print(f"❌ Authentication failed: {e}")
        print()
        print("💡 Possible issues:")
        print("   1. Token is expired - generate a new token")
        print("   2. Token was revoked - check Dropbox app settings")
        print("   3. Token doesn't have required permissions")
        print("   4. Token is for a different app")
        print()
        print("🔧 How to fix:")
        print("   1. Go to https://www.dropbox.com/developers/apps")
        print("   2. Select your app (or create a new one)")
        print("   3. Go to 'Permissions' tab")
        print("   4. Ensure 'files.content.write' and 'files.content.read' are enabled")
        print("   5. Generate a new access token")
        print("   6. Copy the full token (it should start with 'sl.')")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test file operations
    try:
        print()
        print("📁 Testing file operations...")
        test_path = "/test_engine_connection.txt"
        test_content = b"Engine connection test - you can delete this file"
        
        # Upload test file
        dbx.files_upload(
            test_content,
            test_path,
            mode=dropbox.files.WriteMode.overwrite,
        )
        print(f"✅ File upload successful: {test_path}")
        
        # Download test file
        metadata, response = dbx.files_download(test_path)
        print(f"✅ File download successful: {metadata.name}")
        
        # Delete test file
        dbx.files_delete_v2(test_path)
        print(f"✅ File delete successful")
        
    except ApiError as e:
        print(f"❌ File operation failed: {e}")
        print()
        print("💡 Possible issues:")
        print("   1. Token doesn't have 'files.content.write' permission")
        print("   2. Token doesn't have 'files.content.read' permission")
        print("   3. App folder access is restricted")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print()
    print("🎉 All tests passed! Token is valid and working.")
    return True

if __name__ == "__main__":
    success = test_token()
    sys.exit(0 if success else 1)





