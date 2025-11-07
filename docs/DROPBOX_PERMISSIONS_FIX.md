# Fix Dropbox Permissions

## Issue
Your Dropbox app is missing required permissions:
- `files.content.write` - needed to create folders and upload files
- `files.metadata.read` - needed to read metadata and list folders

## Solution

1. **Go to Dropbox App Console:**
   - Visit: https://www.dropbox.com/developers/apps
   - Sign in with your Dropbox account

2. **Find Your App:**
   - Look for your app (ID: 7988481)
   - Click on your app to open its settings

3. **Enable Required Permissions:**
   - Go to the **"Permissions"** tab
   - Find the following scopes and **enable** them:
     - ✅ `files.content.write` - Write files
     - ✅ `files.metadata.read` - Read file metadata
   - Click **"Submit"** to save the changes

4. **Generate New Access Token:**
   - After enabling permissions, you need a new token
   - Go to the **"Settings"** tab
   - Scroll down to **"OAuth 2"** section
   - Find **"Generated access token"**
   - Click **"Generate"** button (or regenerate if one exists)
   - **Copy the entire new token** (it will be very long, 1000+ characters)
   - The token should start with `sl.`

5. **Update the Token:**
   - Update the token in the code or set it as environment variable:
     ```bash
     export DROPBOX_ACCESS_TOKEN="your_new_token_here"
     ```
   - Or update it in `src/cloud/training/config/settings.py`

6. **Restart the Engine:**
   - After updating the token, restart the engine
   - The Dropbox sync should now work properly

## Required Permissions Summary

| Permission | Purpose | Required |
|------------|---------|----------|
| `files.content.write` | Create folders, upload files | ✅ Yes |
| `files.metadata.read` | Read metadata, list folders | ✅ Yes |

## After Fixing

Once permissions are enabled and a new token is generated:
- ✅ Dated folder creation will work
- ✅ File uploads will work
- ✅ Directory syncing will work
- ✅ Data cache restore will work

