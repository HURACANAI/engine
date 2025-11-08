# RunPod Dropbox Setup Guide

## Quick Setup

### Option 1: Environment Variable (Recommended)

In your RunPod pod, set the `DROPBOX_ACCESS_TOKEN` environment variable:

```bash
export DROPBOX_ACCESS_TOKEN="sl.u.AGHQ_GZ-nAHejWf16c-t38yKlUaPa7jo9QK7Hrb0RYHPSl5ElgRLq2eoFVt3MTp1pf8vXo7i6--E4c-8Y7_GmnvwvTFE7FdsTzjLdYd9-tIVbtzPtyDr8Y0Eo4qgJ4kmWqMGkowVsRKVJhXA23sMwJrvbNO4vqpGHK5n1wDY9bwyOg1t2uF7CgI0kWW8ftRqkzl8iYCMkPIumkaFwE2kuKn7qbQ0Gqd3LK6s1yFq_DLFoXg8m77Ji_m8m6EpOSdGQuPkfVtOQn2qs9SRtI17KlDHA27dlcpu2-6eiebCafHYZ_l4CvkRCERohITkcAuopSsXhrI9qcEI6NzfwRgrb5NSYHJdTUI-7dzHEMJfLBg4YCBmuetm_lFOS2iMhBZoVQWj2kV9VGvF8FWtjBGCtGnPGb5a69tXcFUPudKDGmWuF4WmpQDLhBzxPCdeo2n-yBV8QNoZ9Qn5pwwfJIPozafSQI6qOr4sSwxGOq_DBQ3QaLTlgy8JGFAxK2f-khqAWfxMdmXvX3FKf-D5cUnjbC5gWGn-IaCAWgQeemncPVLUm5I3fA9_Q12IAJgbhnXZbT-CDj1LdzVo1NyIAIMM7bZFaTkIHquKZPQ9BARkYlznPMJpkzQ4IjnO3M63ne4ZRLFOUwLbVAhVvewPPA6VqotbFG5VnP2eIoJn3W4wcKNRJqPAye7JhUlNUgaJwH8RjZf0YiLUVmiSxhbXGq-7U0jpustKnMhs56rG59gIHaApvWlCWBNGOGqhcvZxi-egZX9m_vQT6jf0tunyb9szmhC7rvaV-oQth1dobmJ10EN-fiKHcmzhlPHzNz1pEhz83gD-WSrfqJr4SgL6nFqmbioNGkQTNqlvxGdMq92qfu9DUL4XN0i-z7SIolqEcBZge5RdPNtBHQCW4osiU0i3FabFWbt6lJIa8-71r93hBgfY4F6uEGVPSfDU0ECEbe0ONAS-hpBXswb33TzeNeKiGGPl2q-eauqbFb1wkLKeghV07Uv3H9BjPHaI2hGW4C9GdKHYiLUphHY0_pYt5xW2l9Ka21N2N8IWkibNT0M4_up4aIRAgTrnYFuSfGreHv8ZzW4PoQ6Q0tx7kjtGbpW7Ej-rzSLptO1-Nu5PNXEBsWbEdQWFrGVNwEfbZ61pdHJ8cV2yCFYlwzW4iGSo4fsE4MSDh6GtUvwlJP9HEPiDmMvQ_xAeMH-Mxa9Oi_pd-G6h6CzuD5jvxh_wNl5eFtGOxJLHWvSTbJ5ls5PjUb1XuTDH6jmHLtSmFfbWEu6Hg0PpKx4"
```

### Option 2: RunPod Environment Variables (Persistent)

1. Go to your RunPod pod settings
2. Navigate to "Environment Variables"
3. Add:
   - **Key:** `DROPBOX_ACCESS_TOKEN`
   - **Value:** `sl.u.AGHQ_GZ-nAHejWf16c-t38yKlUaPa7jo9QK7Hrb0RYHPSl5ElgRLq2eoFVt3MTp1pf8vXo7i6--E4c-8Y7_GmnvwvTFE7FdsTzjLdYd9-tIVbtzPtyDr8Y0Eo4qgJ4kmWqMGkowVsRKVJhXA23sMwJrvbNO4vqpGHK5n1wDY9bwyOg1t2uF7CgI0kWW8ftRqkzl8iYCMkPIumkaFwE2kuKn7qbQ0Gqd3LK6s1yFq_DLFoXg8m77Ji_m8m6EpOSdGQuPkfVtOQn2qs9SRtI17KlDHA27dlcpu2-6eiebCafHYZ_l4CvkRCERohITkcAuopSsXhrI9qcEI6NzfwRgrb5NSYHJdTUI-7dzHEMJfLBg4YCBmuetm_lFOS2iMhBZoVQWj2kV9VGvF8FWtjBGCtGnPGb5a69tXcFUPudKDGmWuF4WmpQDLhBzxPCdeo2n-yBV8QNoZ9Qn5pwwfJIPozafSQI6qOr4sSwxGOq_DBQ3QaLTlgy8JGFAxK2f-khqAWfxMdmXvX3FKf-D5cUnjbC5gWGn-IaCAWgQeemncPVLUm5I3fA9_Q12IAJgbhnXZbT-CDj1LdzVo1NyIAIMM7bZFaTkIHquKZPQ9BARkYlznPMJpkzQ4IjnO3M63ne4ZRLFOUwLbVAhVvewPPA6VqotbFG5VnP2eIoJn3W4wcKNRJqPAye7JhUlNUgaJwH8RjZf0YiLUVmiSxhbXGq-7U0jpustKnMhs56rG59gIHaApvWlCWBNGOGqhcvZxi-egZX9m_vQT6jf0tunyb9szmhC7rvaV-oQth1dobmJ10EN-fiKHcmzhlPHzNz1pEhz83gD-WSrfqJr4SgL6nFqmbioNGkQTNqlvxGdMq92qfu9DUL4XN0i-z7SIolqEcBZge5RdPNtBHQCW4osiU0i3FabFWbt6lJIa8-71r93hBgfY4F6uEGVPSfDU0ECEbe0ONAS-hpBXswb33TzeNeKiGGPl2q-eauqbFb1wkLKeghV07Uv3H9BjPHaI2hGW4C9GdKHYiLUphHY0_pYt5xW2l9Ka21N2N8IWkibNT0M4_up4aIRAgTrnYFuSfGreHv8ZzW4PoQ6Q0tx7kjtGbpW7Ej-rzSLptO1-Nu5PNXEBsWbEdQWFrGVNwEfbZ61pdHJ8cV2yCFYlwzW4iGSo4fsE4MSDh6GtUvwlJP9HEPiDmMvQ_xAeMH-Mxa9Oi_pd-G6h6CzuD5jvxh_wNl5eFtGOxJLHWvSTbJ5ls5PjUb1XuTDH6jmHLtSmFfbWEu6Hg0PpKx4`

### Verification

After setting the token, verify it works:

```bash
cd /workspace/engine
export HURACAN_ENV=runpod
./run.sh
```

You should see:
- ✅ `dropbox_token_source: environment_variable`
- ✅ `dropbox_dated_folder_created_successfully`

## Dropbox App Credentials

**App Key:** `2nflb2iihy16hcd`  
**App Secret:** `eolbcn04hfswjgy`

Store these securely. You may need them to generate new tokens in the future.

## Token Priority

The engine checks for tokens in this order:
1. **Environment variable** `DROPBOX_ACCESS_TOKEN` (highest priority)
2. Settings file `settings.dropbox.access_token`
3. Hardcoded fallback (updated in code)

## Troubleshooting

### Token Expired Error
If you see `expired_access_token`:
1. Generate a new token at https://www.dropbox.com/developers/apps
2. Update the `DROPBOX_ACCESS_TOKEN` environment variable
3. Restart the engine

### Token Invalid Error
If you see `invalid_access_token`:
- Check that the entire token was copied (tokens are 1000+ characters)
- Ensure no extra spaces or quotes
- Verify token starts with `sl.`

### Bot Won't Start
If Dropbox is enabled and initialization fails, the bot will stop immediately. This is intentional to prevent data loss. Either:
1. Fix the Dropbox token, OR
2. Disable Dropbox in settings (not recommended for production)

