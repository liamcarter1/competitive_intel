# Rate Limiting Implementation

## What Was Added

To prevent API cost abuse from public users, the app now has IP-based rate limiting on chat features.

## Limits

- **Quick Chat**: 10 requests per hour per IP address
- **Deep-dive Research**: 3 requests per hour per IP address
- **Briefing/Annual Report Generation**: No limits (requires explicit action)

## How It Works

1. **IP Tracking**: Each user's IP address is tracked
2. **Hourly Window**: Limits reset every 60 minutes from first request
3. **Per-Feature**: Chat and deep-dive have separate quotas
4. **User Feedback**: Users see remaining quota after each request

## User Experience

### Normal Usage
- User asks a question
- Response appears normally
- Status shows: "✅ 7/10 chat requests remaining this hour"

### Rate Limit Hit
- User tries request #11
- Response shows: "⏱️ Rate limit reached. You've used 10/10 chat requests this hour. Try again in a few minutes."
- Resets automatically after 60 minutes

## Cost Impact

### Before Rate Limiting
- 14,508 requests in 24 hours = $18
- ~$540/month if sustained

### After Rate Limiting (Per User)
- Max 10 chats/hour × 24 hours = 240 chats/day
- Max 3 deep-dives/hour × 24 hours = 72 deep-dives/day
- ~$0.24 + $7.20 = **$7.44/user/day** max

### With 10 Active Users
- 10 users × $7.44 = **~$74/day** worst case
- Likely much lower (users won't max out daily)
- Typical: **$10-20/day** with legitimate usage

## Adjusting Limits

Edit these values in `app.py` line 29-30:

```python
RATE_LIMIT_CHAT = 10      # Increase/decrease quick chat limit
RATE_LIMIT_DEEPDIVE = 3   # Increase/decrease deep-dive limit
```

## Monitoring

Check your HF Space analytics to see:
- Total requests per day
- Peak usage times
- Whether limits need adjustment

If costs are still too high, consider:
1. Lowering limits (e.g., 5 chats/hour instead of 10)
2. Making Space private
3. Adding authentication requirement
