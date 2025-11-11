# How The Bot Works - Simple Explanation

## 📖 What This Bot Does (And Doesn't Do)

This bot is an automated trading system that learns from historical market data to make trading decisions. It studies price patterns, calculates features, trains machine learning models, and uses those models to generate trading signals. The bot will attempt to make profitable trades by buying and selling cryptocurrencies based on its learned patterns. However, **trading involves significant risk and losses can and will happen**. The bot is not a guarantee of profits, and past performance does not guarantee future results. You should only trade with money you can afford to lose.

---

## 🎮 Think of the Bot Like a Smart Robot That Learns to Trade

Imagine you have a robot friend that's really good at learning from history. Every day, it wakes up and does a bunch of things to get smarter about trading. Let me tell you what it does, step by step!

---

## 🎯 How It Actually Decides

**Consensus Score (S):** The bot combines votes from multiple trading engines (typically 23 different strategies) into a single consensus score that ranges from negative (sell) to positive (buy).

**Confidence:** Each decision has a confidence level (0-100%) based on how strongly the engines agree and how reliable they've been recently.

**Threshold:** The bot only trades when the consensus score exceeds a threshold that adapts to market volatility - typically it needs to be at least 0.75 times the recent volatility of consensus scores.

---

## ⚠️ Risk Rules

**Per Trade Risk:** Each trade risks a small percentage of total equity (typically 0.5-1.5%, configurable).

**Daily Stop:** If losses exceed a daily limit (typically 2.5% of equity), trading stops for the day.

**Max Leverage:** Total exposure is capped at a maximum leverage (typically 2-3x, configurable).

**Exposure Caps:** No single coin can exceed 25% of equity, and total open risk (sum of stop losses) cannot exceed the daily loss limit.

---

## 💰 Costs

**Fees:** Every trade pays exchange fees (typically 2-5 basis points for maker orders, 4-5 bps for taker orders).

**Spread:** The difference between buy and sell prices costs money (typically 1-10 basis points).

**Funding:** Trading perpetual contracts incurs funding costs (typically 1 basis point per 8 hours).

**Slippage:** Large orders move prices, causing additional costs (varies with order size and market conditions).

**The bot only trades when the expected edge (profit) beats all these costs by a safety margin (typically 3 basis points).**

---

## 🌅 **FIRST THING: Wake Up and Get Ready!**

**What happens:**
1. The bot wakes up and says "Hello! Time to work!"
2. It creates a special folder for today's date where it will save all its work
3. It turns on its background helpers that automatically save things

**Why:** So everything it does today gets saved in the right place!

---

## 📚 **STEP 2: Get Its Tools Ready**

**What happens:**
1. It loads its settings (like reading a recipe book)
2. It connects to the exchange (like connecting to a store where it buys/sells coins)
3. It connects to the database (like opening its memory book)
4. It sets up Telegram notifications (so it can text you updates!)

**Why:** Just like you need your backpack and supplies ready before school!

---

## 🪙 **STEP 3: Pick Which Coins to Study**

**What happens:**
1. The bot looks at ALL the coins available
2. It picks the best coins to study (typically around 20, configurable)
   - It picks coins that:
     - Have lots of trading (popular)
     - Don't cost too much to trade
     - Are easy to buy and sell

**Why:** It can't study everything, so it picks the best ones!

---

## 📥 **STEP 4: Download Coin History (Like Getting Old Photos)**

**What happens:**
1. For each coin (like BTC, ETH, etc.), the bot downloads price history (typically the last 120-150 days, configurable)
   - It's like getting photos showing what the price did each day
2. It saves this data to its computer

**Why:** The bot needs to see what happened in the past to learn what might happen in the future!

---

## 🧮 **STEP 5: Calculate Features (Like Making a Report Card)**

**What happens:**
1. The bot looks at all the price data and calculates many different "features" (typically 50+, configurable)
   - Features are like measurements:
     - "Is the price going up or down?" (trend)
     - "Is it moving fast or slow?" (volatility)
     - "Is it a good time to buy?" (momentum)
     - And many more things!
2. It's like making a report card with many grades for each moment in time

**Why:** The bot needs to understand what's happening, not just see the numbers!

---

## 🎓 **STEP 6: Learn and Test**

**What happens:**
1. The bot practices trading on old data (like playing a video game with old levels) to see what would have happened
2. It learns patterns from these practice trades and creates a "model" (like a brain) that remembers what works
3. It tests the model on new data it hasn't seen before to make sure it's good enough
4. Only models that pass the tests get saved for use

**Why:** Practice makes perfect! The bot only uses models that actually work!

---

## 💾 **STEP 7: Save Everything**

**What happens:**
1. If the model passed, the bot saves it along with all the information about how good it is
2. It also saves all the trades it practiced, all the things it learned, and all the reports

**Why:** So the trading system can use these models to make real trades!

---

## 🔄 **STEP 8: Keep Syncing (Background Helpers)**

**What happens:**
1. While the bot is working, helper robots are running in the background
2. They automatically save things periodically (typically every 5-30 minutes, configurable)

**Why:** So nothing gets lost, even if something goes wrong!

---

## 🏥 **STEP 9: Safety Checks**

**What happens:**
1. **Health Checks:** The bot continuously checks if everything is working (database, exchange connection, services)
2. **Circuit Breakers:** If losses get too high (daily stop) or there are too many losses in a row, trading automatically stops or reduces size
3. **Kill Switch:** In emergencies, a kill switch can immediately stop all trading and cancel all orders

**Why:** Safety first! The bot protects itself and your money from runaway losses!

---

## 📱 **STEP 10: Send Updates (Tell You What Happened)**

**What happens:**
1. The bot sends you messages on Telegram:
   - "I'm starting!"
   - "I'm training on BTC..."
   - "I finished! Here's what I learned!"
   - "I created X models, Y passed, Z failed"

**Why:** So you know what the bot is doing, even when you're not watching!

---

## 🎉 **LAST THING: Clean Up and Finish!**

**What happens:**
1. The bot saves the log file (a diary of everything it did)
2. It makes sure background sync is still running
3. It shuts down extra workers
4. It says "Done! See you tomorrow!"

**Why:** Clean up after yourself, just like putting toys away!

---

## 📋 **Summary: What Happens in Order**

1. **Wake up** → Create folder for today's work
2. **Get ready** → Load settings, connect to everything
3. **Pick coins** → Choose best coins to study (typically ~20)
4. **Download data** → Get price history (typically 120-150 days)
5. **Calculate features** → Make many measurements (typically 50+)
6. **Learn and test** → Practice trading, learn patterns, test models
7. **Save model** → Save the brain if it passed tests
8. **Keep syncing** → Background helpers keep saving things
9. **Safety checks** → Health checks, circuit breakers, kill switch
10. **Send updates** → Text you on Telegram about what happened
11. **Clean up** → Finish and get ready for tomorrow!

---

## 🎯 **The Big Picture**

Think of it like this:

1. **The bot is like a student** 📚
   - It studies old data (like reading history books)
   - It practices (like doing homework)
   - It learns patterns (like memorizing facts)
   - It takes a test (to see if it learned well)

2. **The model is like a report card** 📊
   - It shows what the bot learned
   - It shows how good the bot is
   - If it's good enough, the trading system uses it!

3. **The trading system is like the real player** 🎮
   - It uses the models the bot created
   - It makes real trades with real money
   - It's like the bot is the coach, and the trading system is the player!

---

## ⏰ **How Long Does It Take?**

- **Total time:** Typically 1-2 hours (varies with number of coins and models)
- **Most time spent on:** Downloading coin data and training models
- **Runs:** Once per day (typically at 2:00 AM UTC, configurable)

---

## 📊 Outcomes We Track

**Win Rate:** The percentage of trades that make money (typically tracked per engine and overall).

**Sharpe Ratio:** A measure of risk-adjusted returns - higher is better (typically we aim for > 1.0).

**Max Drawdown:** The largest peak-to-trough decline in equity - lower is better (typically we limit to 2.5-3.5% daily).

**Cost Share:** What percentage of gross profits are eaten by costs (fees, slippage, funding) - lower is better (typically 20-40%).

---

## 🎉 **That's It!**

The bot is like a smart student that:
- Studies every day
- Learns from history
- Practices trading
- Creates models (brains)
- Saves everything
- Tells you what it did
- Stays safe with circuit breakers and kill switches

And then the trading system uses those models to make real trades! 🚀

**Remember: Trading involves risk. Losses can and will happen. Only trade with money you can afford to lose.**
