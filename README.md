# S&P 500 Prediction – Weekly Direction & LSTM Experiments

I built this project at the Penn State **Office of Investment Management (OIM)** as part of my personal projects. I mostly reported to **Trajan Robertson** (Investment Analyst). The project was overseen by **Jocelin Reed** (Managing Director of Investments), and **Joseph Cullen**, Penn State’s Chief Investment Officer, reviewed my work as part of our process.

I originally proposed a fund-performance model to rank funds and predict which ones would lead future performance. We didn’t have enough clean, long-horizon fund data to support a robust model, and the team wasn’t sure it would be immediately useful. So I pivoted—on my manager’s recommendation—to **S&P 500 prediction**, alongside a few other stocks. This aligned with our **weekly market discussion** meetings where we unpacked tax, trade, and market policy shifts, and broader developments that shape the next few weeks. Weekly direction turned out to be the tightest fit for how the team actually works.

---

## What’s here

* **`app/streamlit_app.py`** — a Streamlit app that:

  * fetches **daily** data (Yahoo Finance), aggregates to **weekly (Friday close)**,
  * builds features (momentum, trend, volatility, volume),
  * trains either **Logistic Regression** or **Random Forest**,
  * predicts **next week**: **UP** or **DOWN**, and
  * outputs a compact weekly report.

* **`experiments/lstm_practice.py`** — my first LSTM model (daily close, lookback=60), mainly to sanity-check a deep time-series approach before adapting ideas to the weekly framing.

---

## Why weekly?

Daily returns are noisy; aggregating to weekly smooths microstructure noise and makes the signal more relevant to our workflow. A growing body of work studies forecast horizons vs. input windows and shows that predictive performance depends on **both** choices; my emphasis on a weekly horizon reflects that tradeoff between information and noise, and it matches the cadence of our discussions. ([irep.ntu.ac.uk][1])

---

## Features & targets (how I engineered the signal). Sources provide insight into my research and reasons for some choices. 

For the weekly classifier (Logistic/Random Forest), I engineered a compact set of indicators that cover **momentum, trend, volatility, and volume**:

* **Returns & momentum:** 1–4 week returns, RSI(14). RSI is widely used for gauging overbought/oversold conditions; it complements trend measures and can add information in range-bound markets. ([Investopedia][2])
* **Trend:** 4- and 12-week moving averages and their ratio; **MACD** (12/26 with 9-signal) for momentum/trend shifts. MACD captures changes in the relationship of fast vs. slow EMAs and is a staple in trend assessment. ([Investopedia][3])
* **Volatility:** 4-week rolling std of weekly returns (helps distinguish quiet vs. high-variance regimes). Empirically, volatility and regime interact with predictability. ([arXiv][4])
* **Volume:** **On-Balance Volume (OBV)** and a short-slope over 4 weeks as a simple volume-price pressure proxy; OBV has documented profitability in certain markets and is a canonical volume indicator. ([IDEAS/RePEc][5])

**Target:** binary next-week direction (close_t+1 > close_t). This keeps the output aligned to how the team frames risk (“do we lean bullish or bearish next week?”).

---

## Model choices (and why)

### 1) Logistic Regression (baseline, transparent)

I wanted a simple, interpretable baseline where coefficients tell me whether the engineered signals align with intuition. It also checks if a **linear decision boundary** already gets you most of the way there.

### 2) Random Forest (non-linear, robust)

Random Forests capture non-linear interactions among indicators, are relatively robust to outliers, and have a strong track record in directional classification for equities. Multiple studies (classic to recent) show RF to be competitive for stock direction tasks, especially with technical features. ([arXiv][6])

### 3) LSTM (sequence modeling, tested in practice script)

I first tried a **daily LSTM** to get a feel for sequential dependence and whether a deep model would add value. LSTMs are designed for temporal dependencies and have repeatedly performed well for short-term financial prediction when the sequence length and inputs are tuned to the data’s volatility and regime. ([sciencedirect.com][7])

---

## The “why” behind my personal decisions 

**(a) Using technical indicators at all.**
Technical indicators are not magic; they summarize patterns in price/volume (trend, momentum, volatility) that can be predictive in certain regimes. Reviews and empirical papers continue to explore this space, and—even with caveats about overfitting—show contexts where indicator-based models add value. ([research.ed.ac.uk][8])

**(b) Exact indicator set (RSI, MACD, MAs/Bollinger-style bands, OBV).**
These are among the most studied and widely deployed indicators; comparative work evaluates RSI/MACD/Bollinger across assets and finds setting-dependent value. Volume-based signals like OBV capture accumulation/distribution that price alone can miss. ([Journal of Marketing & Social Research][9])

**(c) Weekly aggregation (Friday close) and forecast horizon.**
The combination of **input window** and **forecast horizon** matters materially for performance; using weekly closes reduces noise and aligns with our decision cadence. I’m explicitly trading some immediacy for signal stability and process fit. ([irep.ntu.ac.uk][1])

**(d) Lookback lengths for the LSTM.**
Sequence length is not arbitrary; several studies show accuracy can improve with longer lookbacks (e.g., 100 days) when volatility/trends warrant it—consistent with the idea that richer temporal context helps the model learn. ([Preprints][10])

**(e) Using Random Forest beside Logistic.**
Beyond interpretability, side-by-side baselines guard against overfitting illusions. RF often outperforms simple linear models in classification of stock direction with technical features, while not assuming stationarity in coefficients. ([arXiv][6])

**(f) Why start daily with LSTM, then speak weekly.**
Daily training is an easy proving ground for architecture and data plumbing; once validated, I roll insights into the **weekly** framing the team actually uses. Literature supports that horizon choice changes signal-to-noise and can move the needle on predictability. ([irep.ntu.ac.uk][1])

---

## Streamlit, because stakeholders matter

I chose **Streamlit** so the investment team could run the model without wrestling with notebooks: select a ticker, pick a date range, choose Logistic vs. Random Forest, then get a clear **“Next week: UP/DOWN”** with probability, basic CV metrics, feature importance, and a downloadable 12-week report. The point wasn’t to build a trading engine; it was to make discussion sharper and more grounded in data.

---

## Reproducibility (how to run)

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
# or
python experiments/lstm_practice.py
```

**Data source:** Yahoo Finance (via `yfinance`).
**Train/test:** TimeSeriesSplit for cross-validation; an expanding-window backtest in the app to approximate out-of-sample behavior.

---

## Limitations & ethics (reality check)

Markets are non-stationary. Technical signals come and go with regimes. Backtests can overstate edge, and small samples can mislead. Treat outputs as **probabilistic**, not promises. Nothing here is investment advice. Papers themselves caution about overfitting and generalization—even when indicators or ML show promise. ([arXiv][11])

---

## Acknowledgements

Trajan Robertson for day-to-day guidance, Jocelin Reed for oversight, and Joseph Cullen for engaging with the work. Weekly meetings drove this project’s shape; I built the models to serve those conversations—not the other way around.

---

[1]: https://irep.ntu.ac.uk/id/eprint/32787/1/PubSub10294_702a_McGinnity.pdf? "Forecasting Price Movements using Technical Indicators"
[2]: https://www.investopedia.com/terms/r/rsi.asp? "Relative Strength Index (RSI): What It Is, How It Works, and Formula"
[3]: https://www.investopedia.com/terms/m/macd.asp? "What Is MACD?"
[4]: https://arxiv.org/html/2510.03236v1? "Improving S&P 500 Volatility Forecasting through Regime- ..."
[5]: https://ideas.repec.org/a/ebl/ecbull/eb-09-00423.html? "Profitability of the On-Balance Volume Indicator"
[6]: https://arxiv.org/abs/1605.00003? "Predicting the direction of stock market prices using random forest"
[7]: https://www.sciencedirect.com/science/article/pii/S1566253524003944? "Data-driven stock forecasting models based on neural ..."
[8]: https://www.research.ed.ac.uk/files/16837254/Jacobsen_SSRN_id2449344.pdf? "Technical Market Indicators: An Overview"
[9]: https://www.jmsr-online.com/article/download/pdf/82/? "A Comparative Study of Bollinger Bands, RSI and MACD ..."
[10]: https://www.preprints.org/manuscript/202501.1424? "Dynamic Optimisation of Window Sizes for Enhanced Time ..."
[11]: https://arxiv.org/html/2412.15448v1? "Assessing the Impact of Technical Indicators on Machine ..."
