# Description: Streamlit app that fetches daily stock data, aggregates it to weekly (Friday close),
# builds features, trains a model based on 1 of 2 methods:  (Logistic Regression or Random Forest)
# to predict whether NEXT week will close Up or Down, and produces a concise weekly report.

import math
import io
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import streamlit as st

# =============================
# Utility: Technical Indicators
# =============================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index computed on price differences.
    Assumes equally spaced samples (weekly here)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def weekly_ohlcv(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to weekly (Friday close)."""
    weekly = pd.DataFrame()
    weekly['Open'] = df_daily['Open'].resample('W-FRI').first()
    weekly['High'] = df_daily['High'].resample('W-FRI').max()
    weekly['Low'] = df_daily['Low'].resample('W-FRI').min()
    weekly['Close'] = df_daily['Close'].resample('W-FRI').last()
    weekly['Volume'] = df_daily['Volume'].resample('W-FRI').sum()
    weekly.dropna(how='any', inplace=True)
    return weekly


def build_features(weekly: pd.DataFrame) -> pd.DataFrame:
    df = weekly.copy()

    # Returns
    df['ret_1w'] = df['Close'].pct_change(1)
    df['ret_2w'] = df['Close'].pct_change(2)
    df['ret_4w'] = df['Close'].pct_change(4)

    # Moving averages & ratios
    df['ma_4'] = df['Close'].rolling(4).mean()
    df['ma_12'] = df['Close'].rolling(12).mean()
    df['ma_ratio_4_12'] = df['ma_4'] / (df['ma_12'] + 1e-12)

    # Volatility (rolling std of weekly returns)
    df['vol_4'] = df['ret_1w'].rolling(4).std()

    # RSI & MACD (on weekly Close)
    df['rsi_14'] = rsi(df['Close'], 14)
    macd_line, signal_line, macd_hist = macd(df['Close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_hist

    # OBV (weekly)
    dirn = np.sign(df['Close'].diff()).fillna(0)
    df['obv'] = (dirn * df['Volume']).cumsum()
    df['obv_slope_4'] = df['obv'].diff(4) / 4.0

    # Target: next week's direction
    df['target_up'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop initial NaNs from rolling calcs
    df = df.dropna().copy()

    # Feature set
    features = [
        'ret_1w','ret_2w','ret_4w',
        'ma_ratio_4_12','vol_4',
        'rsi_14','macd','macd_signal','macd_hist',
        'obv_slope_4'
    ]
    return df, features


# =============================
# Data
# =============================
@st.cache_data(show_spinner=False)
def load_daily(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, interval='1d')
    if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
        df.index = pd.to_datetime(df.index)
        return df[['Open','High','Low','Close','Volume']]
    return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])


# =============================
# Modeling
# =============================
@dataclass
class ModelResult:
    model_name: str
    metrics: dict
    latest_pred: int
    latest_proba: float
    feature_importance: pd.Series | None
    backtest: pd.DataFrame


def time_series_cv_scores(X: pd.DataFrame, y: pd.Series, model) -> dict:
    tscv = TimeSeriesSplit(n_splits=5)
    acc, prec, rec, f1s = [], [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        acc.append(accuracy_score(y_te, y_pred))
        prec.append(precision_score(y_te, y_pred, zero_division=0))
        rec.append(recall_score(y_te, y_pred, zero_division=0))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
    return {
        'accuracy': float(np.mean(acc)),
        'precision': float(np.mean(prec)),
        'recall': float(np.mean(rec)),
        'f1': float(np.mean(f1s))
    }


def fit_and_report(df_feat: pd.DataFrame, features: list[str], model_choice: str) -> ModelResult:
    X = df_feat[features]
    y = df_feat['target_up']

    if model_choice == 'Logistic Regression':
        base_model = LogisticRegression(max_iter=2000, n_jobs=None)
    else:
        base_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    metrics = time_series_cv_scores(X, y, base_model)

    # Fit on full history up to the most recent complete week 
    # So we exclude the last row when training to avoid peeking.
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', base_model)
    ])
    pipe.fit(X_train, y_train)

    # Predict the NEXT week using the last row of features
    X_latest = X.iloc[[-1]]
    proba = 0.5
    if hasattr(pipe.named_steps['clf'], 'predict_proba'):
        proba = float(pipe.predict_proba(X_latest)[0,1])
    latest_pred = int(1 if proba >= 0.5 else 0)

    # Backtest 
    preds, probas, dates = [], [], []
    start_idx = max(30, int(0.3 * len(X)))
    for i in range(start_idx, len(X)-1):  # predict i+1 using history up to i
        X_tr, y_tr = X.iloc[:i], y.iloc[:i]
        X_te = X.iloc[[i]]
        p = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', base_model)
        ])
        p.fit(X_tr, y_tr)
        if hasattr(p.named_steps['clf'], 'predict_proba'):
            prob = float(p.predict_proba(X_te)[0,1])
        else:
            # approximate probability from decision_function if available
            if hasattr(p.named_steps['clf'], 'predict'):
                prob = float(p.predict(X_te)[0])
            else:
                prob = 0.5
        pred = 1 if prob >= 0.5 else 0
        preds.append(pred)
        probas.append(prob)
        dates.append(X.index[i])

    bt = pd.DataFrame({'pred': preds, 'proba_up': probas}, index=pd.to_datetime(dates))
    bt['actual'] = y.loc[bt.index]
    bt['correct'] = (bt['pred'] == bt['actual']).astype(int)

    # Feature importance (for random forest) or coefficients (for logarithmic)
    fi = None
    clf = pipe.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        fi = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    elif hasattr(clf, 'coef_'):
        coefs = clf.coef_[0]
        fi = pd.Series(coefs, index=features).sort_values(key=np.abs, ascending=False)

    return ModelResult(
        model_name=model_choice,
        metrics=metrics,
        latest_pred=latest_pred,
        latest_proba=proba,
        feature_importance=fi,
        backtest=bt
    )


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Weekly Stock Direction Predictor", layout="wide")

st.title("📈 Weekly Stock Direction Predictor")
st.caption("Predict whether a stock will close *Up or Down* next week based on weekly technicals.")

# Sidebar Controls
st.sidebar.header("Controls")

def_default_ticker = "AAPL"

def_end = dt.date.today()
def_start = def_end - dt.timedelta(days=365*10)  # 10 years default

ticker = st.sidebar.text_input("Ticker (Yahoo Finance symbol)", value=def_default_ticker).strip().upper()
start_date = st.sidebar.date_input("Start date", value=def_start)
end_date = st.sidebar.date_input("End date", value=def_end)
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"], index=0)

min_hist_weeks = st.sidebar.slider("Minimum weeks of history", min_value=60, max_value=520, value=156, step=4,
                                   help="Require at least this many weekly rows before modeling.")

st.sidebar.markdown("---")
st.sidebar.write("**Weekly report** shows last 12 weeks plus the next-week prediction.")

# Load & Prepare Data
if ticker:
    with st.spinner("Downloading & preparing data..."):
        daily = load_daily(ticker, start_date, end_date)
        if daily.empty:
            st.error("No data returned. Check the ticker or date range.")
            st.stop()
        weekly = weekly_ohlcv(daily)
        if weekly.shape[0] < min_hist_weeks:
            st.warning(f"Not enough weekly history (have {weekly.shape[0]}, need {min_hist_weeks}). Try extending the date range.")
            st.stop()
        df_feat, features = build_features(weekly)
        if df_feat.shape[0] < min_hist_weeks:
            st.warning("Insufficient rows after feature engineering. Try a longer date range.")
            st.stop()

    # Fit & Evaluate
    res = fit_and_report(df_feat, features, model_choice)

    # Top row: Latest Prediction Card
    latest_week = df_feat.index[-1].date()
    next_week_end = (df_feat.index[-1] + pd.offsets.Week(weekday=4)).date()  # next Friday
    pred_text = "UP" if res.latest_pred == 1 else "DOWN"

    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1.2])
    with col1:
        st.subheader("Next Week Direction")
        st.metric(label=f"{ticker} by {next_week_end}", value=pred_text, delta=f"P(up) = {res.latest_proba:.2%}")
    with col2:
        st.subheader("CV Accuracy")
        st.metric(label="Mean", value=f"{res.metrics['accuracy']:.1%}")
    with col3:
        st.subheader("CV F1")
        st.metric(label="Mean", value=f"{res.metrics['f1']:.1%}")
    with col4:
        st.subheader("Model")
        st.write(res.model_name)

    # Price chart (weekly)
    st.markdown("### Price (weekly closes)")
    price_df = weekly[['Close']].copy()
    price_df.columns = [f"{ticker} Close"]
    st.line_chart(price_df)

    # Feature importance/coefficients
    if res.feature_importance is not None:
        st.markdown("### Feature importance / coefficients")
        st.bar_chart(res.feature_importance)

    # Backtest summary
    st.markdown("### Rolling backtest (expanding window)")
    left, right = st.columns(2)
    with left:
        bt_acc = res.backtest['correct'].mean() if res.backtest.shape[0] else float('nan')
        st.metric("Backtest Hit Rate", f"{bt_acc:.1%}" if not math.isnan(bt_acc) else "n/a")
        st.caption("Out-of-sample directional accuracy using an expanding-window validation.")
    with right:
        st.line_chart(res.backtest[['proba_up']])

    # Weekly Report (last 12 weeks + next week's prediction)
    st.markdown("### Weekly report")
    report = df_feat[['Close','ret_1w','rsi_14','macd_hist']].copy()
    report['actual_this_week'] = df_feat['target_up']  # whether THIS week closed up vs prior week
    # Align predictions from backtest to the week they predict
    aligned = res.backtest[['pred']].copy()
    aligned.index = aligned.index + pd.offsets.Week(weekday=4)  # they predict next Friday
    report = report.join(aligned.rename(columns={'pred':'model_pred_next'}), how='left')

    # Add final next-week live prediction on the last row
    report.loc[df_feat.index[-1], 'model_pred_next'] = res.latest_pred

    # Tidy labels
    out = report.tail(12).copy()
    map_ud = {1: 'UP', 0: 'DOWN', np.nan: ''}
    out['actual_this_week'] = out['actual_this_week'].map(map_ud)
    out['model_pred_next'] = out['model_pred_next'].map(map_ud)
    out = out.rename(columns={
        'Close': 'Close (Fri)',
        'ret_1w': 'Return 1W',
        'rsi_14': 'RSI(14)',
        'macd_hist': 'MACD Hist',
        'actual_this_week': 'This week vs last',
        'model_pred_next': 'Predict next week'
    })
    out = out[['Close (Fri)', 'Return 1W', 'RSI(14)', 'MACD Hist', 'This week vs last', 'Predict next week']]

    st.dataframe(out.style.format({
        'Close (Fri)': '{:,.2f}',
        'Return 1W': '{:+.2%}',
        'RSI(14)': '{:.1f}',
        'MACD Hist': '{:+.4f}'
    }))

    # Download button for CSV report
    csv = out.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="⬇️ Download weekly report (CSV)",
        data=csv,
        file_name=f"{ticker}_weekly_report.csv",
        mime='text/csv'
    )

    st.markdown("---")
    st.caption("Educational use only — not financial advice. Models are probabilistic and can be wrong.")
