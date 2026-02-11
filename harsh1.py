
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os
import warnings
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations (silences logs)
# Quiet known third-party warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
# Optional: Finance-tuned LLMs and ML libraries
try:
    import bloomberggpt  # type: ignore  # Placeholder for BloombergGPT API/client
except Exception:
    bloomberggpt = None
try:
    import fingpt  # type: ignore  # Placeholder for FinGPT API/client
except Exception:
    fingpt = None
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None
try:
    from prophet import Prophet
except Exception:
    Prophet = None
try:
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense  # type: ignore
    import tensorflow as tf  # type: ignore
except Exception:
    Sequential = LSTM = Dense = tf = None
if 'tf' in globals() and tf is not None:
    try:
        # Reduce TensorFlow/Keras log noise
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass

# Optional dependencies for advanced features
try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore
try:
    from googleapiclient.discovery import build  # type: ignore
except Exception:
    build = None  # type: ignore
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None  # type: ignore
try:
    from sklearn.ensemble import IsolationForest  # type: ignore
except Exception:
    IsolationForest = None  # type: ignore

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Real-Time Market & Social Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit cache helpers for external data
@st.cache_data(ttl=300)
def fetch_yf_history(symbol: str, period: str):
    try:
        return yf.Ticker(symbol).history(period=period)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_yf_info(symbol: str):
    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}

# --- Utilities: make DataFrames Arrow/Streamlit friendly ---
def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where object/dict/list values are converted to strings for Arrow serialization.
    Keeps numeric/boolean dtypes intact; normalizes datetimes; safe for st.dataframe/st.table.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        df2 = df.copy()
        import numpy as _np  # local to avoid global pollution
        import datetime as _dt
        import pandas as _pd
        import json as _json

        for col in df2.columns:
            # Only coerce object-like columns; keep numeric/bool as-is
            if pd.api.types.is_object_dtype(df2[col].dtype):
                def _conv(x):
                    try:
                        if x is None or (isinstance(x, float) and _np.isnan(x)):
                            return None
                    except Exception:
                        pass
                    # Normalize timestamps/dates
                    if isinstance(x, (_dt.datetime, _dt.date, _pd.Timestamp)):
                        try:
                            return _pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            return str(x)
                    # Dicts/lists/tuples -> JSON
                    if isinstance(x, (dict, list, tuple)):
                        try:
                            return _json.dumps(x, default=str)
                        except Exception:
                            return str(x)
                    # Numpy scalars -> Python native, then stringify for object cols
                    if isinstance(x, _np.generic):
                        try:
                            x = x.item()
                        except Exception:
                            return str(x)
                    # For object dtype columns, ensure string to avoid mixed-type Arrow issues
                    if not isinstance(x, str):
                        try:
                            return str(x)
                        except Exception:
                            return _json.dumps(x, default=str)
                    return x
                df2[col] = df2[col].map(_conv).astype('string')
        return df2
    except Exception:
        # Fallback: ensure we at least don't crash the UI
        try:
            return df.astype(str)
        except Exception:
            return df

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# API Keys 
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_GEMINI_API_KEY = os.getenv('GOOGLE_GEMINI_API_KEY')

# Initialize AI clients
openai_client = None
legacy_openai = False
gemini_model = None

if OPENAI_API_KEY and openai is not None:
    try:
        # Try a new OpenAI SDK (v1+)
        try:
            from openai import OpenAI as _OpenAI  # type: ignore
            openai_client = _OpenAI(api_key=OPENAI_API_KEY)
        except Exception:
            # Fallback to attribute on module
            try:
                openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)  # type: ignore
            except Exception:
                pass
    except Exception:
        # Fallback to legacy style
        legacy_openai = True
        try:
            openai.api_key = OPENAI_API_KEY  # type: ignore
        except Exception:
            pass

if GOOGLE_GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
        try:
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception:
            # Fallback to a commonly available model name
            gemini_model = genai.GenerativeModel('gemini-pro')
    except Exception:
        gemini_model = None


def ask_openai(prompt: str, max_tokens: int = 150) -> str:
    """Call OpenAI Chat API with compatibility for new and legacy SDKs."""
    if not OPENAI_API_KEY or openai is None:
        raise RuntimeError("OpenAI API not available")

    # New SDK path
    if openai_client is not None and not legacy_openai:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    # Legacy path (openai.ChatCompletion)
    if hasattr(openai, 'ChatCompletion'):
        legacy_resp = openai.ChatCompletion.create(  # type: ignore
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return legacy_resp["choices"][0]["message"]["content"]

    raise RuntimeError("OpenAI client unavailable")


def test_openai_simple():
    """Return tuple (ok: bool, message: str)."""
    try:
        _ = ask_openai("Hello. Reply with 'ok' only.", max_tokens=5)
        return (True, "Working")
    except Exception as e:
        msg = str(e)
        if "quota" in msg.lower() or "429" in msg:
            return (False, "Quota Exceeded")
        if "invalid" in msg.lower() or "api key" in msg.lower():
            return (False, "Key Invalid")
        return (False, msg[:80] + ("..." if len(msg) > 80 else ""))
        
def predict_price_lstm(_self, hist_df, steps=5, lookback: int = 20, epochs: int = 10):
        if tf is None or Sequential is None or LSTM is None or Dense is None:
            return None, "TensorFlow/Keras not available"
        try:
            preds = lstm_residual_only_forecast(hist_df, steps=steps, lookback=int(lookback or 20), epochs=int(epochs or 10))
            return preds.values.tolist() if hasattr(preds, 'values') else list(preds), None
        except Exception as e:
            return None, str(e)


class MarketInsightsDashboard:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Common stock symbols for validation
        self.common_stocks = {
            'AAPL': 'Apple Inc.', 'TSLA': 'Tesla Inc.', 'GOOGL': 'Alphabet Inc.', 'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.', 'META': 'Meta Platforms Inc.', 'NVDA': 'NVIDIA Corporation', 'NFLX': 'Netflix Inc.',
            'JPM': 'JPMorgan Chase & Co.', 'JNJ': 'Johnson & Johnson', 'PG': 'Procter & Gamble Co.', 'UNH': 'UnitedHealth Group Inc.',
            'HD': 'Home Depot Inc.', 'MA': 'Mastercard Inc.', 'V': 'Visa Inc.', 'PYPL': 'PayPal Holdings Inc.', 'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.', 'INTC': 'Intel Corporation', 'ORCL': 'Oracle Corporation'
        }
        # Popular Indian (NSE) symbols for quick suggestions
        self.indian_stocks = {
            'TCS.NS': 'Tata Consultancy Services', 'RELIANCE.NS': 'Reliance Industries', 'INFY.NS': 'Infosys', 'HDFCBANK.NS': 'HDFC Bank',
            'ICICIBANK.NS': 'ICICI Bank', 'SBIN.NS': 'State Bank of India', 'ITC.NS': 'ITC', 'AXISBANK.NS': 'Axis Bank',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank', 'LT.NS': 'Larsen & Toubro', 'BAJFINANCE.NS': 'Bajaj Finance', 'HINDUNILVR.NS': 'Hindustan Unilever',
            'ASIANPAINT.NS': 'Asian Paints', 'MARUTI.NS': 'Maruti Suzuki', 'TATAMOTORS.NS': 'Tata Motors', 'WIPRO.NS': 'Wipro',
            'TATASTEEL.NS': 'Tata Steel', 'POWERGRID.NS': 'Power Grid Corporation', 'ULTRACEMCO.NS': 'UltraTech Cement', 'SUNPHARMA.NS': 'Sun Pharmaceutical'
        }
        self.last_news_query = None
        self.SERPAPI_KEY = os.getenv('SERPAPI_KEY')

    # --- Fine-Tuned LLMs for Finance ---
    def get_finance_llm_response(self, prompt: str, max_tokens: int = 150) -> tuple:
        """Try BloombergGPT/FinGPT, fallback to OpenAI/Gemini."""
        if bloomberggpt is not None:
            try:
                # Placeholder: Replace with actual BloombergGPT API call
                resp = bloomberggpt.generate(prompt, max_tokens=max_tokens)
                return ("BloombergGPT", resp)
            except Exception:
                pass
        if fingpt is not None:
            try:
                # Placeholder: Replace with actual FinGPT API call
                resp = fingpt.generate(prompt, max_tokens=max_tokens)
                return ("FinGPT", resp)
            except Exception:
                pass
        # Fallback to existing AI
        return self.ask_ai_unified(prompt, max_tokens)

    # --- ML-Based Price Prediction ---
    @st.cache_data(ttl=1800)
    def predict_price_arima(self, hist_df, steps=5, order=(5,1,0), return_ci: bool = True):
        if ARIMA is None:
            return None, "ARIMA not available"
        try:
            close = hist_df['Close'].dropna()
            model = ARIMA(close, order=tuple(order) if order else (5,1,0))
            model_fit = model.fit()
            if return_ci and hasattr(model_fit, 'get_forecast'):
                fc_obj = model_fit.get_forecast(steps=steps)
                fc = fc_obj.predicted_mean
                ci = fc_obj.conf_int(alpha=0.2)  # 80% band
                return (fc, ci), None
            else:
                forecast = model_fit.forecast(steps=steps)
                return forecast, None
        except Exception as e:
            return None, str(e)

    @st.cache_data(ttl=1800)
    def detect_trend_patterns_prophet(_self, hist_df):
        if Prophet is None:
            return None, "Prophet not available"
        try:
            df = hist_df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
            # Ensure ds column is timezone-naive
            if isinstance(df['ds'].dtype, pd.DatetimeTZDtype):
                df['ds'] = df['ds'].dt.tz_localize(None)
            else:
                df['ds'] = pd.to_datetime(df['ds'])
                if hasattr(df['ds'], 'dt') and df['ds'].dt.tz is not None:
                    df['ds'] = df['ds'].dt.tz_localize(None)
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=7)
            forecast = m.predict(future)
            return forecast, None
        except Exception as e:
            return None, str(e)


    @st.cache_data(ttl=1800)
    def predict_price_lstm(self, hist_df, steps=5, lookback: int = 20, epochs: int = 10):
        if tf is None:
            return None, "TensorFlow/Keras not available"
        try:
            import numpy as np
            close = hist_df['Close'].values.reshape(-1,1)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(close)
            X, y = [], []
            lookback = int(lookback or 20)
            for i in range(lookback, len(scaled)):
                X.append(scaled[i-lookback:i, 0])
                y.append(scaled[i, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(lookback,1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=int(epochs or 10), batch_size=16, verbose=0)
            last_seq = scaled[-lookback:]
            preds = []
            for _ in range(steps):
                inp = last_seq.reshape((1, lookback, 1))
                pred = model.predict(inp, verbose=0)[0][0]
                preds.append(pred)
                last_seq = np.append(last_seq[1:], [[pred]], axis=0)
            preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            return preds, None
        except Exception as e:
            return None, str(e)

    # --- Sentiment-Weighted Signals ---
    def compute_sentiment_weighted_signal(self, news_sentiment, yt_sentiment, technicals):
        """Combine sentiment and technicals for Buy/Sell/Neutral recommendation."""
        # Assign weights (customize as needed)
        w_news, w_yt, w_tech = 0.4, 0.2, 0.4
        score = 0
        details = []
        if news_sentiment is not None:
            score += w_news * news_sentiment
            details.append(f"News: {news_sentiment:+.2f} × {w_news}")
        if yt_sentiment is not None:
            score += w_yt * yt_sentiment
            details.append(f"YouTube: {yt_sentiment:+.2f} × {w_yt}")
        if technicals is not None:
            score += w_tech * technicals
            details.append(f"Technicals: {technicals:+.2f} × {w_tech}")
        # Recommendation
        if score > 0.2:
            reco = "Buy"
        elif score < -0.2:
            reco = "Sell"
        else:
            reco = "Neutral"
        return score, reco, details

    def ask_ai_unified(self, prompt: str, max_tokens: int = 150) -> tuple:
        order = self._ai_order()
        last_err = None
        for prov in order:
            if prov == 'gemini' and GOOGLE_GEMINI_API_KEY and gemini_model is not None and genai is not None:
                try:
                    resp = gemini_model.generate_content(prompt)
                    return ('Gemini', getattr(resp, 'text', ''))
                except Exception as e:
                    last_err = f"Gemini: {str(e)[:120]}"
                    continue
            if prov == 'openai' and OPENAI_API_KEY and (openai_client is not None or legacy_openai) and openai is not None:
                try:
                    txt = ask_openai(prompt, max_tokens=max_tokens)
                    return ('OpenAI', txt)
                except Exception as e:
                    last_err = f"OpenAI: {str(e)[:120]}"
                    continue
        if not order:
            return (None, "AI not available - configure OpenAI or Gemini in .env, or adjust AI Options.")
        return (None, f"All AI providers failed. Last error: {last_err or 'Unknown'}")
        
    # --- Pattern Detection with Prophet ---
    @st.cache_data(ttl=1800)
    def detect_trend_patterns_prophet(self, hist_df):
        if Prophet is None:
            return None, "Prophet not available"
        try:
            df = hist_df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=7)
            forecast = m.predict(future)
            return forecast, None
        except Exception as e:
            return None, str(e)

    # --- Alerts Helper ---
    def generate_alerts(self, hist_df, anomalies=None, patterns=None):
        alerts = []
        if anomalies is not None and not anomalies.empty:
            for idx, row in anomalies.iterrows():
                alerts.append({
                    'type': 'Anomaly',
                    'date': idx,
                    'price': row['Close'],
                    'desc': f"Unusual price/volume movement"
                })
        if patterns is not None:
            # Example: Prophet trend change points
            if 'trend' in patterns:
                for i, row in patterns.iterrows():
                    if row.get('trend_change', False):
                        alerts.append({
                            'type': 'Trend',
                            'date': row['ds'],
                            'price': row['yhat'],
                            'desc': f"Trend change detected"
                        })
        return alerts

    def _ai_order(self) -> list:
        pref = st.session_state.get('ai_provider', 'Auto')
        has_gemini = bool(GOOGLE_GEMINI_API_KEY and gemini_model is not None)
        has_openai = bool(OPENAI_API_KEY and (openai_client is not None or legacy_openai))
        if pref == 'Gemini only':
            return ['gemini'] if has_gemini else []
        if pref == 'OpenAI only':
            return ['openai'] if has_openai else []
        if pref == 'Gemini first':
            return [p for p in ['gemini', 'openai'] if (p == 'gemini' and has_gemini) or (p == 'openai' and has_openai)]
        if pref == 'OpenAI first':
            return [p for p in ['openai', 'gemini'] if (p == 'openai' and has_openai) or (p == 'gemini' and has_gemini)]
        if has_gemini and has_openai:
            return ['gemini', 'openai']
        if has_gemini:
            return ['gemini']
        if has_openai:
            return ['openai']
        return []

    def ask_ai_unified(self, prompt: str, max_tokens: int = 150) -> tuple:
        order = self._ai_order()
        last_err = None
        for prov in order:
            if prov == 'gemini' and GOOGLE_GEMINI_API_KEY and gemini_model is not None and genai is not None:
                try:
                    resp = gemini_model.generate_content(prompt)
                    return ('Gemini', getattr(resp, 'text', ''))
                except Exception as e:
                    last_err = f"Gemini: {str(e)[:120]}"
                    continue
            if prov == 'openai' and OPENAI_API_KEY and (openai_client is not None or legacy_openai) and openai is not None:
                try:
                    txt = ask_openai(prompt, max_tokens=max_tokens)
                    return ('OpenAI', txt)
                except Exception as e:
                    last_err = f"OpenAI: {str(e)[:120]}"
                    continue
        if not order:
            return (None, "AI not available - configure OpenAI or Gemini in .env, or adjust AI Options.")
        return (None, f"All AI providers failed. Last error: {last_err or 'Unknown'}")

    def normalize_symbol(self, symbol: str) -> str:
        s = (symbol or '').strip().upper()
        if ':' in s:
            s = s.replace(':', '.')
        return s

    def validate_stock_symbol(self, symbol):
        symbol = self.normalize_symbol(symbol)
        if symbol in self.common_stocks:
            return {'valid': True, 'symbol': symbol, 'name': self.common_stocks[symbol], 'message': f"✅ Valid symbol: {self.common_stocks[symbol]}"}
        if symbol in self.indian_stocks:
            return {'valid': True, 'symbol': symbol, 'name': self.indian_stocks[symbol], 'message': f"✅ Valid symbol: {self.indian_stocks[symbol]}"}
        try:
            info = yf.Ticker(symbol).info
            if info.get('longName') or info.get('shortName'):
                return {'valid': True, 'symbol': symbol, 'name': info.get('longName', info.get('shortName', symbol)), 'message': f"✅ Valid symbol: {info.get('longName', info.get('shortName', symbol))}"}
            if '.' not in symbol:
                for sfx in ['.NS', '.BO']:
                    test_sym = f"{symbol}{sfx}"
                    try:
                        info2 = yf.Ticker(test_sym).info
                        if info2.get('longName') or info2.get('shortName'):
                            return {'valid': True, 'symbol': test_sym, 'name': info2.get('longName', info2.get('shortName', test_sym)), 'message': f"✅ Valid symbol: {info2.get('longName', info2.get('shortName', test_sym))}"}
                    except Exception:
                        continue
            return {'valid': False, 'symbol': symbol, 'name': None, 'message': f"❌ Invalid symbol: '{symbol}' not found"}
        except Exception:
            return {'valid': False, 'symbol': symbol, 'name': None, 'message': f"❌ Invalid symbol: '{symbol}' not found"}

    def _resolve_company_name(self, symbol: str) -> str | None:
        s = self.normalize_symbol(symbol)
        if s in self.common_stocks:
            return self.common_stocks[s]
        if s in self.indian_stocks:
            return self.indian_stocks[s]
        try:
            info = yf.Ticker(s).info
            return info.get('longName') or info.get('shortName')
        except Exception:
            return None

    def _build_news_query(self, raw_query: str) -> str:
        name = self._resolve_company_name(raw_query)
        sym = self.normalize_symbol(raw_query)
        base = sym.split('.')[0] if sym else raw_query
        if name:
            return f'"{name}" OR {base}'
        return base

    def get_stock_suggestions(self, partial_symbol):
        partial_symbol = (partial_symbol or '').upper().strip().replace(':', '.')
        suggestions = []
        all_stocks = {**self.common_stocks, **self.indian_stocks}
        for symbol, name in all_stocks.items():
            if partial_symbol in symbol or partial_symbol in name.upper():
                suggestions.append({'symbol': symbol, 'name': name})
        return suggestions[:5]

    def get_stock_data(self, symbol, period="1mo"):
        try:
            symbol_norm = self.normalize_symbol(symbol)
            hist = fetch_yf_history(symbol_norm, period)
            info = fetch_yf_info(symbol_norm)
            if hist.empty:
                if '.' not in symbol_norm:
                    for sfx in ['.NS', '.BO']:
                        test_sym = f"{symbol_norm}{sfx}"
                        try:
                            hist2 = fetch_yf_history(test_sym, period)
                            if not hist2.empty:
                                info = fetch_yf_info(test_sym)
                                hist = hist2
                                symbol_norm = test_sym
                                break
                        except Exception:
                            continue
                if hist is None or hist.empty:
                    return None
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else None
            if current_price is None or pd.isna(current_price):
                return None
            return {
                'history': hist,
                'info': info,
                'current_price': current_price,
                'previous_close': previous_close,
                'currency': info.get('currency') if isinstance(info, dict) else None,
                'resolved_symbol': symbol_norm,
            }
        except Exception as e:
            st.error(f"Error fetching stock data for {symbol}: {e}")
            return None

    def get_alpha_vantage_data(self, symbol):
        if not ALPHA_VANTAGE_API_KEY:
            return None
        try:
            url = "https://www.alphavantage.co/query"
            params = {"function": "TIME_SERIES_INTRADAY", "symbol": symbol, "interval": "5min", "apikey": ALPHA_VANTAGE_API_KEY}
            response = requests.get(url, params=params)
            data = response.json()
            if "Time Series (5min)" in data:
                time_series = data["Time Series (5min)"]
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                return df
            return None
        except Exception as e:
            st.error(f"Error fetching Alpha Vantage data: {e}")
            return None

    def get_news_data(self, query, days=7, prefer_indian_sources: bool = True, max_results: int = 20):
        if not NEWS_API_KEY:
            st.warning("⚠️ News API key not configured. Please add NEWS_API_KEY to your .env file")
            return None
        try:
            url = "https://newsapi.org/v2/everything"
            q_str = self._build_news_query(query)
            self.last_news_query = q_str
            page_size = max(1, min(int(max_results or 20), 100))
            params = {
                "q": q_str,
                "from": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": page_size,
                "apiKey": NEWS_API_KEY,
            }
            if prefer_indian_sources:
                params["domains"] = "moneycontrol.com,livemint.com,economictimes.indiatimes.com,business-standard.com,businesstoday.in,hindustantimes.com,ndtv.com,news18.com,thehindu.com"
            response = requests.get(url, params=params)
            data = response.json()
            articles = []
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                if prefer_indian_sources and len(articles) < page_size:
                    try:
                        params_global = dict(params)
                        params_global.pop("domains", None)
                        params_global["pageSize"] = page_size - len(articles)
                        response2 = requests.get(url, params=params_global)
                        data2 = response2.json()
                        if data2.get("status") == "ok":
                            extra = data2.get("articles", [])
                            seen = set(a.get('url') for a in articles if a.get('url'))
                            for a in extra:
                                u = a.get('url')
                                if u and u in seen:
                                    continue
                                seen.add(u)
                                articles.append(a)
                    except Exception:
                        pass
                return articles
            else:
                st.error(f"News API Error: {data.get('message', 'Unknown error')}")
                return None
        except Exception as e:
            st.error(f"Error fetching news data: {e}")
            return None

    def analyze_sentiment(self, text):
        try:
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            return {
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 'negative' if vader_scores['compound'] < -0.05 else 'neutral',
            }
        except Exception:
            return None

    def get_youtube_trends(self, query, max_results=10):
        if not YOUTUBE_API_KEY or build is None:
            st.warning("⚠️ YouTube API key not configured or googleapiclient not installed.")
            return None
        try:
            youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            search_response = youtube.search().list(
                q=query, part='id,snippet', maxResults=max_results, type='video', order='relevance'
            ).execute()
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            videos_response = youtube.videos().list(part='snippet,statistics', id=','.join(video_ids)).execute()
            return videos_response['items']
        except Exception as e:
            st.error(f"Error fetching YouTube data: {e}")
            st.info("💡 This could be due to API rate limits, invalid API key, or network issues")
            return None

    def generate_ai_summary(self, stock_data, news_data, query):
        try:
            context = f"Stock: {query}"
            price_val = stock_data.get('current_price') if stock_data else None
            if isinstance(price_val, (int, float)):
                context += f", Price: {price_val:.2f}"
            if news_data:
                context += f", {len(news_data)} news articles"
            prompt = (
                f"Brief market summary for {query}:\n\n{context}\n\n"
                "Provide a 2-3 sentence summary covering:\n"
                "- Key market factors\n- News sentiment\n- Short-term outlook\n\nKeep it very concise."
            )
            prov, txt = self.ask_ai_unified(prompt, max_tokens=150)
            if prov is None:
                return txt
            return txt
        except Exception as e:
            return f"Error generating AI summary: {e}"

    def answer_natural_language_query(self, query, stock_data, news_data):
        try:
            context = f"Stock: {query}"
            price_val = stock_data.get('current_price') if stock_data else None
            if isinstance(price_val, (int, float)):
                context += f", Price: {price_val:.2f}"
            if news_data:
                context += f", {len(news_data)} news articles"
            prompt = (
                f"Stock question: {query}\n\nContext: {context}\n\n"
                "Give a brief, factual answer (2-3 sentences max)."
            )
            prov, txt = self.ask_ai_unified(prompt, max_tokens=200)
            if prov is None:
                return txt
            return txt
        except Exception as e:
            return f"Error answering query: {e}"

    def summarize_and_detect_bias(self, news_data: list | None, symbol: str) -> dict:
        """Summarize recent news and flag potential bias using simple heuristics.
        Returns dict with 'summary', 'sentiment_breakdown', 'source_counts'.
        Robust to None, dict inputs, and unexpected sentiment labels.
        """
        if not news_data:
            return {"summary": "No news available.", "sentiment_breakdown": {}, "source_counts": {}}
        # Accept dict payloads like {"articles": [...]} as well
        if isinstance(news_data, dict):
            news_list = news_data.get('articles') or []
        else:
            news_list = news_data
        if not isinstance(news_list, list) or not news_list:
            return {"summary": "No news available.", "sentiment_breakdown": {}, "source_counts": {}}

        texts: list[str] = []
        sources: dict[str, int] = {}
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}

        for a in news_list:
            try:
                title = (a.get('title') or '').strip() if isinstance(a, dict) else ''
                desc = (a.get('description') or '').strip() if isinstance(a, dict) else ''
                src = None
                if isinstance(a, dict):
                    src_val = a.get('source')
                    if isinstance(src_val, dict):
                        src = (src_val or {}).get('name')
                    elif isinstance(src_val, str):
                        src = src_val
                if src:
                    sources[src] = sources.get(src, 0) + 1
                txt = (f"{title}. {desc}" if title or desc else '').strip()
                if not txt:
                    continue
                texts.append(txt)
                s = self.analyze_sentiment(txt)
                if s:
                    label = (s.get('overall_sentiment') or 'neutral').lower()
                    if label not in sentiments:
                        # Fallback classification by score if label unexpected
                        try:
                            comp = float(s.get('vader_compound', 0.0))
                            label = 'positive' if comp > 0.05 else 'negative' if comp < -0.05 else 'neutral'
                        except Exception:
                            label = 'neutral'
                    sentiments[label] += 1
            except Exception:
                continue

        combined = "\n".join(texts[:20])
        ai_summary = None
        if combined:
            try:
                prompt = (
                    f"You are a market analyst. Summarize the key themes from recent news about {symbol} in 3 bullets. "
                    "Note any clear positive/negative tilt or partisan framing. Keep it objective and concise."
                )
                prov, txt = self.ask_ai_unified(prompt + "\n\nNews:\n" + combined[:4000], max_tokens=180)
                if prov is not None and isinstance(txt, str) and txt.strip():
                    ai_summary = txt.strip()
            except Exception:
                ai_summary = None

        if not ai_summary:
            if texts:
                bullets = [t[:140] + ("…" if len(t) > 140 else "") for t in texts[:3]]
                ai_summary = " • " + "\n • ".join(bullets)
            else:
                ai_summary = "No digestible titles/descriptions found."

        return {"summary": ai_summary, "sentiment_breakdown": sentiments, "source_counts": sources}


def create_price_chart(stock_data):
    if not stock_data or stock_data.get('history') is None or stock_data['history'].empty:
        return None
    df = stock_data['history']
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2'))
    currency = stock_data.get('currency')
    y_title = f"Price ({currency})" if currency else "Price"
    fig.update_layout(title='Stock Price Chart', yaxis_title=y_title, yaxis2=dict(title='Volume', overlaying='y', side='right'), xaxis_title='Date', height=500)
    return fig


def create_sentiment_chart(news_data):
    if not news_data:
        return None
    sentiments = []
    for article in news_data:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        text = title + ' ' + description
        sentiment = dashboard.analyze_sentiment(text)
        if sentiment:
            sentiments.append(sentiment)
    if not sentiments:
        return None
    df_sentiment = pd.DataFrame(sentiments)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('VADER Sentiment Distribution', 'TextBlob Polarity', 'Sentiment Categories', 'Subjectivity vs Polarity'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}], [{"type": "pie"}, {"type": "scatter"}]],
    )
    fig.add_trace(go.Histogram(x=df_sentiment['vader_compound'], name='VADER Compound'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df_sentiment['textblob_polarity'], name='TextBlob Polarity'), row=1, col=2)
    sentiment_counts = df_sentiment['overall_sentiment'].value_counts()
    fig.add_trace(go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_sentiment['textblob_polarity'], y=df_sentiment['textblob_subjectivity'], mode='markers', name='Subjectivity vs Polarity'), row=2, col=2)
    fig.update_layout(height=600, title_text="Sentiment Analysis Dashboard")
    return fig


def render_news_section(symbol: str):
    st.subheader("📰 News & Sentiment Analysis")
    news_data = dashboard.get_news_data(
        symbol,
        days=st.session_state.get('news_days', 7),
        prefer_indian_sources=st.session_state.get('prefer_indian_sources', True),
        max_results=st.session_state.get('news_max_articles', 10),
    )
    if news_data:
        total = len(news_data)
        to_show = int(st.session_state.get('news_max_articles', 10))
        if getattr(dashboard, 'last_news_query', None):
            st.caption(f"News search used: {dashboard.last_news_query}")
        st.write(f"Found {total} recent articles")
    else:
        st.warning("⚠️ No news data available. This could be due to:")
        st.write("• Missing News API key")
        st.write("• No recent news articles found for this symbol")
        st.write("• API rate limits or connectivity issues")
        st.info("💡 To enable news analysis, please add your News API key to the .env file")
        st.subheader("📰 Demo News Data")
        demo_news = [
            {'title': f'Demo: {symbol} Stock Analysis', 'description': f'This is a demo news article about {symbol} stock performance and market trends.', 'url': 'https://example.com', 'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')},
            {'title': f'Demo: {symbol} Market Update', 'description': f'Demo article discussing {symbol} market position and investor sentiment.', 'url': 'https://example.com', 'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')},
        ]
        st.info("🔍 Showing demo news data (add API key for real data)")
        news_data = demo_news
    total = len(news_data)

    if news_data:
        st.subheader("📊 Sentiment Analysis")
        sentiments = []
        to_show = int(st.session_state.get('news_max_articles', 10))
        for article in news_data[:to_show]:
            title = article.get('title', '') or ''
            description = article.get('description', '') or ''
            text = title + ' ' + description
            sentiment = dashboard.analyze_sentiment(text)
            if sentiment:
                sentiment['title'] = article.get('title', 'No title')
                sentiment['url'] = article.get('url', '')
                sentiment['publishedAt'] = article.get('publishedAt', '')
                sentiments.append(sentiment)
        if sentiments:
            sentiment_chart = create_sentiment_chart(news_data)
            if sentiment_chart:
                st.plotly_chart(sentiment_chart, use_container_width=True)
            st.subheader("📋 Recent News Articles")
            shown = min(len(sentiments), to_show)
            st.caption(f"Showing {shown} of {total} articles")
            for i, sentiment in enumerate(sentiments[:to_show]):
                with st.expander(f"{i+1}. {sentiment['title']}"):
                    st.write(f"**Sentiment:** {sentiment['overall_sentiment'].title()}")
                    st.write(f"**VADER Score:** {sentiment['vader_compound']:.3f}")
                    st.write(f"**TextBlob Polarity:** {sentiment['textblob_polarity']:.3f}")
                    st.write(f"**Published:** {sentiment['publishedAt']}")
                    st.write(f"**URL:** {sentiment['url']}")
    st.session_state.news_data = news_data if news_data else None

    # Add bias-aware summary at the end
    try:
        summary_res = dashboard.summarize_and_detect_bias(st.session_state.get('news_data'), symbol)
        if not summary_res:
            st.info("No bias summary available.")
        else:
            st.subheader("🧭 News Summary & Bias Signals")
            st.write(summary_res.get('summary') or "")
            colA, colB = st.columns(2)
            with colA:
                sb = summary_res.get('sentiment_breakdown') or {}
                if sb:
                    st.write("Sentiment breakdown:")
                    ordered = {
                        'positive': int((sb.get('positive') or 0) or 0),
                        'negative': int((sb.get('negative') or 0) or 0),
                        'neutral': int((sb.get('neutral') or 0) or 0),
                    }
                    df_sb = pd.DataFrame(list(ordered.items()), columns=["Sentiment", "Count"])
                    st.table(df_sb)
            with colB:
                sc = summary_res.get('source_counts') or {}
                if sc:
                    st.write("Top sources:")
                    top_items = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:5]
                    df_sc = pd.DataFrame(top_items, columns=["Source", "Articles"])
                    st.table(df_sc)
    except Exception as e:
        st.warning(f"Bias summary failed: {e}")


# Initialize dashboard
dashboard = MarketInsightsDashboard()

# Try optional TA library
try:
    import pandas_ta as pta  # noqa: F401
except Exception:
    pta = None  # type: ignore


def main():
    st.title("📊 Real-Time Market & Social Insights Dashboard")
    st.markdown("---")

    def render_market_snapshot():
        try:
            st.subheader("🇮🇳 India Market Snapshot")
            cols = st.columns(3)
            indices = {'NIFTY 50': '^NSEI', 'SENSEX': '^BSESN', 'NIFTY BANK': '^NSEBANK'}
            for i, (name, tick) in enumerate(indices.items()):
                hist = fetch_yf_history(tick, '5d')
                if hist is not None and not hist.empty and len(hist) >= 2:
                    last = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2])
                    change = last - prev
                    pct = (change / prev) * 100 if prev else 0.0
                    cols[i].metric(f"{name} (INR)", f"{last:.2f}", f"{pct:.2f}%")
                else:
                    cols[i].metric(f"{name}", "N/A", "")
        except Exception:
            pass

    render_market_snapshot()

    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""

    st.sidebar.header("🔧 Configuration")

    if 'symbol' not in st.session_state:
        st.session_state.symbol = "AAPL"

    symbol_input = st.sidebar.text_input(
        "Enter Stock Symbol",
        value=st.session_state.symbol,
        help="For Indian stocks use .NS (NSE) or .BO (BSE), e.g., TCS.NS, RELIANCE.NS",
    )
    symbol = dashboard.normalize_symbol(symbol_input)
    st.session_state.symbol = symbol

    if symbol:
        validation = dashboard.validate_stock_symbol(symbol)
        if validation['valid']:
            st.sidebar.success(validation['message'])
            if validation['name']:
                st.sidebar.info(f"**Company:** {validation['name']}")
        else:
            st.sidebar.error(validation['message'])
            if len(symbol) >= 2:
                suggestions = dashboard.get_stock_suggestions(symbol)
                if suggestions:
                    st.sidebar.markdown("**💡 Did you mean:**")
                    for suggestion in suggestions:
                        if st.sidebar.button(f"{suggestion['symbol']} - {suggestion['name']}", key=f"suggest_{suggestion['symbol']}"):
                            st.session_state.symbol = suggestion['symbol']
                            st.rerun()

    st.sidebar.markdown("**🔍 Stock Search:**")
    search_term = st.sidebar.text_input("Search stocks by name or symbol", placeholder="e.g., Apple, TSLA")
    if search_term:
        search_suggestions = dashboard.get_stock_suggestions(search_term)
        if search_suggestions:
            st.sidebar.markdown("**📋 Search Results:**")
            for suggestion in search_suggestions:
                if st.sidebar.button(f"{suggestion['symbol']} - {suggestion['name']}", key=f"search_{suggestion['symbol']}"):
                    st.session_state.symbol = suggestion['symbol']
                    st.rerun()
        else:
            st.sidebar.info("No matches found. Try a different search term.")

    st.sidebar.markdown("**💡 Popular Stocks:**")
    popular_stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX"]
    cols = st.sidebar.columns(4)
    for i, stock in enumerate(popular_stocks):
        if cols[i % 4].button(stock, key=f"stock_{stock}"):
            st.session_state.symbol = stock
            st.rerun()

    st.sidebar.markdown("**🇮🇳 Popular India (NSE):**")
    popular_india = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "TATAMOTORS.NS"]
    cols_in = st.sidebar.columns(4)
    for i, stock in enumerate(popular_india):
        if cols_in[i % 4].button(stock, key=f"stock_in_{stock}"):
            st.session_state.symbol = stock
            st.rerun()

    period = st.sidebar.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)

    # Forecast options
    with st.sidebar.expander("📈 Forecasts"):
        st.session_state.use_arima = st.checkbox(
            "ARIMA forecast", value=st.session_state.get("use_arima", True)
        )
        st.session_state.use_lstm = st.checkbox(
            "LSTM forecast", value=st.session_state.get("use_lstm", False)
        )
        st.session_state.use_prophet = st.checkbox(
            "Prophet 7-day trend", value=st.session_state.get("use_prophet", False)
        )
        st.session_state.forecast_steps = st.slider(
            "Forecast steps (days)", min_value=1, max_value=14, value=int(st.session_state.get("forecast_steps", 5))
        )
        with st.expander("Advanced"):
            c1, c2, c3 = st.columns(3)
            st.session_state.arima_p = c1.number_input("ARIMA p", min_value=0, max_value=10, value=int(st.session_state.get('arima_p', 5)))
            st.session_state.arima_d = c2.number_input("d", min_value=0, max_value=2, value=int(st.session_state.get('arima_d', 1)))
            st.session_state.arima_q = c3.number_input("q", min_value=0, max_value=10, value=int(st.session_state.get('arima_q', 0)))
            c4, c5 = st.columns(2)
            st.session_state.lstm_lookback = c4.number_input("LSTM lookback", min_value=5, max_value=60, value=int(st.session_state.get('lstm_lookback', 20)))
            st.session_state.lstm_epochs = c5.number_input("epochs", min_value=1, max_value=50, value=int(st.session_state.get('lstm_epochs', 10)))

    st.sidebar.markdown("**🔧 API Status:**")

    def test_api_connectivity():
        api_status = {}
        if NEWS_API_KEY:
            try:
                url = "https://newsapi.org/v2/top-headlines"
                params = {"country": "us", "apiKey": NEWS_API_KEY}
                response = requests.get(url, params=params, timeout=5)
                api_status["News API"] = "✅ Working" if response.status_code == 200 else "⚠️ Error"
            except Exception:
                api_status["News API"] = "❌ Failed"
        else:
            api_status["News API"] = "❌ No Key"
        if YOUTUBE_API_KEY and build is not None:
            try:
                yt = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
                _ = yt.search().list(q="test", part='id', maxResults=1).execute()
                api_status["YouTube API"] = "✅ Working"
            except Exception:
                api_status["YouTube API"] = "❌ Failed"
        else:
            api_status["YouTube API"] = "❌ No Key"
        if OPENAI_API_KEY and (openai_client is not None or legacy_openai) and openai is not None:
            try:
                if openai_client is not None and not legacy_openai:
                    _ = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}], max_tokens=5)
                else:
                    _ = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}], max_tokens=5)  # type: ignore
                api_status["OpenAI API"] = "✅ Working"
            except Exception as e:
                em = str(e)
                api_status["OpenAI API"] = "⚠️ Quota Exceeded" if ("quota" in em.lower() or "429" in em) else "❌ Failed"
        else:
            api_status["OpenAI API"] = "❌ No Key"
        if GOOGLE_GEMINI_API_KEY and gemini_model is not None and genai is not None:
            try:
                _ = gemini_model.generate_content("Hello")
                api_status["Gemini API"] = "✅ Working"
            except Exception as e:
                em = str(e)
                if "quota" in em.lower() or "429" in em:
                    api_status["Gemini API"] = "⚠️ Quota Exceeded"
                elif "expired" in em.lower() or "invalid" in em.lower():
                    api_status["Gemini API"] = "⚠️ Key Expired"
                else:
                    api_status["Gemini API"] = "❌ Failed"
        else:
            api_status["Gemini API"] = "❌ No Key"
        return api_status

    if st.sidebar.button("🔄 Test API Status"):
        with st.spinner("Testing APIs..."):
            api_status = test_api_connectivity()
        for api_name, status in api_status.items():
            st.sidebar.text(f"{api_name}: {status}")
    else:
        api_status = {
            "News API": "✅ Key Set" if NEWS_API_KEY else "❌ No Key",
            "YouTube API": "✅ Key Set" if (YOUTUBE_API_KEY and build is not None) else "❌ No Key",
            "OpenAI API": "✅ Key Set" if OPENAI_API_KEY else "❌ No Key",
            "Gemini API": "✅ Key Set" if GOOGLE_GEMINI_API_KEY else "❌ No Key",
        }
        for api_name, status in api_status.items():
            st.sidebar.text(f"{api_name}: {status}")
        st.sidebar.info("💡 Click 'Test API Status' to check connectivity")

    with st.sidebar.expander("📰 News Options"):
        st.session_state.news_days = st.slider("Days window", min_value=3, max_value=30, value=st.session_state.get('news_days', 7))
        st.session_state.prefer_indian_sources = st.checkbox("Prefer Indian sources", value=st.session_state.get('prefer_indian_sources', True))
    st.session_state.news_max_articles = st.slider("Articles to show", min_value=3, max_value=50, value=st.session_state.get('news_max_articles', 10))

    with st.sidebar.expander("🤖 AI Options"):
        st.session_state.ai_provider = st.selectbox(
            "Provider preference",
            options=["Auto", "Gemini first", "OpenAI first", "Gemini only", "OpenAI only"],
            index=["Auto", "Gemini first", "OpenAI first", "Gemini only", "OpenAI only"].index(st.session_state.get('ai_provider', 'Auto')),
        )

    with st.sidebar.expander("⚙️ Advanced"):
        if st.button("Clear cache (data)"):
            try:
                st.cache_data.clear()
                st.success("Cache cleared.")
            except Exception as e:
                st.warning(f"Could not clear cache: {e}")

    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        [
            "Stock Analysis",
            "Fundamentals",
            "Technicals",
            "News Sentiment",
            "YouTube Trends",
            "Sector Heatmaps",
            "Correlation Matrix",
            "Strategy Backtests",
            "Screener",
            "AI Insights",
            "AI Trading Assistant",
            "Anomaly Detection",
            "Alt Data & ESG",
            "Market Ripple Engine",
        ],
    )

    if analysis_type == "Correlation Matrix":
        default_list = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA"
        st.session_state.corr_tickers = st.sidebar.text_input("Tickers (comma-separated)", value=st.session_state.get("corr_tickers", default_list))
    if analysis_type == "Sector Heatmaps":
        st.session_state.heatmap_period = st.sidebar.selectbox("Heatmap Period", ["1d"], index=0)
    if analysis_type == "Strategy Backtests":
        st.sidebar.markdown("**Backtest Settings**")
        st.session_state.bt_strategy = st.sidebar.selectbox(
            "Strategy", ["SMA Crossover", "RSI Mean Reversion"], index=["SMA Crossover", "RSI Mean Reversion"].index(st.session_state.get('bt_strategy', 'SMA Crossover'))
        )
        if st.session_state.bt_strategy == "SMA Crossover":
            c1, c2 = st.sidebar.columns(2)
            st.session_state.bt_fast = c1.number_input("Fast SMA", min_value=3, max_value=100, value=int(st.session_state.get('bt_fast', 10)))
            st.session_state.bt_slow = c2.number_input("Slow SMA", min_value=5, max_value=300, value=int(st.session_state.get('bt_slow', 50)))
        else:
            c1, c2, c3 = st.sidebar.columns(3)
            st.session_state.bt_rsi_period = c1.number_input("RSI Period", min_value=5, max_value=50, value=int(st.session_state.get('bt_rsi_period', 14)))
            st.session_state.bt_rsi_buy = c2.number_input("Buy <", min_value=5, max_value=60, value=int(st.session_state.get('bt_rsi_buy', 30)))
            st.session_state.bt_rsi_sell = c3.number_input("Sell >", min_value=40, max_value=95, value=int(st.session_state.get('bt_rsi_sell', 70)))
        c4, c5 = st.sidebar.columns(2)
        st.session_state.bt_fee_bps = c4.number_input("Fee (bps)", min_value=0, max_value=100, value=int(st.session_state.get('bt_fee_bps', 5)))
        st.session_state.bt_initial = c5.number_input("Initial Capital", min_value=1000, max_value=1_000_000, value=int(st.session_state.get('bt_initial', 10000)), step=1000)
    if analysis_type == "Screener":
        default_universe = "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NFLX"
        st.session_state.screener_universe = st.sidebar.text_input("Universe (comma-separated)", value=st.session_state.get('screener_universe', default_universe))
        st.session_state.screener_rsi = st.sidebar.slider("RSI range", 5, 95, value=tuple(st.session_state.get('screener_rsi', (30, 70))))
        st.session_state.screener_trend = st.sidebar.checkbox("Require SMA50 > SMA200", value=bool(st.session_state.get('screener_trend', False)))
        st.session_state.screener_pct = st.sidebar.slider("Min 1D % change", -10, 10, value=int(st.session_state.get('screener_pct', 0)))
        st.session_state.screener_volx = st.sidebar.slider("Min volume spike (x 20D avg)", 1, 5, value=int(st.session_state.get('screener_volx', 2)))
        st.session_state.screener_sort = st.sidebar.selectbox("Sort by", ["1D %", "VolSpike", "RSI", "Ticker"], index=["1D %", "VolSpike", "RSI", "Ticker"].index(st.session_state.get('screener_sort', '1D %')))
        st.session_state.screener_topn = st.sidebar.slider("Top N", 5, 50, value=int(st.session_state.get('screener_topn', 20)))

    if st.sidebar.button("🚀 Analyze", type="primary"):
        with st.spinner("Fetching data..."):
            stock_data = dashboard.get_stock_data(symbol, period)
            st.session_state.stock_data = stock_data
            if stock_data and stock_data.get('resolved_symbol') and stock_data.get('resolved_symbol') != symbol:
                st.session_state.symbol = stock_data['resolved_symbol']
                symbol = stock_data['resolved_symbol']
            if stock_data and stock_data.get('current_price') is not None:
                validation = dashboard.validate_stock_symbol(symbol)
                if validation['valid'] and validation['name']:
                    st.subheader(f"📊 {validation['name']} ({symbol})")
                    st.caption(
                        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"[View on Yahoo Finance](https://finance.yahoo.com/quote/{symbol})"
                    )
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    cur_ccy = stock_data.get('currency')
                    metric_label = f"Current Price ({cur_ccy})" if cur_ccy else "Current Price"
                    st.metric(metric_label, f"{stock_data['current_price']:.2f}")
                with col2:
                    prev = stock_data.get('previous_close')
                    if isinstance(prev, (int, float)) and prev is not None:
                        change = stock_data['current_price'] - prev
                        change_pct = (change / prev) * 100 if prev else 0.0
                        chg_label = f"Change ({cur_ccy})" if cur_ccy else "Change"
                        st.metric(chg_label, f"{change:.2f}", f"{change_pct:.2f}%")
                    else:
                        st.metric("Change", "N/A")
                with col3:
                    mc = stock_data['info'].get('marketCap') if isinstance(stock_data.get('info'), dict) else None
                    st.metric("Market Cap", f"{(mc/1e9):.2f}B" if isinstance(mc, (int, float)) else "N/A")
                with col4:
                    vol = stock_data['info'].get('volume') if isinstance(stock_data.get('info'), dict) else None
                    st.metric("Volume", f"{(vol/1e6):.1f}M" if isinstance(vol, (int, float)) else "N/A")
                st.subheader("📈 Price Chart")
                price_chart = create_price_chart(stock_data)
                # Overlay forecasts if requested
                try:
                    hist_df = stock_data.get('history')
                    steps = int(st.session_state.get('forecast_steps', 5))
                    if hist_df is not None and not hist_df.empty and price_chart is not None:
                        last_dt = hist_df.index[-1]
                        future_idx = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=steps, freq='B')
                        # ARIMA
                        if st.session_state.get('use_arima'):
                            order = (
                                int(st.session_state.get('arima_p', 5)),
                                int(st.session_state.get('arima_d', 1)),
                                int(st.session_state.get('arima_q', 0)),
                            )
                            fc, err = dashboard.predict_price_arima(hist_df, steps=steps, order=order, return_ci=True)
                            if fc is not None:
                                try:
                                    if isinstance(fc, tuple) and len(fc) == 2:
                                        mean_fc, ci = fc
                                        yvals = mean_fc.values if hasattr(mean_fc, 'values') else np.array(mean_fc)
                                        price_chart.add_trace(
                                            go.Scatter(x=future_idx, y=yvals[:len(future_idx)], name='ARIMA Forecast', line=dict(color='orange', dash='dash'))
                                        )
                                        # Confidence band
                                        try:
                                            ci_idxed = ci.iloc[:len(future_idx)] if hasattr(ci, 'iloc') else ci
                                            lower = ci_idxed.iloc[:, 0].values
                                            upper = ci_idxed.iloc[:, 1].values
                                            price_chart.add_trace(go.Scatter(
                                                x=np.concatenate([future_idx, future_idx[::-1]]),
                                                y=np.concatenate([upper, lower[::-1]]),
                                                fill='toself',
                                                fillcolor='rgba(255,165,0,0.15)',
                                                line=dict(color='rgba(255,165,0,0)'),
                                                hoverinfo='skip',
                                                name='ARIMA 80% CI'
                                            ))
                                        except Exception:
                                            pass
                                    else:
                                        yvals = fc.values if hasattr(fc, 'values') else np.array(fc)
                                        price_chart.add_trace(
                                            go.Scatter(x=future_idx, y=yvals[:len(future_idx)], name='ARIMA Forecast', line=dict(color='orange', dash='dash'))
                                        )
                                except Exception:
                                    pass
                            elif err:
                                st.caption(f"ARIMA: {err}")
                        # LSTM
                        if st.session_state.get('use_lstm'):
                            preds, err = dashboard.predict_price_lstm(
                                hist_df,
                                steps=steps,
                                lookback=int(st.session_state.get('lstm_lookback', 20)),
                                epochs=int(st.session_state.get('lstm_epochs', 10)),
                            )
                            if preds is not None:
                                try:
                                    price_chart.add_trace(
                                        go.Scatter(x=future_idx, y=list(preds)[:len(future_idx)], name='LSTM Forecast', line=dict(color='purple', dash='dot'))
                                    )
                                except Exception:
                                    pass
                            elif err:
                                st.caption(f"LSTM: {err}")
                        # Prophet trend overlay (last actual + 7d)
                        if st.session_state.get('use_prophet'):
                            fc_prophet, p_err = dashboard.detect_trend_patterns_prophet(hist_df)
                            if fc_prophet is not None:
                                try:
                                    dfp = fc_prophet[['ds', 'yhat']].tail(steps)
                                    price_chart.add_trace(
                                        go.Scatter(x=pd.to_datetime(dfp['ds']), y=dfp['yhat'], name='Prophet Trend', line=dict(color='teal', dash='dashdot'))
                                    )
                                except Exception:
                                    pass
                            elif p_err:
                                st.caption(f"Prophet: {p_err}")
                except Exception:
                    pass

                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
                    # Tiny holdout error for ARIMA on last 5 points
                    try:
                        if st.session_state.get('use_arima') and stock_data.get('history') is not None:
                            dfh = stock_data['history']['Close'].dropna()
                            if len(dfh) > 30:
                                train, test = dfh.iloc[:-5], dfh.iloc[-5:]
                                order = (
                                    int(st.session_state.get('arima_p', 5)),
                                    int(st.session_state.get('arima_d', 1)),
                                    int(st.session_state.get('arima_q', 0)),
                                )
                                model = ARIMA(train, order=order) if ARIMA is not None else None
                                if model is not None:
                                    fit = model.fit()
                                    pred = fit.forecast(steps=5)
                                    mape = float(np.mean(np.abs((test.values - pred.values) / test.values))) * 100
                                    st.caption(f"ARIMA holdout MAPE (last 5): {mape:.2f}%")
                    except Exception:
                        pass
                    try:
                        hist_df = stock_data.get('history')
                        if hist_df is not None and not hist_df.empty:
                            csv_data = hist_df.to_csv().encode('utf-8')
                            st.download_button(label="📥 Download Price History (CSV)", data=csv_data, file_name=f"{symbol}_{period}_history.csv", mime='text/csv')
                    except Exception:
                        pass
                else:
                    st.warning("No price data available for this symbol. Please try a different stock symbol.")
            else:
                st.error(f"❌ Unable to fetch data for '{symbol}'. This symbol may be invalid, delisted, or not available.")
                st.info("💡 Try using a valid stock symbol like: AAPL, TSLA, GOOGL, MSFT, AMZN")

            if analysis_type in ["Fundamentals", "Stock Analysis"]:
                render_fundamentals_section(symbol)
            if analysis_type in ["Technicals", "Stock Analysis"]:
                render_technicals_section(symbol, stock_data)
            if analysis_type in ["News Sentiment", "Stock Analysis"]:
                render_news_section(symbol)
            if analysis_type == "Strategy Backtests":
                render_strategy_backtests(symbol, stock_data)
            if analysis_type == "Screener":
                render_screener()
            if analysis_type in ["YouTube Trends", "Stock Analysis"]:
                st.subheader("📺 YouTube Trends")
                youtube_data = dashboard.get_youtube_trends(symbol, max_results=5)
                if youtube_data:
                    st.write(f"Found {len(youtube_data)} related videos")
                    for video in youtube_data:
                        snippet = video['snippet']
                        statistics = video.get('statistics', {})
                        with st.expander(f"🎥 {snippet['title']}"):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                thumb = snippet.get('thumbnails', {}).get('medium', {}).get('url')
                                if thumb:
                                    st.image(thumb, width=200)
                            with col2:
                                st.write(f"**Channel:** {snippet['channelTitle']}")
                                st.write(f"**Published:** {snippet['publishedAt']}")
                                st.write(f"**Views:** {statistics.get('viewCount', 'N/A')}")
                                st.write(f"**Likes:** {statistics.get('likeCount', 'N/A')}")
                                desc = snippet.get('description', '') or ''
                                st.write(f"**Description:** {desc[:200]}...")
                                video_id = video.get('id') if isinstance(video.get('id'), str) else video.get('id', {}).get('videoId')
                                video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
                                if video_url:
                                    st.write(f"**URL:** {video_url}")
                else:
                    st.warning("⚠️ No YouTube data available. This could be due to:")
                    st.write("• Missing YouTube API key")
                    st.write("• No videos found for this symbol")
                    st.write("• API rate limits or connectivity issues")
                    st.info("💡 To enable YouTube analysis, please add your YouTube API key to the .env file")
                    st.subheader("📺 Demo YouTube Data")
                    demo_youtube = [
                        {
                            'id': 'demo_video_1',
                            'snippet': {
                                'title': f'Demo: {symbol} Stock Analysis Video',
                                'channelTitle': 'Demo Channel',
                                'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                                'description': f'Demo video discussing {symbol} stock performance and market analysis.',
                                'thumbnails': {'medium': {'url': 'https://via.placeholder.com/320x180?text=Demo+Video'}},
                            },
                            'statistics': {'viewCount': '1,234', 'likeCount': '56'},
                        },
                        {
                            'id': 'demo_video_2',
                            'snippet': {
                                'title': f'Demo: {symbol} Market Update',
                                'channelTitle': 'Demo Channel',
                                'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                                'description': f'Demo video about {symbol} market trends and investor sentiment.',
                                'thumbnails': {'medium': {'url': 'https://via.placeholder.com/320x180?text=Demo+Video'}},
                            },
                            'statistics': {'viewCount': '2,345', 'likeCount': '78'},
                        },
                    ]
                    st.info("🔍 Showing demo YouTube data (add API key for real data)")
                    for video in demo_youtube:
                        snippet = video['snippet']
                        statistics = video.get('statistics', {})
                        with st.expander(f"🎥 {snippet['title']}"):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                thumb = snippet.get('thumbnails', {}).get('medium', {}).get('url')
                                if thumb:
                                    st.image(thumb, width=200)
                            with col2:
                                st.write(f"**Channel:** {snippet['channelTitle']}")
                                st.write(f"**Published:** {snippet['publishedAt']}")
                                st.write(f"**Views:** {statistics.get('viewCount', 'N/A')}")
                                st.write(f"**Likes:** {statistics.get('likeCount', 'N/A')}")
                                desc = snippet.get('description', '') or ''
                                st.write(f"**Description:** {desc[:200]}...")
                                video_url = f"https://www.youtube.com/watch?v={video['id']}"
                                st.write(f"**URL:** {video_url}")

            if analysis_type == "Sector Heatmaps":
                render_sector_heatmap(st.session_state.get("heatmap_period", "1d"))
            if analysis_type == "Correlation Matrix":
                render_correlation_matrix(st.session_state.get("corr_tickers", "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA"))

            # Composite signal and alerts (post-chart)
            try:
                hist_df = st.session_state.get('stock_data', {}).get('history') if isinstance(st.session_state.get('stock_data'), dict) else None
                if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
                    # Compute technical score
                    close = hist_df['Close']
                    rsi_series = _rsi(close)
                    rsi_last = rsi_series.iloc[-1] if not rsi_series.empty else np.nan
                    ema12 = _ema(close, 12)
                    ema26 = _ema(close, 26)
                    macd = ema12 - ema26
                    macd_signal = _ema(macd, 9)
                    tech_score = 0.0
                    if pd.notna(rsi_last):
                        tech_score += float(np.clip((rsi_last - 50.0) / 50.0, -1.0, 1.0)) * 0.7
                    try:
                        if pd.notna(macd.iloc[-1]) and pd.notna(macd_signal.iloc[-1]):
                            tech_score += (0.3 if macd.iloc[-1] > macd_signal.iloc[-1] else -0.3)
                    except Exception:
                        pass
                    # News sentiment avg (VADER compound)
                    news_data_ss = st.session_state.get('news_data')
                    news_sent = None
                    if isinstance(news_data_ss, list) and len(news_data_ss) > 0:
                        vals = []
                        for a in news_data_ss[:30]:
                            t = f"{a.get('title','')}. {a.get('description','')}".strip()
                            s = dashboard.analyze_sentiment(t)
                            if s and isinstance(s.get('vader_compound'), (int, float)):
                                vals.append(s['vader_compound'])
                        if vals:
                            news_sent = float(np.mean(vals))
                    # No YouTube sentiment implemented yet
                    yt_sent = None
                    score, reco, details = dashboard.compute_sentiment_weighted_signal(news_sent, yt_sent, tech_score)
                    st.subheader("🧮 Composite Signal")
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.metric("Score", f"{score:+.2f}")
                    with c2:
                        st.metric("Recommendation", reco)
                    if details:
                        st.caption(" + ".join(details))
            except Exception:
                pass

    if analysis_type in ["AI Insights", "Stock Analysis"]:
        st.subheader("🤖 AI-Powered Insights")
        sd_stock_data = st.session_state.get('stock_data')
        sd_news_data = st.session_state.get('news_data')
        if OPENAI_API_KEY or GOOGLE_GEMINI_API_KEY:
            st.subheader("💬 Ask Questions")
            st.markdown("**💡 Example questions:**")
            ex_cols = st.columns(3)
            with ex_cols[0]:
                if st.button(f"Why did {symbol} change today?", key="ex_q1"):
                    st.session_state["question_text"] = f"Why did {symbol} change today?"
                    st.rerun()
            with ex_cols[1]:
                if st.button(f"Is {symbol} a good investment?", key="ex_q2"):
                    st.session_state["question_text"] = f"Is {symbol} a good investment?"
                    st.rerun()
            with ex_cols[2]:
                if st.button(f"What affects {symbol} price?", key="ex_q3"):
                    st.session_state["question_text"] = f"What affects {symbol} price?"
                    st.rerun()
            col1, col2 = st.columns([3, 1])
            with col1:
                user_query_p = st.text_input(
                    "Ask a question about the stock:",
                    placeholder="e.g., 'Why did Tesla fall today?' or 'Is Apple a good buy?'",
                    key="question_text",
                )
            with col2:
                st.write("")
                ask_button_p = st.button("🤖 Ask AI", key="ask_ai_persistent", type="primary", use_container_width=True)
            if user_query_p and ask_button_p:
                with st.spinner("🤖 AI is analyzing your question..."):
                    cur_price = sd_stock_data.get('current_price') if sd_stock_data else None
                    price_str = f"{cur_price:.2f}" if isinstance(cur_price, (int, float)) else "N/A"
                    prompt = f"Stock: {symbol}, Price: {price_str}. Question: {user_query_p}. Give 2-3 sentence answer."
                    prov, txt = dashboard.ask_ai_unified(prompt, max_tokens=100)
                    if prov is None:
                        st.error(txt)
                    else:
                        st.success(f"**🤖 AI Answer ({prov}):**")
                        st.write(txt)
            st.subheader("🔧 AI API Status")
            c1, c2 = st.columns(2)
            with c1:
                if OPENAI_API_KEY:
                    st.success("✅ OpenAI API Key: Configured")
                else:
                    st.error("❌ OpenAI API Key: Not configured")
            with c2:
                if GOOGLE_GEMINI_API_KEY:
                    st.success("✅ Gemini API Key: Configured")
                else:
                    st.error("❌ Gemini API Key: Not configured")
            if st.button("🧪 Test AI APIs", key="test_ai_providers"):
                with st.spinner("Testing AI providers..."):
                    results = []
                    pref = st.session_state.get('ai_provider', 'Auto')
                    test_openai_flag = pref in ['Auto', 'OpenAI first', 'Gemini first', 'OpenAI only']
                    test_gemini_flag = pref in ['Auto', 'OpenAI first', 'Gemini first', 'Gemini only']
                    if test_openai_flag:
                        if OPENAI_API_KEY and (openai_client is not None or legacy_openai) and openai is not None:
                            ok, msg = test_openai_simple()
                            results.append("✅ OpenAI: Working" if ok else f"❌ OpenAI: {msg}")
                        else:
                            results.append("❌ OpenAI: No Key")
                    if test_gemini_flag:
                        if GOOGLE_GEMINI_API_KEY and gemini_model is not None and genai is not None:
                            try:
                                _ = gemini_model.generate_content("Test")
                                results.append("✅ Gemini: Working")
                            except Exception as e:
                                em = str(e)
                                if "quota" in em.lower() or "429" in em:
                                    results.append("⚠️ Gemini: Quota Exceeded")
                                elif "expired" in em.lower() or "invalid" in em.lower():
                                    results.append("⚠️ Gemini: Key Expired/Invalid")
                                else:
                                    results.append(f"❌ Gemini: {em[:50]}...")
                        else:
                            results.append("❌ Gemini: No Key")
                    for r in results:
                        st.text(r)
            st.subheader("🤖 Demo AI Insights")
            if sd_stock_data:
                price_val = sd_stock_data.get('current_price') if sd_stock_data else None
                price_str = f"{price_val:.2f}" if isinstance(price_val, (int, float)) else "N/A"
                demo_summary = f"""
                Based on the available data for {symbol}:

                📊 **Current Status**: {symbol} is currently trading at {price_str}

                📈 **Key Factors**:
                • Market sentiment appears to be mixed
                • Recent trading volume suggests moderate investor interest
                • Technical indicators show potential for both upside and downside movement

                🔮 **Outlook**: Short-term outlook remains uncertain, with potential for volatility in the coming days.

                *This is a demo summary. Add AI API keys for real AI-powered insights.*
                """
                st.write(demo_summary)
            if (OPENAI_API_KEY or GOOGLE_GEMINI_API_KEY) and sd_stock_data and sd_news_data:
                st.write("**Market Summary:**")
                summary = dashboard.generate_ai_summary(sd_stock_data, sd_news_data, symbol)
                st.write(summary)
                if isinstance(summary, str) and ("quota exceeded" in summary.lower() or "api key" in summary.lower() or "error" in summary.lower()):
                    st.warning("⚠️ API limits or configuration issues detected. Showing demo summary as fallback:")
                    price_val = sd_stock_data.get('current_price') if sd_stock_data else None
                    price_str = f"{price_val:.2f}" if isinstance(price_val, (int, float)) else "N/A"
                    demo_summary = f"""
                    Based on the available data for {symbol}:

                    📊 **Current Status**: {symbol} is currently trading at {price_str}

                    📈 **Key Factors**:
                    • Market sentiment appears to be mixed
                    • Recent trading volume suggests moderate investor interest
                    • Technical indicators show potential for both upside and downside movement

                    📰 **News Impact**: Recent news articles indicate varying sentiment about {symbol}'s performance

                    🔮 **Outlook**: Short-term outlook remains uncertain, with potential for volatility in the coming days.

                    *This is a demo summary. Add API keys and check billing for real AI-powered insights.*
                    """
                    st.write(demo_summary)

    if analysis_type == "AI Trading Assistant":
        render_ai_trading_assistant(symbol, st.session_state.get('stock_data'))
    if analysis_type == "Anomaly Detection":
        render_anomaly_detection(symbol, st.session_state.get('stock_data'))
    if analysis_type == "Alt Data & ESG":
        render_alt_data_esg(symbol)
    if analysis_type == "Market Ripple Engine":
        render_market_ripple_engine(symbol)

    if analysis_type == "News Sentiment" and st.button("🔄 Refresh News", use_container_width=True):
        render_news_section(st.session_state.get('symbol', 'AAPL'))

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Built with Streamlit | Powered by Multiple APIs | Real-time Market Insights</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def render_technicals_section(symbol: str, stock_data: dict | None):
    st.subheader("🧮 Technical Analysis")
    if stock_data and stock_data.get('history') is not None and not stock_data['history'].empty:
        df = stock_data['history'].copy()
    else:
        try:
            df = yf.Ticker(symbol).history(period="6mo")
        except Exception as e:
            st.error(f"Failed to fetch price history for technicals: {e}")
            return
    if df.empty:
        st.warning("No price history available for technicals.")
        return
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA12'] = _ema(df['Close'], 12)
    df['EMA26'] = _ema(df['Close'], 26)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = _ema(df['MACD'], 9)
    df['RSI14'] = _rsi(df['Close'], 14)
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    signals = []
    if len(df.dropna(subset=['SMA50', 'SMA200'])) > 3:
        last50 = df['SMA50'].iloc[-1]
        last200 = df['SMA200'].iloc[-1]
        prev50 = df['SMA50'].iloc[-2]
        prev200 = df['SMA200'].iloc[-2]
        if prev50 < prev200 and last50 > last200:
            signals.append("Golden cross (SMA50 > SMA200)")
        if prev50 > prev200 and last50 < last200:
            signals.append("Death cross (SMA50 < SMA200)")
    rsi = df['RSI14'].iloc[-1]
    if pd.notna(rsi):
        if rsi > 70:
            signals.append("RSI overbought (>70)")
        elif rsi < 30:
            signals.append("RSI oversold (<30)")
    if len(df.dropna(subset=['MACD', 'MACD_signal'])) > 3:
        if df['MACD'].iloc[-2] < df['MACD_signal'].iloc[-2] and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
            signals.append("MACD bullish crossover")
        if df['MACD'].iloc[-2] > df['MACD_signal'].iloc[-2] and df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]:
            signals.append("MACD bearish crossover")
    from plotly.subplots import make_subplots as _make_subplots
    fig = _make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA200', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_mid'], name='BB Mid', line=dict(width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], name='RSI(14)'), row=2, col=1)
    fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="lightgray", opacity=0.2, row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal'), row=3, col=1)
    fig.update_layout(height=800, title=f"Technicals for {symbol}")
    st.plotly_chart(fig, use_container_width=True)
    if signals:
        st.info("\n".join([f"• {s}" for s in signals]))
    try:
        last_row = df.iloc[-1]
        P = (last_row['High'] + last_row['Low'] + last_row['Close']) / 3
        R1 = 2 * P - last_row['Low']
        S1 = 2 * P - last_row['High']
        R2 = P + (last_row['High'] - last_row['Low'])
        S2 = P - (last_row['High'] - last_row['Low'])
        pivots = {"Pivot": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Pivot", f"{pivots['Pivot']:.2f}")
        c2.metric("R1", f"{pivots['R1']:.2f}")
        c3.metric("S1", f"{pivots['S1']:.2f}")
        c4.metric("R2", f"{pivots['R2']:.2f}")
        c5.metric("S2", f"{pivots['S2']:.2f}")
    except Exception:
        pass


def _daily_returns(series: pd.Series) -> pd.Series:
    try:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    except Exception:
        return pd.Series(dtype=float)


def _max_drawdown(equity: pd.Series) -> float:
    try:
        roll_max = equity.cummax()
        dd = (equity / roll_max) - 1.0
        return float(dd.min()) if not dd.empty else 0.0
    except Exception:
        return 0.0


def _sharpe(daily: pd.Series) -> float:
    try:
        mu = daily.mean()
        sd = daily.std()
        if sd == 0 or np.isnan(sd):
            return 0.0
        return float((mu / sd) * np.sqrt(252))
    except Exception:
        return 0.0


def _cagr(equity: pd.Series) -> float:
    try:
        if equity.empty or equity.iloc[0] == 0:
            return 0.0
        n = len(equity)
        return float((equity.iloc[-1] / equity.iloc[0]) ** (252 / max(1, n)) - 1.0)
    except Exception:
        return 0.0


def _backtest_sma(df: pd.DataFrame, fast: int, slow: int, fee_bps: float, initial: float = 10000) -> dict:
    px = df['Close'].dropna()
    if len(px) < max(fast, slow) + 5:
        return {"equity": None, "trades": pd.DataFrame(), "stats": {}}
    sma_f = px.rolling(fast).mean()
    sma_s = px.rolling(slow).mean()
    signal = (sma_f > sma_s).astype(int)
    pos = signal.shift(1).fillna(0)
    rets = _daily_returns(px)
    strat = pos * rets
    # Fees on change in position
    churn = pos.diff().abs().fillna(0)
    fee = churn * (fee_bps / 10000.0)
    strat_net = strat - fee
    equity = (1 + strat_net).cumprod() * initial
    # Trades
    entries = (pos.diff() > 0)
    exits = (pos.diff() < 0)
    trades = []
    in_trade = False
    entry_price = None
    for i, dt in enumerate(px.index):
        if entries.iloc[i] and not in_trade:
            in_trade = True
            entry_price = px.iloc[i]
        elif exits.iloc[i] and in_trade:
            trades.append({"EntryDate": dt, "ExitDate": dt, "EntryPx": entry_price, "ExitPx": px.iloc[i]})
            in_trade = False
    trades_df = pd.DataFrame(trades)
    daily = equity.pct_change().fillna(0.0)
    stats = {
        "CAGR": _cagr(equity),
        "MaxDD": _max_drawdown(equity),
        "Sharpe": _sharpe(daily),
        "Trades": int(len(trades_df)),
    }
    return {"equity": equity, "trades": trades_df, "stats": stats}


def _backtest_rsi(df: pd.DataFrame, period: int, buy_th: int, sell_th: int, fee_bps: float, initial: float = 10000) -> dict:
    px = df['Close'].dropna()
    if len(px) < period + 5:
        return {"equity": None, "trades": pd.DataFrame(), "stats": {}}
    rsi = _rsi(px, period).fillna(50)
    pos = pd.Series(0, index=px.index)
    in_pos = False
    for i, dt in enumerate(px.index):
        v = rsi.iloc[i]
        if not in_pos and v < buy_th:
            in_pos = True
            pos.iloc[i] = 1
        elif in_pos and v > sell_th:
            in_pos = False
            pos.iloc[i] = 0
        else:
            pos.iloc[i] = int(in_pos)
    rets = _daily_returns(px)
    strat = pos.shift(1).fillna(0) * rets
    churn = pos.diff().abs().fillna(0)
    fee = churn * (fee_bps / 10000.0)
    strat_net = strat - fee
    equity = (1 + strat_net).cumprod() * initial
    # Trades simplistic
    entries = (pos.diff() > 0)
    exits = (pos.diff() < 0)
    trades = []
    in_trade = False
    entry_price = None
    for i, dt in enumerate(px.index):
        if entries.iloc[i] and not in_trade:
            in_trade = True
            entry_price = px.iloc[i]
        elif exits.iloc[i] and in_trade:
            trades.append({"EntryDate": dt, "ExitDate": dt, "EntryPx": entry_price, "ExitPx": px.iloc[i]})
            in_trade = False
    trades_df = pd.DataFrame(trades)
    daily = equity.pct_change().fillna(0.0)
    stats = {
        "CAGR": _cagr(equity),
        "MaxDD": _max_drawdown(equity),
        "Sharpe": _sharpe(daily),
        "Trades": int(len(trades_df)),
    }
    return {"equity": equity, "trades": trades_df, "stats": stats}


def render_strategy_backtests(symbol: str, stock_data: dict | None):
    st.subheader("🧪 Strategy Backtests")
    df = None
    try:
        if stock_data and isinstance(stock_data.get('history'), pd.DataFrame) and not stock_data['history'].empty:
            df = stock_data['history']
        else:
            df = yf.Ticker(symbol).history(period='1y')
    except Exception:
        df = None
    if df is None or df.empty or 'Close' not in df:
        st.info("No price data for backtest.")
        return
    strat = st.session_state.get('bt_strategy', 'SMA Crossover')
    fee = float(st.session_state.get('bt_fee_bps', 5))
    initial = float(st.session_state.get('bt_initial', 10000))
    if strat == 'SMA Crossover':
        fast = int(st.session_state.get('bt_fast', 10))
        slow = int(st.session_state.get('bt_slow', 50))
        res = _backtest_sma(df, fast, slow, fee, initial)
    else:
        period = int(st.session_state.get('bt_rsi_period', 14))
        buy_th = int(st.session_state.get('bt_rsi_buy', 30))
        sell_th = int(st.session_state.get('bt_rsi_sell', 70))
        res = _backtest_rsi(df, period, buy_th, sell_th, fee, initial)
    equity = res.get('equity')
    trades = res.get('trades')
    stats = res.get('stats', {})
    if equity is None or equity.empty:
        st.info("Insufficient data for selected parameters.")
        return
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Price & Signals**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Equity Curve**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=equity.index, y=equity.values, name='Equity'))
        st.plotly_chart(fig2, use_container_width=True)
    # Stats
    st.markdown("**Performance**")
    perf = pd.DataFrame({
        'Metric': ['CAGR', 'Max Drawdown', 'Sharpe', 'Trades'],
        'Value': [
            f"{stats.get('CAGR', 0.0)*100:.2f}%",
            f"{stats.get('MaxDD', 0.0)*100:.2f}%",
            f"{stats.get('Sharpe', 0.0):.2f}",
            f"{int(stats.get('Trades', 0))}",
        ]
    })
    st.table(sanitize_df_for_streamlit(perf))
    # Downloads
    try:
        st.download_button("📥 Download Equity (CSV)", data=equity.to_csv().encode('utf-8'), file_name=f"{symbol}_equity.csv", mime='text/csv')
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            st.download_button("📥 Download Trades (CSV)", data=trades.to_csv(index=False).encode('utf-8'), file_name=f"{symbol}_trades.csv", mime='text/csv')
    except Exception:
        pass


def render_screener():
    st.subheader("🔎 Screener")
    universe_csv = st.session_state.get('screener_universe', '')
    tickers = [t.strip().upper() for t in universe_csv.split(',') if t.strip()]
    if not tickers:
        st.info("Provide tickers in the sidebar.")
        return
    rows = []
    for t in tickers:
        try:
            h = fetch_yf_history(t, '6mo')
            if h is None or h.empty:
                continue
            close = h['Close']
            rsi = _rsi(close, 14).iloc[-1] if len(close) >= 14 else np.nan
            sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
            sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
            pct1d = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100 if len(close) >= 2 else np.nan
            volx = (h['Volume'].tail(5).mean() / (h['Volume'].rolling(20).mean().iloc[-1] + 1e-9)) if len(h) >= 20 else np.nan
            info = fetch_yf_info(t)
            pe = info.get('trailingPE') if isinstance(info, dict) else None
            rows.append({
                'Ticker': t,
                'RSI': float(rsi) if pd.notna(rsi) else None,
                'SMA50': float(sma50) if pd.notna(sma50) else None,
                'SMA200': float(sma200) if pd.notna(sma200) else None,
                'TrendUp': (float(sma50) > float(sma200)) if (pd.notna(sma50) and pd.notna(sma200)) else False,
                '1D %': float(pct1d) if pd.notna(pct1d) else None,
                'VolSpike': float(volx) if pd.notna(volx) else None,
                'PE': float(pe) if isinstance(pe, (int, float)) else None,
            })
        except Exception:
            continue
    if not rows:
        st.info("No data for the selected universe.")
        return
    df = pd.DataFrame(rows)
    # Filters
    rlo, rhi = st.session_state.get('screener_rsi', (30, 70))
    if rlo is not None and rhi is not None:
        df = df[(df['RSI'].fillna(50) >= rlo) & (df['RSI'].fillna(50) <= rhi)]
    if st.session_state.get('screener_trend'):
        df = df[df['TrendUp'] == True]
    pct_min = float(st.session_state.get('screener_pct', 0))
    df = df[df['1D %'].fillna(-999) >= pct_min]
    volx_min = float(st.session_state.get('screener_volx', 2))
    df = df[df['VolSpike'].fillna(0) >= volx_min]
    # Sort & top N
    sort_key = st.session_state.get('screener_sort', '1D %')
    ascending = False if sort_key in ['1D %', 'VolSpike', 'RSI'] else True
    if sort_key in df.columns:
        df = df.sort_values(sort_key, ascending=ascending)
    topn = int(st.session_state.get('screener_topn', 20))
    df = df.head(topn)
    st.dataframe(sanitize_df_for_streamlit(df))


def render_fundamentals_section(symbol: str):
    st.subheader("🏦 Fundamentals")
    tkr = yf.Ticker(symbol)
    ratios = {}
    try:
        info = tkr.info
        keys = [
            'trailingPE', 'forwardPE', 'priceToBook', 'profitMargins', 'grossMargins',
            'operatingMargins', 'returnOnEquity', 'returnOnAssets', 'debtToEquity',
            'currentRatio', 'quickRatio', 'dividendYield', 'payoutRatio',
        ]
        for k in keys:
            if k in info and info[k] is not None:
                ratios[k] = info[k]
    except Exception:
        pass
    if ratios:
        st.markdown("**Key Ratios**")
        cols = st.columns(4)
        items = list(ratios.items())
        for i, (k, v) in enumerate(items):
            label = k.replace('trailingPE', 'P/E (TTM)').replace('forwardPE', 'Forward P/E').replace('priceToBook', 'P/B')
            cols[i % 4].metric(label, f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
    else:
        st.info("Ratios unavailable.")

    def show_df(df: pd.DataFrame, title: str):
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.markdown(f"**{title}**")
            st.dataframe(sanitize_df_for_streamlit(df).iloc[:20])
        else:
            st.info(f"{title} unavailable.")

    with st.expander("Income Statement"):
        try:
            show_df(tkr.financials, "Income Statement")
        except Exception as e:
            st.info(f"Unavailable: {e}")
    with st.expander("Balance Sheet"):
        try:
            show_df(tkr.balance_sheet, "Balance Sheet")
        except Exception as e:
            st.info(f"Unavailable: {e}")
    with st.expander("Cash Flow"):
        try:
            show_df(tkr.cashflow, "Cash Flow")
        except Exception as e:
            st.info(f"Unavailable: {e}")

    with st.expander("Earnings Calendar"):
        try:
            ed = tkr.get_earnings_dates(limit=4)
            st.dataframe(sanitize_df_for_streamlit(ed))
        except Exception:
            try:
                cal = tkr.calendar
                st.dataframe(sanitize_df_for_streamlit(cal))
            except Exception as e:
                st.info(f"Unavailable: {e}")

    with st.expander("Institutional / Major Holders"):
        try:
            ih = tkr.institutional_holders
            if ih is not None and not ih.empty:
                st.dataframe(sanitize_df_for_streamlit(ih))
            mh = tkr.major_holders
            if mh is not None and not mh.empty:
                st.dataframe(sanitize_df_for_streamlit(mh))
        except Exception as e:
            st.info(f"Unavailable: {e}")

    with st.expander("Insider Transactions"):
        try:
            it = getattr(tkr, 'insider_transactions', None)
            if it is not None and isinstance(it, pd.DataFrame) and not it.empty:
                st.dataframe(sanitize_df_for_streamlit(it))
            else:
                st.info("No recent insider transactions found.")
        except Exception as e:
            st.info(f"Unavailable: {e}")


def render_sector_heatmap(period: str = "1d"):
    st.subheader("🗺️ Sector Heatmap")
    sectors = {
        'Energy': 'XLE', 'Technology': 'XLK', 'Financials': 'XLF', 'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP', 'Industrials': 'XLI', 'Health Care': 'XLV', 'Utilities': 'XLU',
        'Materials': 'XLB', 'Real Estate': 'XLRE', 'Communication': 'XLC',
    }
    pct_changes = []
    for name, etf in sectors.items():
        try:
            hist = yf.Ticker(etf).history(period='5d')
            if len(hist) >= 2:
                last = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                pct = (last - prev) / prev * 100
                pct_changes.append({"Sector": name, "ETF": etf, "%Change": pct})
        except Exception:
            continue
    if not pct_changes:
        st.info("No sector data available.")
        return
    df = pd.DataFrame(pct_changes)
    fig = px.treemap(df, path=['Sector'], values='%Change', color='%Change', color_continuous_scale=['red', 'white', 'green'])
    fig.update_layout(height=450, title="Sector 1-day % Change (via ETFs)")
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_matrix(tickers_csv: str):
    st.subheader("🔗 Correlation Matrix")
    tickers = [t.strip().upper() for t in (tickers_csv or '').split(',') if t.strip()]
    if not tickers:
        st.info("Enter tickers to compute correlation.")
        return
    prices = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period='6mo')
            if not hist.empty:
                prices[t] = hist['Close']
        except Exception:
            continue
    if not prices:
        st.info("No price data for the provided tickers.")
        return
    df = pd.DataFrame(prices).dropna()
    if df.empty:
        st.info("Insufficient overlapping data.")
        return
    rets = df.pct_change().dropna()
    corr = rets.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdYlGn', origin='lower')
    fig.update_layout(height=500, title="6M Daily Return Correlations")
    st.plotly_chart(fig, use_container_width=True)


def render_ai_trading_assistant(symbol: str, stock_data: dict | None):
    st.subheader("🧠 AI Trading Assistant")
    if (stock_data is None) or (stock_data.get('history') is None) or (isinstance(stock_data.get('history'), pd.DataFrame) and stock_data['history'].empty):
        st.info("Run Analyze to load price data first.")
        return
    df = stock_data['history']
    rsi_val = _rsi(df['Close']).iloc[-1]
    trend = 'Uptrend' if df['Close'].iloc[-1] > df['Close'].rolling(50).mean().iloc[-1] else 'Down/Sideways'

    # --- Action Recommendation: Short-term and Long-term (Buy/Hold) ---
    try:
        close = df['Close']
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd = ema12 - ema26
        macd_signal = _ema(macd, 9)
        sma50 = close.rolling(50, min_periods=25).mean()
        sma200 = close.rolling(200, min_periods=100).mean()
        last_close = float(close.iloc[-1])
        last_rsi = float(rsi_val) if pd.notna(rsi_val) else None
        macd_bull = bool(pd.notna(macd.iloc[-1]) and pd.notna(macd_signal.iloc[-1]) and macd.iloc[-1] > macd_signal.iloc[-1])
        price_above_ema = bool(pd.notna(ema12.iloc[-1]) and last_close > float(ema12.iloc[-1]))
        have_long_ma = pd.notna(sma50.iloc[-1]) and pd.notna(sma200.iloc[-1])

        # Short-term rule: Buy if momentum turns up or oversold bounce
        short_buy = False
        reasons_short = []
        if macd_bull and price_above_ema:
            short_buy = True
            reasons_short.append("MACD > signal and price > EMA12")
        if (last_rsi is not None) and (last_rsi < 35):
            short_buy = True
            reasons_short.append(f"RSI({int(14)}) < 35")

        # Long-term rule: Buy if uptrend (SMA50>SMA200) and price above SMA200
        long_buy = False
        reasons_long = []
        if have_long_ma:
            if float(sma50.iloc[-1]) > float(sma200.iloc[-1]) and last_close > float(sma200.iloc[-1]):
                long_buy = True
                reasons_long.append("SMA50 > SMA200 and price > SMA200")
        # Fallback: if not enough data for SMA200, use EMA26 proxy
        else:
            if price_above_ema:
                long_buy = True
                reasons_long.append("Price > EMA12 (proxy)")

        st.subheader("📌 Action Recommendation")
        ca, cb = st.columns(2)
        with ca:
            st.metric("Short-term", "Buy" if short_buy else "Hold", 
                      ", ".join(reasons_short) if short_buy and reasons_short else ("Momentum weak" if not short_buy else ""))
        with cb:
            st.metric("Long-term", "Buy" if long_buy else "Hold", 
                      ", ".join(reasons_long) if long_buy and reasons_long else ("Trend unclear" if not long_buy else ""))

        # Optional AI justification
        if st.button("🤖 Explain recommendation", key="explain_reco"):
            with st.spinner("AI reviewing signals..."):
                ctx = [
                    f"RSI14={last_rsi:.1f}" if last_rsi is not None else "RSI14=N/A",
                    f"MACD_bull={macd_bull}",
                    f"Price={last_close:.2f}",
                ]
                if have_long_ma:
                    ctx += [f"SMA50={float(sma50.iloc[-1]):.2f}", f"SMA200={float(sma200.iloc[-1]):.2f}"]
                prompt = (
                    f"For {symbol}, classify short-term and long-term action strictly as 'Buy' or 'Hold' using these signals: "
                    f"{', '.join(ctx)}. Short-term considers momentum (MACD, EMA12, RSI). "
                    f"Long-term considers trend (SMA50 vs SMA200). "
                    f"Return two lines exactly in this format:\n"
                    f"Short-term: <Buy|Hold> - <one-line reason>\nLong-term: <Buy|Hold> - <one-line reason>"
                )
                provider, msg = dashboard.ask_ai_unified(prompt, max_tokens=120)
                if provider is None:
                    st.warning(msg)
                else:
                    st.info(f"Explanation by {provider}:")
                    st.write(msg)
    except Exception:
        # If any computation fails, just skip the recommendation block gracefully
        pass
    col1, col2, col3 = st.columns(3)
    risk = col1.select_slider("Risk tolerance", options=["Low", "Medium", "High"], value="Medium")
    horizon = col2.selectbox("Timeframe", ["Intraday", "Swing (days)", "Position (weeks+)"], index=1)
    allocation = col3.slider("Max position size (%)", 1, 50, 10)
    question = st.text_input("Your objective (optional)", placeholder="e.g., Find a low-risk swing setup")
    if st.button("🤖 Generate Plan", type="primary"):
        with st.spinner("AI drafting a plan..."):
            ta_context = f"RSI(14)={rsi_val:.1f}, Trend={trend}, Last Close={df['Close'].iloc[-1]:.2f}"
            prompt = (
                f"Create a concise trading plan for {symbol}. Context: {ta_context}. "
                f"Risk tolerance: {risk}. Timeframe: {horizon}. Max position: {allocation}%. "
                f"Objective: {question or 'General alpha-seeking'}. "
                "Include: thesis, entry, stop, target, risk management, invalidation. 6-8 lines max."
            )
            prov, txt = dashboard.ask_ai_unified(prompt, max_tokens=220)
            if prov is None:
                st.error(txt)
            else:
                st.success(f"Plan by {prov}")
                st.write(txt)


def render_anomaly_detection(symbol: str, stock_data: dict | None):
    st.subheader("🚨 Price/Volume Anomaly Detection")
    if IsolationForest is None:
        st.warning("scikit-learn is not installed. Install it: pip install scikit-learn")
        return
    if (stock_data is None) or (stock_data.get('history') is None) or (isinstance(stock_data.get('history'), pd.DataFrame) and stock_data['history'].empty):
        st.info("Run Analyze to load price data first.")
        return
    df = stock_data['history'].copy()
    df['ret'] = df['Close'].pct_change()
    df['vol_z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-9)
    feat = df[['ret', 'vol_z']].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    model.fit(feat)
    scores = model.decision_function(feat)
    preds = model.predict(feat)
    df['anomaly'] = (preds == -1)
    df['score'] = scores
    recent = df.tail(120)
    anom = recent[recent['anomaly']]
    c1, c2 = st.columns([3, 1])
    with c1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=recent.index, y=recent['Close'], name='Close'), row=1, col=1)
        if not anom.empty:
            fig.add_trace(go.Scatter(x=anom.index, y=anom['Close'], mode='markers', name='Anomaly', marker=dict(color='red', size=8)), row=1, col=1)
        fig.add_trace(go.Bar(x=recent.index, y=recent['Volume'], name='Volume'), row=2, col=1)
        fig.update_layout(height=600, title=f"{symbol} anomalies (IsolationForest)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Anomalies (last 120 bars)", int(anom.shape[0]))
        if not anom.empty:
            st.dataframe(sanitize_df_for_streamlit(anom[['Close', 'ret', 'vol_z', 'score']].tail(10)))


def render_alt_data_esg(symbol: str):
    st.subheader("🌱 Alt Data & ESG")
    t = yf.Ticker(symbol)
    with st.expander("ESG & Sustainability"):
        try:
            sus = t.sustainability
            if isinstance(sus, pd.DataFrame) and not sus.empty:
                st.dataframe(sanitize_df_for_streamlit(sus))
            else:
                st.info("No sustainability data available.")
        except Exception as e:
            st.info(f"Unavailable: {e}")
    with st.expander("Company Profile"):
        try:
            info = t.info
            if not isinstance(info, dict) or not info:
                st.info("Company profile unavailable.")
            else:
                def _fmt_int(n):
                    try:
                        return f"{int(n):,}"
                    except Exception:
                        return str(n) if n is not None else "N/A"

                def _fmt_cap(v):
                    try:
                        v = float(v)
                        if v >= 1e12:
                            return f"{v/1e12:.2f}T"
                        if v >= 1e9:
                            return f"{v/1e9:.2f}B"
                        if v >= 1e6:
                            return f"{v/1e6:.2f}M"
                        if v >= 1e3:
                            return f"{v/1e3:.2f}K"
                        return f"{v:.0f}"
                    except Exception:
                        return "N/A"

                name = info.get('longName') or info.get('shortName') or symbol
                st.markdown(f"### {name}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Sector**")
                    st.write(info.get('sector') or 'N/A')
                    st.markdown("**Industry**")
                    st.write(info.get('industry') or 'N/A')
                    st.markdown("**Country**")
                    st.write(info.get('country') or 'N/A')
                with c2:
                    st.markdown("**Full-time Employees**")
                    st.write(_fmt_int(info.get('fullTimeEmployees')) if info.get('fullTimeEmployees') else 'N/A')
                    st.markdown("**Market Cap**")
                    st.write(_fmt_cap(info.get('marketCap')) if info.get('marketCap') else 'N/A')
                    st.markdown("**Website**")
                    website = info.get('website')
                    if website:
                        st.markdown(f"[{website}]({website})")
                    else:
                        st.write('N/A')

                # Address and contact
                addr_parts = [info.get('address1'), info.get('city'), info.get('state'), info.get('zip'), info.get('country')]
                address = ", ".join([p for p in addr_parts if p])
                if address:
                    st.caption(address)
                if info.get('phone'):
                    st.caption(f"Phone: {info.get('phone')}")

                # Business summary
                summary = info.get('longBusinessSummary')
                if summary:
                    with st.expander("Business Summary"):
                        st.write(summary)
        except Exception as e:
            st.info(f"Unavailable: {e}")
    with st.expander("Alt Signals (simple)"):
        try:
            hist = t.history(period='3mo')
            if not hist.empty:
                vol_spike = (hist['Volume'].tail(5).mean() / (hist['Volume'].rolling(20).mean().iloc[-1] + 1e-9))
               
                st.metric("Recent volume vs 20D avg", f"{vol_spike:.2f}x")
                gaps = (hist['Open'] - hist['Close'].shift(1)).abs() / hist['Close'].shift(1)
                st.metric("Gap frequency (3mo)", f"{(gaps > 0.02).mean()*100:.1f}%")
            else:
                st.info("No 3-month history.")
        except Exception:
            st.info("Alt signals unavailable.")


def render_market_ripple_engine(symbol: str):
    st.subheader("🌊 Predictive Market Ripple Engine")
    if nx is None:
        st.warning("networkx not installed. Install it: pip install networkx")
        return
    peers_default = f"{symbol},XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLU,XLB,XLC,XLRE"
    peers_csv = st.text_input("Universe (comma-separated tickers)", value=peers_default)
    shock_pct = st.slider("Shock to root symbol (%)", -10, 10, -3)
    tickers = [t.strip().upper() for t in peers_csv.split(',') if t.strip()]
    prices = {}
    for tkr in tickers:
        try:
            h = yf.Ticker(tkr).history(period='6mo')
            if not h.empty:
                prices[tkr] = h['Close']
        except Exception:
            continue
    if symbol not in prices:
        st.info("Root symbol missing or no data. Run Analyze first or include it in the universe.")
        return
    df = pd.DataFrame(prices).dropna()
    if df.empty:
        st.info("Not enough overlapping data.")
        return
    corr = df.pct_change().dropna().corr()
    G = nx.Graph()
    for n in corr.columns:
        G.add_node(n)
    for i in corr.columns:
        for j in corr.columns:
            if i < j:
                w = corr.loc[i, j]
                if abs(w) >= 0.45:
                    G.add_edge(i, j, weight=float(w))
    if G.number_of_edges() == 0:
        st.info("No strong relationships found (|corr| >= 0.45). Try more tickers or longer window.")
        return
    pos = nx.spring_layout(G, seed=42, weight='weight')
    edge_x, edge_y = [], []
    for e in G.edges(data=True):
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x, node_y, node_text, node_color = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)
        node_color.append('tomato' if n == symbol else 'skyblue')
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='gray'), hoverinfo='none', showlegend=False)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='bottom center', marker=dict(size=14, color=node_color, line=dict(width=1, color='white')))
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(height=520, title="Correlation Network (|corr|>=0.45)", xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)
    impacts = []
    for nbr in G.neighbors(symbol):
        w = G.edges[symbol, nbr]['weight']
        impacts.append({"Ticker": nbr, "Impact %": shock_pct * w})
    if impacts:
        st.markdown("**Immediate Impact Estimates**")
        st.dataframe(sanitize_df_for_streamlit(pd.DataFrame(impacts).sort_values("Impact %", key=np.abs, ascending=False)))
    else:
        st.info("No directly connected neighbors to show impacts for.")
    

if __name__ == "__main__":
    main()


