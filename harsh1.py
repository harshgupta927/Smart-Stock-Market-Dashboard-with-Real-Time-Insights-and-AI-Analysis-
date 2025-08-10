import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import openai
import google.generativeai as genai
from googleapiclient.discovery import build
import time

# Load environment variables
load_dotenv()

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

# Page configuration
st.set_page_config(
    page_title="Real-Time Market & Social Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

if OPENAI_API_KEY:
    try:
        # Try new OpenAI SDK (v1+)
        try:
            from openai import OpenAI as _OpenAI
            openai_client = _OpenAI(api_key=OPENAI_API_KEY)
        except Exception:
            # Fallback to attribute on module
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        # Fallback to legacy style
        legacy_openai = True
        try:
            openai.api_key = OPENAI_API_KEY
        except Exception as e:
            st.error(f"❌ OpenAI initialization failed: {e}")

if GOOGLE_GEMINI_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"❌ Gemini initialization failed: {e}")


def ask_openai(prompt: str, max_tokens: int = 150) -> str:
    """Call OpenAI Chat API with compatibility for new and legacy SDKs.
    Returns the assistant text or raises an Exception with the underlying error.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured")

    # New SDK path
    if openai_client is not None and not legacy_openai:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            # Try legacy on specific attribute errors
            err = str(e)
            if "AttributeError" in err or "chat" in err.lower() and "completions" in err.lower():
                # fallthrough to legacy
                pass
            else:
                raise

    # Legacy path (openai.ChatCompletion)
    try:
        # Some environments need explicit api_key set
        if not getattr(openai, "api_key", None):
            openai.api_key = OPENAI_API_KEY
        legacy_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return legacy_resp["choices"][0]["message"]["content"]
    except Exception as e:
        raise


def test_openai_simple():
    """Return tuple (ok: bool, message: str)."""
    try:
        txt = ask_openai("Hello. Reply with 'ok' only.", max_tokens=5)
        return (True, "Working")
    except Exception as e:
        msg = str(e)
        if "quota" in msg.lower() or "429" in msg:
            return (False, "Quota Exceeded")
        if "invalid" in msg.lower() or "api key" in msg.lower():
            return (False, "Key Invalid")
        return (False, msg[:80] + ("..." if len(msg) > 80 else ""))

class MarketInsightsDashboard:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Common stock symbols for validation
        self.common_stocks = {
            'AAPL': 'Apple Inc.',
            'TSLA': 'Tesla Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'PG': 'Procter & Gamble Co.',
            'UNH': 'UnitedHealth Group Inc.',
            'HD': 'Home Depot Inc.',
            'MA': 'Mastercard Inc.',
            'V': 'Visa Inc.',
            'PYPL': 'PayPal Holdings Inc.',
            'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.',
            'INTC': 'Intel Corporation',
            'ORCL': 'Oracle Corporation'
        }
        # Popular Indian (NSE) symbols for quick suggestions
        self.indian_stocks = {
            'TCS.NS': 'Tata Consultancy Services',
            'RELIANCE.NS': 'Reliance Industries',
            'INFY.NS': 'Infosys',
            'HDFCBANK.NS': 'HDFC Bank',
            'ICICIBANK.NS': 'ICICI Bank',
            'SBIN.NS': 'State Bank of India',
            'ITC.NS': 'ITC',
            'AXISBANK.NS': 'Axis Bank',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'LT.NS': 'Larsen & Toubro',
            'BAJFINANCE.NS': 'Bajaj Finance',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'ASIANPAINT.NS': 'Asian Paints',
            'MARUTI.NS': 'Maruti Suzuki',
            'TATAMOTORS.NS': 'Tata Motors',
            'WIPRO.NS': 'Wipro',
            'TATASTEEL.NS': 'Tata Steel',
            'POWERGRID.NS': 'Power Grid Corporation',
            'ULTRACEMCO.NS': 'UltraTech Cement',
            'SUNPHARMA.NS': 'Sun Pharmaceutical'
        }
        # Last news query used (for UI display)
        self.last_news_query = None

    def _ai_order(self) -> list[str]:
        pref = st.session_state.get('ai_provider', 'Auto')
        has_gemini = bool(GOOGLE_GEMINI_API_KEY and gemini_model)
        has_openai = bool(OPENAI_API_KEY and (openai_client or legacy_openai))
        if pref == 'Gemini only':
            return ['gemini'] if has_gemini else []
        if pref == 'OpenAI only':
            return ['openai'] if has_openai else []
        if pref == 'Gemini first':
            return [p for p in ['gemini', 'openai'] if (p=='gemini' and has_gemini) or (p=='openai' and has_openai)]
        if pref == 'OpenAI first':
            return [p for p in ['openai', 'gemini'] if (p=='gemini' and has_gemini) or (p=='openai' and has_openai)]
        # Auto: prefer Gemini if available else OpenAI
        if has_gemini and has_openai:
            return ['gemini', 'openai']
        if has_gemini:
            return ['gemini']
        if has_openai:
            return ['openai']
        return []

    def ask_ai_unified(self, prompt: str, max_tokens: int = 150) -> tuple[str | None, str]:
        order = self._ai_order()
        last_err = None
        for prov in order:
            if prov == 'gemini' and GOOGLE_GEMINI_API_KEY and gemini_model:
                try:
                    resp = gemini_model.generate_content(prompt)
                    return ('Gemini', resp.text)
                except Exception as e:
                    last_err = f"Gemini: {str(e)[:120]}"
                    continue
            if prov == 'openai' and OPENAI_API_KEY and (openai_client or legacy_openai):
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
        """Normalize user input to Yahoo Finance format.
        - Replace ':' with '.' so 'TCS:NS' becomes 'TCS.NS'
        - Upper-case and strip whitespace
        """
        s = (symbol or '').strip().upper()
        if ':' in s:
            s = s.replace(':', '.')
        return s
    
    def validate_stock_symbol(self, symbol):
        """Validate stock symbol and provide suggestions"""
        symbol = self.normalize_symbol(symbol)
        
        # Check if it's a common stock
        if symbol in self.common_stocks:
            return {
                'valid': True,
                'symbol': symbol,
                'name': self.common_stocks[symbol],
                'message': f"✅ Valid symbol: {self.common_stocks[symbol]}"
            }
        # Check popular Indian list
        if symbol in self.indian_stocks:
            return {
                'valid': True,
                'symbol': symbol,
                'name': self.indian_stocks[symbol],
                'message': f"✅ Valid symbol: {self.indian_stocks[symbol]}"
            }
        
        # Try to get info from Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info.get('longName') or info.get('shortName'):
                return {
                    'valid': True,
                    'symbol': symbol,
                    'name': info.get('longName', info.get('shortName', symbol)),
                    'message': f"✅ Valid symbol: {info.get('longName', info.get('shortName', symbol))}"
                }
            # Fallback: if no exchange suffix, try Indian exchanges
            if '.' not in symbol:
                for sfx in ['.NS', '.BO']:
                    test_sym = f"{symbol}{sfx}"
                    try:
                        info2 = yf.Ticker(test_sym).info
                        if info2.get('longName') or info2.get('shortName'):
                            return {
                                'valid': True,
                                'symbol': test_sym,
                                'name': info2.get('longName', info2.get('shortName', test_sym)),
                                'message': f"✅ Valid symbol: {info2.get('longName', info2.get('shortName', test_sym))}"
                            }
                    except Exception:
                        continue
                # If still not found, fall through to invalid
            else:
                return {
                    'valid': False,
                    'symbol': symbol,
                    'name': None,
                    'message': f"❌ Invalid symbol: '{symbol}' not found"
                }
        except Exception:
            return {
                'valid': False,
                'symbol': symbol,
                'name': None,
                'message': f"❌ Invalid symbol: '{symbol}' not found"
            }
    
    def _resolve_company_name(self, symbol: str) -> str | None:
        """Return a human-friendly company name from dictionaries or Yahoo info."""
        s = self.normalize_symbol(symbol)
        if s in self.common_stocks:
            return self.common_stocks[s]
        if s in self.indian_stocks:
            return self.indian_stocks[s]
        # Try yfinance as a fallback
        try:
            info = yf.Ticker(s).info
            return info.get('longName') or info.get('shortName')
        except Exception:
            return None

    def _build_news_query(self, raw_query: str) -> str:
        """Build a better NewsAPI query string from a symbol or name.
        For tickers like RELIANCE.NS, prefer the company name, but include
        the base symbol as an OR fallback.
        """
        name = self._resolve_company_name(raw_query)
        sym = self.normalize_symbol(raw_query)
        base = sym.split('.')[0] if sym else raw_query
        if name:
            return f'"{name}" OR {base}'
        return base

    def get_stock_suggestions(self, partial_symbol):
        """Get stock suggestions based on partial input"""
        # Normalize common variants like ':' to '.' for Indian tickers during search
        partial_symbol = (partial_symbol or '').upper().strip().replace(':', '.')
        suggestions = []
        
        all_stocks = {**self.common_stocks, **self.indian_stocks}
        for symbol, name in all_stocks.items():
            if partial_symbol in symbol or partial_symbol in name.upper():
                suggestions.append({'symbol': symbol, 'name': name})
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def get_stock_data(self, symbol, period="1mo"):
        """Get stock data using Yahoo Finance"""
        try:
            symbol_norm = self.normalize_symbol(symbol)
            hist = fetch_yf_history(symbol_norm, period)
            info = fetch_yf_info(symbol_norm)
            
            # Check if we got valid data
            if hist.empty:
                # Fallback for Indian exchanges if no suffix provided
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
            
            # Additional validation
            if current_price is None or pd.isna(current_price):
                return None
            
            return {
                'history': hist,
                'info': info,
                'current_price': current_price,
                'previous_close': previous_close,
                'currency': info.get('currency') if isinstance(info, dict) else None,
                'resolved_symbol': symbol_norm
            }
        except Exception as e:
            st.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def get_alpha_vantage_data(self, symbol):
        """Get real-time data from Alpha Vantage"""
        if not ALPHA_VANTAGE_API_KEY:
            return None
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": "5min",
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if "Time Series (5min)" in data:
                time_series = data["Time Series (5min)"]
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                return df
            else:
                return None
        except Exception as e:
            st.error(f"Error fetching Alpha Vantage data: {e}")
            return None
    
    def get_news_data(self, query, days=7, prefer_indian_sources: bool = True, max_results: int = 20):
        """Get news articles from NewsAPI"""
        if not NEWS_API_KEY:
            st.warning("⚠️ News API key not configured. Please add NEWS_API_KEY to your .env file")
            return None
        
        try:
            url = "https://newsapi.org/v2/everything"
            # Build a richer query (company name + base ticker) for better relevance
            q_str = self._build_news_query(query)
            self.last_news_query = q_str
            page_size = max(1, min(int(max_results or 20), 100))
            params = {
                "q": q_str,
                "from": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": page_size,
                "apiKey": NEWS_API_KEY
            }
            if prefer_indian_sources:
                # Prefer Indian sources when searching NSE/BSE related names
                params["domains"] = "moneycontrol.com,livemint.com,economictimes.indiatimes.com,business-standard.com,businesstoday.in,hindustantimes.com,ndtv.com,news18.com,thehindu.com"
            
            response = requests.get(url, params=params)
            data = response.json()
            
            articles = []
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                # If we're short and were restricting to Indian domains, top up with global results
                if prefer_indian_sources and len(articles) < page_size:
                    try:
                        params_global = dict(params)
                        params_global.pop("domains", None)
                        params_global["pageSize"] = page_size - len(articles)
                        response2 = requests.get(url, params=params_global)
                        data2 = response2.json()
                        if data2.get("status") == "ok":
                            extra = data2.get("articles", [])
                            # Deduplicate by URL first, then by title
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
        """Analyze sentiment using VADER and TextBlob"""
        try:
            # VADER sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # TextBlob sentiment
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
                'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 'negative' if vader_scores['compound'] < -0.05 else 'neutral'
            }
        except Exception as e:
            return None
    
    def get_youtube_trends(self, query, max_results=10):
        """Get YouTube trending videos related to the query"""
        if not YOUTUBE_API_KEY:
            st.warning("⚠️ YouTube API key not configured. Please add YOUTUBE_API_KEY to your .env file")
            return None
        
        try:
            youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            
            # Search for videos
            search_response = youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=max_results,
                type='video',
                order='relevance'
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            # Get video details
            videos_response = youtube.videos().list(
                part='snippet,statistics',
                id=','.join(video_ids)
            ).execute()
            
            return videos_response['items']
        except Exception as e:
            st.error(f"Error fetching YouTube data: {e}")
            st.info("💡 This could be due to API rate limits, invalid API key, or network issues")
            return None
    
    def generate_ai_summary(self, stock_data, news_data, query):
        """Generate AI summary using OpenAI or Gemini"""
        try:
            # Prepare minimal context
            context = f"Stock: {query}"
            if stock_data and stock_data.get('current_price'):
                context += f", Price: {stock_data['current_price']:.2f}"
            
            if news_data:
                context += f", {len(news_data)} news articles"
            
            prompt = f"""
            Brief market summary for {query}:
            
            {context}
            
            Provide a 2-3 sentence summary covering:
            - Key market factors
            - News sentiment
            - Short-term outlook
            
            Keep it very concise.
            """
            prov, txt = self.ask_ai_unified(prompt, max_tokens=150)
            if prov is None:
                return txt
            return txt
                
        except Exception as e:
            return f"Error generating AI summary: {e}"
    
    def answer_natural_language_query(self, query, stock_data, news_data):
        """Answer natural language questions about the stock"""
        try:
            context = f"Stock: {query}"
            
            if stock_data:
                context += f", Price: {stock_data.get('current_price', 'N/A')}"
            
            if news_data:
                context += f", {len(news_data)} news articles"
            
            prompt = f"""
            Stock question: {query}
            
            Context: {context}
            
            Give a brief, factual answer (2-3 sentences max).
            """
            prov, txt = self.ask_ai_unified(prompt, max_tokens=200)
            if prov is None:
                return txt
            return txt
                
        except Exception as e:
            return f"Error answering query: {e}"

def create_price_chart(stock_data):
    """Create interactive price chart"""
    if not stock_data or stock_data.get('history') is None or stock_data['history'].empty:
        return None
    
    df = stock_data['history']
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2'
    ))
    
    currency = stock_data.get('currency')
    y_title = f"Price ({currency})" if currency else "Price"
    fig.update_layout(
        title='Stock Price Chart',
        yaxis_title=y_title,
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        xaxis_title='Date',
        height=500
    )
    
    return fig

def create_sentiment_chart(news_data):
    """Create sentiment analysis chart"""
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
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    # VADER sentiment distribution
    fig.add_trace(go.Histogram(x=df_sentiment['vader_compound'], name='VADER Compound'), row=1, col=1)
    
    # TextBlob polarity
    fig.add_trace(go.Histogram(x=df_sentiment['textblob_polarity'], name='TextBlob Polarity'), row=1, col=2)
    
    # Sentiment categories
    sentiment_counts = df_sentiment['overall_sentiment'].value_counts()
    fig.add_trace(go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values), row=2, col=1)
    
    # Subjectivity vs Polarity
    fig.add_trace(go.Scatter(
        x=df_sentiment['textblob_polarity'],
        y=df_sentiment['textblob_subjectivity'],
        mode='markers',
        name='Subjectivity vs Polarity'
    ), row=2, col=2)
    
    fig.update_layout(height=600, title_text="Sentiment Analysis Dashboard")
    return fig

def render_news_section(symbol: str):
    st.subheader("📰 News & Sentiment Analysis")
    # Fetch using current sidebar options
    news_data = dashboard.get_news_data(
        symbol,
        days=st.session_state.get('news_days', 7),
        prefer_indian_sources=st.session_state.get('prefer_indian_sources', True),
        max_results=st.session_state.get('news_max_articles', 10)
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
        # Demo news
        st.subheader("📰 Demo News Data")
        demo_news = [
            {
                'title': f'Demo: {symbol} Stock Analysis',
                'description': f'This is a demo news article about {symbol} stock performance and market trends.',
                'url': 'https://example.com',
                'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            {
                'title': f'Demo: {symbol} Market Update',
                'description': f'Demo article discussing {symbol} market position and investor sentiment.',
                'url': 'https://example.com',
                'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            }
        ]
        st.info("🔍 Showing demo news data (add API key for real data)")
        news_data = demo_news

    # Sentiment analysis
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
        else:
            st.warning("⚠️ No sentiment data could be analyzed from the news articles.")

    # Persist for AI Insights
    st.session_state.news_data = news_data if news_data else None

# Initialize dashboard
dashboard = MarketInsightsDashboard()

# Try optional TA library
try:
    import pandas_ta as pta
except Exception:
    pta = None

# Main app
def main():
    st.title("📊 Real-Time Market & Social Insights Dashboard")
    st.markdown("---")
    
    # Quick India market snapshot
    def render_market_snapshot():
        try:
            st.subheader("🇮🇳 India Market Snapshot")
            cols = st.columns(3)
            indices = {
                'NIFTY 50': '^NSEI',
                'SENSEX': '^BSESN',
                'NIFTY BANK': '^NSEBANK',
            }
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
    
    # Initialize session state for question
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Stock symbol input with validation
    if 'symbol' not in st.session_state:
        st.session_state.symbol = "AAPL"
    
    symbol_input = st.sidebar.text_input("Enter Stock Symbol", value=st.session_state.symbol,
                                         help="For Indian stocks use .NS (NSE) or .BO (BSE), e.g., TCS.NS, RELIANCE.NS")
    # Normalize common variants like TCS:NS -> TCS.NS
    symbol = dashboard.normalize_symbol(symbol_input)
    st.session_state.symbol = symbol
    
    # Real-time stock validation
    if symbol:
        validation = dashboard.validate_stock_symbol(symbol)
        
        if validation['valid']:
            st.sidebar.success(validation['message'])
            if validation['name']:
                st.sidebar.info(f"**Company:** {validation['name']}")
        else:
            st.sidebar.error(validation['message'])
            
            # Show suggestions for invalid symbols
            if len(symbol) >= 2:
                suggestions = dashboard.get_stock_suggestions(symbol)
                if suggestions:
                    st.sidebar.markdown("**💡 Did you mean:**")
                    for suggestion in suggestions:
                        if st.sidebar.button(f"{suggestion['symbol']} - {suggestion['name']}", key=f"suggest_{suggestion['symbol']}"):
                            st.session_state.symbol = suggestion['symbol']
                            st.rerun()
    
    # Stock search and suggestions
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
    
    # Helpful stock suggestions
    st.sidebar.markdown("**💡 Popular Stocks:**")
    popular_stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX"]
    cols = st.sidebar.columns(4)
    for i, stock in enumerate(popular_stocks):
        if cols[i % 4].button(stock, key=f"stock_{stock}"):
            st.session_state.symbol = stock
            st.rerun()
    # Quick shortcuts for Indian stocks
    st.sidebar.markdown("**🇮🇳 Popular India (NSE):**")
    popular_india = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "TATAMOTORS.NS"]
    cols_in = st.sidebar.columns(4)
    for i, stock in enumerate(popular_india):
        if cols_in[i % 4].button(stock, key=f"stock_in_{stock}"):
            st.session_state.symbol = stock
            st.rerun()
    
    # Time period
    period = st.sidebar.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
    
    # API Status with connectivity testing
    st.sidebar.markdown("**🔧 API Status:**")
    
    # Test API connectivity
    def test_api_connectivity():
        api_status = {}
        
        # Test News API
        if NEWS_API_KEY:
            try:
                url = "https://newsapi.org/v2/top-headlines"
                params = {"country": "us", "apiKey": NEWS_API_KEY}
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    api_status["News API"] = "✅ Working"
                else:
                    api_status["News API"] = "⚠️ Error"
            except:
                api_status["News API"] = "❌ Failed"
        else:
            api_status["News API"] = "❌ No Key"
        
        # Test YouTube API
        if YOUTUBE_API_KEY:
            try:
                youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
                # Test with a simple search
                response = youtube.search().list(q="test", part='id', maxResults=1).execute()
                api_status["YouTube API"] = "✅ Working"
            except:
                api_status["YouTube API"] = "❌ Failed"
        else:
            api_status["YouTube API"] = "❌ No Key"
        
        # Test OpenAI API
        if OPENAI_API_KEY and openai_client:
            try:
                # Test with a simple completion
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                api_status["OpenAI API"] = "✅ Working"
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    api_status["OpenAI API"] = "⚠️ Quota Exceeded"
                else:
                    api_status["OpenAI API"] = "❌ Failed"
        else:
            api_status["OpenAI API"] = "❌ No Key"
        
        # Test Gemini API
        if GOOGLE_GEMINI_API_KEY and gemini_model:
            try:
                response = gemini_model.generate_content("Hello")
                api_status["Gemini API"] = "✅ Working"
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    api_status["Gemini API"] = "⚠️ Quota Exceeded"
                elif "expired" in error_msg.lower() or "invalid" in error_msg.lower():
                    api_status["Gemini API"] = "⚠️ Key Expired"
                else:
                    api_status["Gemini API"] = "❌ Failed"
        else:
            api_status["Gemini API"] = "❌ No Key"
        
        return api_status
    
    # Show API status
    if st.sidebar.button("🔄 Test API Status"):
        with st.spinner("Testing APIs..."):
            api_status = test_api_connectivity()
            
        for api_name, status in api_status.items():
            st.sidebar.text(f"{api_name}: {status}")
    else:
        # Show basic status without testing
        api_status = {}
        if NEWS_API_KEY:
            api_status["News API"] = "✅ Key Set"
        else:
            api_status["News API"] = "❌ No Key"
        
        if YOUTUBE_API_KEY:
            api_status["YouTube API"] = "✅ Key Set"
        else:
            api_status["YouTube API"] = "❌ No Key"
        
        if OPENAI_API_KEY:
            api_status["OpenAI API"] = "✅ Key Set"
        else:
            api_status["OpenAI API"] = "❌ No Key"
        
        if GOOGLE_GEMINI_API_KEY:
            api_status["Gemini API"] = "✅ Key Set"
        else:
            api_status["Gemini API"] = "❌ No Key"
        
        for api_name, status in api_status.items():
            st.sidebar.text(f"{api_name}: {status}")
        
        st.sidebar.info("💡 Click 'Test API Status' to check connectivity")

    # News options
    with st.sidebar.expander("📰 News Options"):
        st.session_state.news_days = st.slider("Days window", min_value=3, max_value=30, value=st.session_state.get('news_days', 7))
        st.session_state.prefer_indian_sources = st.checkbox("Prefer Indian sources", value=st.session_state.get('prefer_indian_sources', True))
    st.session_state.news_max_articles = st.slider("Articles to show", min_value=3, max_value=50, value=st.session_state.get('news_max_articles', 10))

    # AI provider options
    with st.sidebar.expander("🤖 AI Options"):
        st.session_state.ai_provider = st.selectbox(
            "Provider preference",
            options=["Auto", "Gemini first", "OpenAI first", "Gemini only", "OpenAI only"],
            index=["Auto", "Gemini first", "OpenAI first", "Gemini only", "OpenAI only"].index(st.session_state.get('ai_provider', 'Auto'))
        )

    # Advanced utilities
    with st.sidebar.expander("⚙️ Advanced"):
        if st.button("Clear cache (data)"):
            try:
                st.cache_data.clear()
                st.success("Cache cleared.")
            except Exception as e:
                st.warning(f"Could not clear cache: {e}")
    
    # Analysis type
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
            "AI Insights",
        ],
    )

    # Extra sidebar controls for certain modes
    if analysis_type == "Correlation Matrix":
        default_list = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA"
        st.session_state.corr_tickers = st.sidebar.text_input("Tickers (comma-separated)", value=st.session_state.get("corr_tickers", default_list))
    if analysis_type == "Sector Heatmaps":
        st.session_state.heatmap_period = st.sidebar.selectbox("Heatmap Period", ["1d"], index=0)

    # Main content
    if st.sidebar.button("🚀 Analyze", type="primary"):
        with st.spinner("Fetching data..."):
            # Get stock data (used in many modes)
            stock_data = dashboard.get_stock_data(symbol, period)
            st.session_state.stock_data = stock_data
            # If a different resolved symbol was used (e.g., added .NS), update session state
            if stock_data and stock_data.get('resolved_symbol') and stock_data.get('resolved_symbol') != symbol:
                st.session_state.symbol = stock_data['resolved_symbol']
                symbol = stock_data['resolved_symbol']
            # Basic stock header & chart (if any data)
            if stock_data and stock_data.get('current_price') is not None:
                validation = dashboard.validate_stock_symbol(symbol)
                if validation['valid'] and validation['name']:
                    st.subheader(f"📊 {validation['name']} ({symbol})")
                    st.caption(
                        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"[View on Yahoo Finance](https://finance.yahoo.com/quote/{symbol})"
                    )
                
                # Display current price and change
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cur_ccy = stock_data.get('currency')
                    st.metric(f"Current Price{f' ({cur_ccy})' if cur_ccy else ''}", f"{stock_data['current_price']:.2f}")
                
                with col2:
                    if stock_data.get('previous_close') and stock_data['previous_close'] is not None:
                        change = stock_data['current_price'] - stock_data['previous_close']
                        change_pct = (change / stock_data['previous_close']) * 100
                        st.metric(f"Change{f' ({cur_ccy})' if cur_ccy else ''}", f"{change:.2f}", f"{change_pct:.2f}%")
                    else:
                        st.metric("Change", "N/A")
                
                with col3:
                    if stock_data['info'].get('marketCap'):
                        market_cap = stock_data['info']['marketCap'] / 1e9
                        st.metric("Market Cap", f"{market_cap:.2f}B")
                    else:
                        st.metric("Market Cap", "N/A")
                
                with col4:
                    if stock_data['info'].get('volume'):
                        volume = stock_data['info']['volume'] / 1e6
                        st.metric("Volume", f"{volume:.1f}M")
                    else:
                        st.metric("Volume", "N/A")
                
                # Price chart
                st.subheader("📈 Price Chart")
                price_chart = create_price_chart(stock_data)
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
                    try:
                        hist_df = stock_data.get('history')
                        if hist_df is not None and not hist_df.empty:
                            csv_data = hist_df.to_csv().encode('utf-8')
                            st.download_button(
                                label="📥 Download Price History (CSV)",
                                data=csv_data,
                                file_name=f"{symbol}_{period}_history.csv",
                                mime='text/csv'
                            )
                    except Exception:
                        pass
                else:
                    st.warning("No price data available for this symbol. Please try a different stock symbol.")
            else:
                st.error(f"❌ Unable to fetch data for '{symbol}'. This symbol may be invalid, delisted, or not available.")
                st.info("💡 Try using a valid stock symbol like: AAPL, TSLA, GOOGL, MSFT, AMZN")
            
            # Fundamentals
            if analysis_type in ["Fundamentals", "Stock Analysis"]:
                render_fundamentals_section(symbol)

            # Technicals
            if analysis_type in ["Technicals", "Stock Analysis"]:
                render_technicals_section(symbol, stock_data)

            # News and sentiment analysis
            if analysis_type in ["News Sentiment", "Stock Analysis"]:
                render_news_section(symbol)

            # YouTube trends
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
                                if snippet.get('thumbnails', {}).get('medium', {}).get('url'):
                                    st.image(snippet['thumbnails']['medium']['url'], width=200)
                            
                            with col2:
                                st.write(f"**Channel:** {snippet['channelTitle']}")
                                st.write(f"**Published:** {snippet['publishedAt']}")
                                st.write(f"**Views:** {statistics.get('viewCount', 'N/A')}")
                                st.write(f"**Likes:** {statistics.get('likeCount', 'N/A')}")
                                st.write(f"**Description:** {snippet['description'][:200]}...")
                                
                                video_url = f"https://www.youtube.com/watch?v={video['id']}"
                                st.write(f"**URL:** {video_url}")
                else:
                    st.warning("⚠️ No YouTube data available. This could be due to:")
                    st.write("• Missing YouTube API key")
                    st.write("• No videos found for this symbol")
                    st.write("• API rate limits or connectivity issues")
                    st.info("💡 To enable YouTube analysis, please add your YouTube API key to the .env file")
                    
                    # Show demo YouTube data
                    st.subheader("📺 Demo YouTube Data")
                    demo_youtube = [
                        {
                            'id': 'demo_video_1',
                            'snippet': {
                                'title': f'Demo: {symbol} Stock Analysis Video',
                                'channelTitle': 'Demo Channel',
                                'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                                'description': f'Demo video discussing {symbol} stock performance and market analysis.',
                                'thumbnails': {
                                    'medium': {'url': 'https://via.placeholder.com/320x180?text=Demo+Video'}
                                }
                            },
                            'statistics': {
                                'viewCount': '1,234',
                                'likeCount': '56'
                            }
                        },
                        {
                            'id': 'demo_video_2',
                            'snippet': {
                                'title': f'Demo: {symbol} Market Update',
                                'channelTitle': 'Demo Channel',
                                'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                                'description': f'Demo video about {symbol} market trends and investor sentiment.',
                                'thumbnails': {
                                    'medium': {'url': 'https://via.placeholder.com/320x180?text=Demo+Video'}
                                }
                            },
                            'statistics': {
                                'viewCount': '2,345',
                                'likeCount': '78'
                            }
                        }
                    ]
                    
                    st.info("🔍 Showing demo YouTube data (add API key for real data)")
                    youtube_data = demo_youtube
                    
                    for video in youtube_data:
                        snippet = video['snippet']
                        statistics = video.get('statistics', {})
                        
                        with st.expander(f"🎥 {snippet['title']}"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if snippet.get('thumbnails', {}).get('medium', {}).get('url'):
                                    st.image(snippet['thumbnails']['medium']['url'], width=200)
                            
                            with col2:
                                st.write(f"**Channel:** {snippet['channelTitle']}")
                                st.write(f"**Published:** {snippet['publishedAt']}")
                                st.write(f"**Views:** {statistics.get('viewCount', 'N/A')}")
                                st.write(f"**Likes:** {statistics.get('likeCount', 'N/A')}")
                                st.write(f"**Description:** {snippet['description'][:200]}...")
                                
                                video_url = f"https://www.youtube.com/watch?v={video['id']}"
                                st.write(f"**URL:** {video_url}")

            # Sector Heatmaps
            if analysis_type == "Sector Heatmaps":
                render_sector_heatmap(st.session_state.get("heatmap_period", "1d"))

            # Correlation Matrix
            if analysis_type == "Correlation Matrix":
                render_correlation_matrix(st.session_state.get("corr_tickers", "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA"))

    # AI Insights (always available outside of Analyze click)
    if analysis_type in ["AI Insights", "Stock Analysis"]:
        st.subheader("🤖 AI-Powered Insights")
        
        # Use last fetched data if available
        sd_stock_data = st.session_state.get('stock_data')
        sd_news_data = st.session_state.get('news_data')
        
        if OPENAI_API_KEY or GOOGLE_GEMINI_API_KEY:
            st.subheader("💬 Ask Questions")

            # Example question shortcuts (set state before rendering the input)
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

            # Input + Ask button
            col1, col2 = st.columns([3, 1])
            with col1:
                user_query_p = st.text_input(
                    "Ask a question about the stock:",
                    placeholder="e.g., 'Why did Tesla fall today?' or 'Is Apple a good buy?'",
                    key="question_text"
                )
            with col2:
                st.write("")
                ask_button_p = st.button("🤖 Ask AI", key="ask_ai_persistent", type="primary", use_container_width=True)
            
            # Handle user question submission
            if user_query_p and ask_button_p:
                with st.spinner("🤖 AI is analyzing your question..."):
                    cur_price = sd_stock_data.get('current_price', 'N/A') if sd_stock_data else 'N/A'
                    prompt = f"Stock: {symbol}, Price: {cur_price}. Question: {user_query_p}. Give 2-3 sentence answer."
                    prov, txt = dashboard.ask_ai_unified(prompt, max_tokens=100)
                    if prov is None:
                        st.error(txt)
                    else:
                        st.success(f"**🤖 AI Answer ({prov}):**")
                        st.write(txt)

            # AI API Status cards
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

            # Quick AI connectivity test
            if st.button("🧪 Test AI APIs", key="test_ai_providers"):
                with st.spinner("Testing AI providers..."):
                    results = []
                    pref = st.session_state.get('ai_provider', 'Auto')
                    test_openai = pref in ['Auto', 'OpenAI first', 'Gemini first', 'OpenAI only']
                    test_gemini = pref in ['Auto', 'OpenAI first', 'Gemini first', 'Gemini only']
                    # OpenAI
                    if test_openai:
                        if OPENAI_API_KEY and (openai_client or legacy_openai):
                            ok, msg = test_openai_simple()
                            if ok:
                                results.append("✅ OpenAI: Working")
                            else:
                                results.append(f"❌ OpenAI: {msg}")
                        else:
                            results.append("❌ OpenAI: No Key")
                    # Gemini
                    if test_gemini:
                        if GOOGLE_GEMINI_API_KEY and gemini_model:
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

            # Demo AI Insights
            st.subheader("🤖 Demo AI Insights")
            if sd_stock_data:
                demo_summary = f"""
                Based on the available data for {symbol}:
                
                📊 **Current Status**: {symbol} is currently trading at {sd_stock_data.get('current_price', 'N/A'):.2f}
                
                📈 **Key Factors**: 
                • Market sentiment appears to be mixed
                • Recent trading volume suggests moderate investor interest
                • Technical indicators show potential for both upside and downside movement
                
                🔮 **Outlook**: Short-term outlook remains uncertain, with potential for volatility in the coming days.
                
                *This is a demo summary. Add AI API keys for real AI-powered insights.*
                """
                st.write(demo_summary)

            # Real AI functionality (when APIs are configured)
            if (OPENAI_API_KEY or GOOGLE_GEMINI_API_KEY) and sd_stock_data and sd_news_data:
                st.write("**Market Summary:**")
                summary = dashboard.generate_ai_summary(sd_stock_data, sd_news_data, symbol)
                st.write(summary)
                if isinstance(summary, str) and ("quota exceeded" in summary.lower() or "api key" in summary.lower() or "error" in summary.lower()):
                    st.warning("⚠️ API limits or configuration issues detected. Showing demo summary as fallback:")
                    demo_summary = f"""
                    Based on the available data for {symbol}:
                    
                    📊 **Current Status**: {symbol} is currently trading at {sd_stock_data.get('current_price', 'N/A'):.2f}
                    
                    📈 **Key Factors**: 
                    • Market sentiment appears to be mixed
                    • Recent trading volume suggests moderate investor interest
                    • Technical indicators show potential for both upside and downside movement
                    
                    📰 **News Impact**: Recent news articles indicate varying sentiment about {symbol}'s performance
                    
                    🔮 **Outlook**: Short-term outlook remains uncertain, with potential for volatility in the coming days.
                    
                    *This is a demo summary. Add API keys and check billing for real AI-powered insights.*
                    """
                    st.write(demo_summary)
    
    # Standalone refresh for News to apply sidebar options without re-running Analyze
    if analysis_type == "News Sentiment" and st.button("🔄 Refresh News", use_container_width=True):
        render_news_section(st.session_state.get('symbol', 'AAPL'))

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit | Powered by Multiple APIs | Real-time Market Insights</p>
    </div>
    """, unsafe_allow_html=True)

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
    # Use existing history when available, else fetch
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

    # Indicators
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

    # Signals summary
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

    # Pivot-based support/resistance (last day)
    pivots = None
    try:
        last_row = df.iloc[-1]
        P = (last_row['High'] + last_row['Low'] + last_row['Close']) / 3
        R1 = 2 * P - last_row['Low']
        S1 = 2 * P - last_row['High']
        R2 = P + (last_row['High'] - last_row['Low'])
        S2 = P - (last_row['High'] - last_row['Low'])
        pivots = {"Pivot": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}
    except Exception:
        pass

    # Plot
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

    # Signals/Pivots
    if signals:
        st.info("\n".join([f"• {s}" for s in signals]))
    if pivots:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Pivot", f"{pivots['Pivot']:.2f}")
        c2.metric("R1", f"{pivots['R1']:.2f}")
        c3.metric("S1", f"{pivots['S1']:.2f}")
        c4.metric("R2", f"{pivots['R2']:.2f}")
        c5.metric("S2", f"{pivots['S2']:.2f}")


def render_fundamentals_section(symbol: str):
    st.subheader("🏦 Fundamentals")
    tkr = yf.Ticker(symbol)

    # Key ratios
    ratios = {}
    try:
        info = tkr.info
        keys = [
            'trailingPE', 'forwardPE', 'priceToBook', 'profitMargins', 'grossMargins',
            'operatingMargins', 'returnOnEquity', 'returnOnAssets', 'debtToEquity',
            'currentRatio', 'quickRatio', 'dividendYield', 'payoutRatio'
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

    # Statements
    def show_df(df: pd.DataFrame, title: str):
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.markdown(f"**{title}**")
            st.dataframe(df.fillna('').astype(str).iloc[:20])
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

    # Earnings calendar
    with st.expander("Earnings Calendar"):
        try:
            ed = tkr.get_earnings_dates(limit=4)
            st.dataframe(ed)
        except Exception:
            try:
                cal = tkr.calendar
                st.dataframe(cal)
            except Exception as e:
                st.info(f"Unavailable: {e}")

    # Holders/Insiders
    with st.expander("Institutional / Major Holders"):
        try:
            ih = tkr.institutional_holders
            if ih is not None and not ih.empty:
                st.dataframe(ih)
            mh = tkr.major_holders
            if mh is not None and not mh.empty:
                st.dataframe(mh)
        except Exception as e:
            st.info(f"Unavailable: {e}")
    with st.expander("Insider Transactions"):
        try:
            it = getattr(tkr, 'insider_transactions', None)
            if it is not None and isinstance(it, pd.DataFrame) and not it.empty:
                st.dataframe(it)
            else:
                st.info("No recent insider transactions found.")
        except Exception as e:
            st.info(f"Unavailable: {e}")


def render_sector_heatmap(period: str = "1d"):
    st.subheader("🗺️ Sector Heatmap")
    # SPDR sector ETFs
    sectors = {
        'Energy': 'XLE', 'Technology': 'XLK', 'Financials': 'XLF', 'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP', 'Industrials': 'XLI', 'Health Care': 'XLV', 'Utilities': 'XLU',
        'Materials': 'XLB', 'Real Estate': 'XLRE', 'Communication': 'XLC'
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

# Ensure the app runs when executed by Streamlit
if __name__ == "__main__":
    main()

