import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Gemini API Configuration
api_key_secrectpass = st.secrets["api_key"]
genai.configure(api_key=api_key_secrectpass)


# Top Company Symbols
TOP_STOCKS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 
    'META', 'TSLA', 'NFLX', 'INTC', 'ADBE'
]

class AIStockAdvisor:
    def __init__(self):
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-pro')

    def fetch_stock_data(self, symbol, period='1mo'):
        """
        Fetch stock data for a given symbol
        
        Args:
            symbol (str): Stock symbol
            period (str): Data retrieval period
        
        Returns:
            pandas.DataFrame: Stock price data
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return None

    def get_stock_news(self, symbol):
        """
        Fetch recent news for a stock using web scraping
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            list: Recent news articles
        """
        try:
            # Use Yahoo Finance company name
            stock = yf.Ticker(symbol)
            company_name = stock.info.get('longName', symbol)
            
            # Google News search URL
            url = f"https://news.google.com/search?q={company_name}&hl=en-US&gl=US&ceid=US%3Aen"
            
            # Fetch news
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract news articles
            articles = []
            for article in soup.find_all('article', limit=5):
                title_elem = article.find('h3')
                link_elem = article.find('a')
                
                if title_elem and link_elem:
                    title = title_elem.text
                    link = f"https://news.google.com{link_elem['href'][1:]}"
                    articles.append({
                        'title': title,
                        'link': link
                    })
            
            return articles
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            return []

    def generate_ai_insights(self, stock_data, symbol):
        """
        Generate AI-powered stock insights using Gemini
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
            symbol (str): Stock symbol
        
        Returns:
            str: AI-generated stock insights
        """
        try:
            # Prepare context for analysis
            last_close = stock_data['Close'].iloc[-1]
            price_change = stock_data['Close'].pct_change().iloc[-1] * 100
            volume = stock_data['Volume'].iloc[-1]

            # Construct prompt for Gemini
            prompt = f"""Analyze the stock performance for {symbol}:
            - Current Price: ${last_close:.2f}
            - Price Change: {price_change:.2f}%
            - Trading Volume: {volume:,}
            
            Provide a comprehensive investment analysis including:
            1. Short-term trend assessment
            2. Potential investment risks
            3. Key market sentiments
            4. Recommended action for investors
            """

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating AI insights: {e}")
            return "Unable to generate insights at this moment."

    def plot_stock_trend(self, stock_data):
        """
        Create interactive stock price trend visualization
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
        
        Returns:
            plotly figure: Stock price trend chart
        """
        fig = px.line(
            stock_data, 
            x=stock_data.index, 
            y=['Close', 'Open'], 
            title='Stock Price Trends',
            labels={'value': 'Price', 'variable': 'Type'}
        )
        return fig

def main():
    st.title('🚀 AI Stock Advisor')
    st.sidebar.header('Stock Selection')

    # Initialize advisor
    advisor = AIStockAdvisor()

    # Stock symbol selection
    selected_symbol = st.sidebar.selectbox(
        'Choose a Stock', 
        TOP_STOCKS, 
        index=0
    )

    # Fetch and display stock data
    stock_data = advisor.fetch_stock_data(selected_symbol)
    
    if stock_data is not None:
        # Stock Price Trend Visualization
        st.plotly_chart(advisor.plot_stock_trend(stock_data))

        # Stock Details Column Layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Current Price", 
                value=f"${stock_data['Close'].iloc[-1]:.2f}"
            )
        
        with col2:
            st.metric(
                label="Price Change", 
                value=f"{stock_data['Close'].pct_change().iloc[-1]*100:.2f}%"
            )

        # AI-Generated Insights
        st.subheader('AI Stock Analysis')
        ai_insights = advisor.generate_ai_insights(stock_data, selected_symbol)
        st.write(ai_insights)

        # Recent News
        st.subheader('Recent News')
        news_articles = advisor.get_stock_news(selected_symbol)
        for article in news_articles:
            st.markdown(f"**{article['title']}**\n{article['link']}")

if __name__ == "__main__":
    main()

# Required dependencies (requirements.txt)
# streamlit
# yfinance
# google-generativeai
# plotly
# requests
# beautifulsoup4
