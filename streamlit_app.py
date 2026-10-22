"""
10-K Sentiment Analyzer
-----------------------
Downloads a company's 10-K filings from SEC EDGAR, scores each section with
FinBERT, and shows how the tone of the filings changes over the years.

Run with:  streamlit run streamlit_app.py
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer

from sec_download import download_filings, load_filings

# The sections of a 10-K report and a short readable name for each
SECTIONS = {
    "item_1": "Business Overview",
    "item_1A": "Risk Factors",
    "item_1B": "Staff Comments",
    "item_1C": "Cybersecurity",
    "item_2": "Properties",
    "item_3": "Legal Proceedings",
    "item_4": "Mine Safety",
    "item_5": "Equity Market",
    "item_6": "Financial Summary",
    "item_7": "Management Analysis",
    "item_7A": "Market Risks",
    "item_8": "Financial Statements",
    "item_9": "Accountant Changes",
    "item_9A": "Controls & Procedures",
    "item_9B": "Other Information",
    "item_9C": "Foreign Inspections",
    "item_10": "Corporate Governance",
    "item_11": "Executive Pay",
    "item_12": "Shareholder Info",
    "item_13": "Related Parties",
    "item_14": "Audit Fees",
    "item_15": "Exhibits",
    "item_16": "10-K Summary",
}


# ---------- 1. Model ----------

@st.cache_resource
def load_finbert():
    """Load FinBERT, a BERT model trained on financial text (cached after first run)."""
    name = "yiyanghkust/finbert-tone"
    return BertTokenizer.from_pretrained(name), BertForSequenceClassification.from_pretrained(name)


def sentiment_score(text, tokenizer, model):
    """
    Return a score from 1 (negative) to 5 (positive).

    FinBERT gives three probabilities: [neutral, positive, negative].
    We turn them into one number with a weighted average: 3, 5 and 1.
    """
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    logits = model(**inputs).logits.detach().numpy()[0]
    probs = np.exp(logits) / np.exp(logits).sum()  # softmax
    return float(np.dot(probs, [3, 5, 1]))


# ---------- 2. Data ----------

@st.cache_data(show_spinner=False, ttl="1d")
def get_filings(ticker, start_year):
    """Download and load the filings (cached, so repeating a search is instant)."""
    download_filings(ticker, start_year)
    return load_filings()


@st.cache_data(show_spinner=False)
def score_filings(filings, _tokenizer, _model):
    """Build a table: one row per year, one column per 10-K section."""
    tokenizer, model = _tokenizer, _model
    rows = []
    for filing in filings:
        row = {"Year": filing["year"]}
        for key, name in SECTIONS.items():
            text = filing.get(key, "")
            row[name] = sentiment_score(text, tokenizer, model) if text.strip() else None
        rows.append(row)

    df = pd.DataFrame(rows).dropna(axis=1, how="all")  # drop sections that are always empty
    df["Overall"] = df.drop(columns="Year").mean(axis=1)
    return df


def fetch_news(query):
    """Get the latest news headlines from NewsAPI (needs NEWSAPI_KEY)."""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return None
    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={"q": query, "sortBy": "publishedAt", "pageSize": 5, "apiKey": api_key},
        timeout=10,
    )
    return response.json().get("articles", []) if response.ok else []


# ---------- 3. User interface ----------

st.set_page_config(page_title="10-K Sentiment Analyzer", page_icon="📈")
st.title("📈 10-K Sentiment Analyzer")
st.write(
    "Score the tone of a company's annual reports (10-K filings) with **FinBERT**. "
    "Scores go from **1 (negative)** to **5 (positive)**, and **3 is neutral**."
)

col1, col2 = st.columns(2)
ticker = col1.text_input("Stock ticker", "GOOG").strip().upper()
start_year = col2.number_input("Start year", min_value=2001, max_value=datetime.now().year, value=2019)

if st.button("Analyze", type="primary"):
    with st.spinner("Downloading 10-K filings from SEC EDGAR... (this can take a few minutes)"):
        filings = get_filings(ticker, int(start_year))

    if not filings:
        st.error(f"No 10-K filings found for {ticker}.")
        st.stop()

    company = filings[0].get("company", ticker)
    st.success(f"Found {len(filings)} filings for {company} ({filings[0]['year']}–{filings[-1]['year']})")

    with st.spinner("Scoring each section with FinBERT..."):
        tokenizer, model = load_finbert()
        df = score_filings(filings, tokenizer, model)

    st.subheader("Overall tone by year")
    fig = px.line(df, x="Year", y="Overall", markers=True, range_y=[1, 5])
    fig.add_hline(y=3, line_dash="dot", annotation_text="neutral")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score for each section")
    st.caption("Red = negative, yellow = neutral, green = positive")
    score_cols = [c for c in df.columns if c != "Year"]
    st.dataframe(
        df.style.background_gradient(cmap="RdYlGn", vmin=1, vmax=5, subset=score_cols)
        .highlight_null(props="background-color: transparent; color: transparent")  # hide empty sections
        .format("{:.2f}", subset=score_cols),
        hide_index=True,
    )

    st.subheader("Latest news")
    articles = fetch_news(company)
    if articles is None:
        st.info("Set the NEWSAPI_KEY environment variable to see related news.")
    elif not articles:
        st.write("No news found.")
    for article in articles or []:
        st.markdown(f"**[{article['title']}]({article['url']})** — {article['source']['name']}")
