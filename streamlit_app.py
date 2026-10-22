"""
10-K Sentiment Analyzer
-----------------------
Downloads a company's 10-K filings from SEC EDGAR, scores each section with
FinBERT, and shows how the tone of the filings changes over the years.

Run with:  streamlit run streamlit_app.py
Share a result with a link:  http://localhost:8501/?ticker=GOOG&from=2019
"""

import html
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer

from sec_download import download_filings, load_filings

# The sections of a 10-K report: item number and a short readable name
SECTIONS = {
    "item_1": ("1", "Business overview"),
    "item_1A": ("1A", "Risk factors"),
    "item_1B": ("1B", "Staff comments"),
    "item_1C": ("1C", "Cybersecurity"),
    "item_2": ("2", "Properties"),
    "item_3": ("3", "Legal proceedings"),
    "item_4": ("4", "Mine safety"),
    "item_5": ("5", "Equity market"),
    "item_6": ("6", "Financial summary"),
    "item_7": ("7", "Management analysis"),
    "item_7A": ("7A", "Market risks"),
    "item_8": ("8", "Financial statements"),
    "item_9": ("9", "Accountant changes"),
    "item_9A": ("9A", "Controls and procedures"),
    "item_9B": ("9B", "Other information"),
    "item_9C": ("9C", "Foreign inspections"),
    "item_10": ("10", "Corporate governance"),
    "item_11": ("11", "Executive pay"),
    "item_12": ("12", "Shareholder info"),
    "item_13": ("13", "Related parties"),
    "item_14": ("14", "Audit fees"),
    "item_15": ("15", "Exhibits"),
    "item_16": ("16", "10-K summary"),
}
NEUTRAL = 3.0


# ---------- 1. Model ----------

@st.cache_resource(show_spinner=False)
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


@st.cache_data(show_spinner=False, ttl="1d")
def score_filings(ticker, start_year, _filings, _tokenizer, _model):
    """Build a table: one row per 10-K section, one column per year (empty sections are NaN)."""
    scores = {}
    for filing in _filings:
        scores[filing["year"]] = {
            key: sentiment_score(filing[key], _tokenizer, _model) if filing.get(key, "").strip() else np.nan
            for key in SECTIONS
        }
    df = pd.DataFrame(scores)
    return df[sorted(df.columns)].dropna(how="all")  # drop sections that are always empty


@st.cache_data(show_spinner=False, ttl="1h")
def fetch_news(query):
    """Get the latest news headlines from NewsAPI (needs NEWSAPI_KEY). None = no key."""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return None
    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={"q": f'"{query}"', "sortBy": "publishedAt", "pageSize": 5, "apiKey": api_key},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("articles", [])


# ---------- 3. Building blocks (HTML) ----------

def esc(text):
    return html.escape(str(text)).replace("$", "&#36;")  # "$" would start a maths formula in markdown


def signed(x):
    """+1.81 / −0.05 with a real minus sign."""
    return f"{x:+.2f}".replace("-", "−")


def tone_word(score):
    if score >= 3.5:
        return "positive"
    if score <= 2.5:
        return "negative"
    return "neutral"


def tint(score):
    """Colour a grid cell only when the tone is clearly off neutral."""
    if np.isnan(score):
        return "na"
    if score >= 4.5:
        return "p2"
    if score >= 3.5:
        return "p1"
    if score <= 1.5:
        return "n2"
    if score <= 2.5:
        return "n1"
    return ""


def section_label(key):
    code, name = SECTIONS[key]
    return f"{esc(name)}<span>Item {code}</span>"


def line_chart(years, values, width, max_labels):
    """Overall tone per year as a plain SVG line on a 1-5 scale, with neutral (3) dashed."""
    height, left, right, top, bottom = 220, 28, 48, 12, 28
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(years)

    def x(i):
        return left + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    def y(v):
        return top + plot_h * (5 - v) / 4

    svg = [f'<svg class="{"desktop" if max_labels > 4 else "phone"}" viewBox="0 0 {width} {height}" '
           f'role="img" aria-label="Overall tone by year">']
    for t in range(1, 6):  # gridlines; the baseline (1) is the one darker line
        colour = "#cfcfcf" if t == 1 else "#ececec"
        svg.append(f'<line x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{colour}"/>')
        svg.append(f'<text x="{left - 10}" y="{y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    svg.append(f'<line x1="{left}" x2="{width - right}" y1="{y(NEUTRAL):.1f}" y2="{y(NEUTRAL):.1f}" '
               f'stroke="#8a8a8a" stroke-dasharray="3 3"/>')

    step = math.ceil(n / max_labels)
    for i in range(n - 1, -1, -step):  # x labels, always including the last year
        svg.append(f'<text x="{x(i):.1f}" y="{height - 8}" text-anchor="middle">{years[i]}</text>')

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    svg.append(f'<polyline points="{points}" fill="none" stroke="#1d5fbf" stroke-width="2" '
               f'stroke-linejoin="round"/>')
    for i, v in enumerate(values):  # hover targets with the exact value
        svg.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="12" fill="transparent">'
                   f'<title>{years[i]}: {v:.2f}</title></circle>')
    svg.append(f'<circle cx="{x(n - 1):.1f}" cy="{y(values[-1]):.1f}" r="3" fill="#1d5fbf"/>')
    svg.append(f'<text class="end" x="{x(n - 1) + 8:.1f}" y="{y(values[-1]) + 4:.1f}">{values[-1]:.2f}</text>')
    svg.append("</svg>")
    return "".join(svg)


def grid_table(df, keys):
    years = list(df.columns)
    head = "".join(f"<th>{y}</th>" for y in years)
    rows = []
    for key in keys:
        cells = "".join(
            f'<td class="{tint(s)}">{"–" if np.isnan(s) else f"{s:.2f}"}</td>' for s in df.loc[key]
        )
        rows.append(f'<tr><td>{section_label(key)}</td>{cells}<td class="avg">{df.loc[key].mean():.2f}</td></tr>')
    return (f'<div class="table-frame"><table class="grid"><thead><tr><th>Section</th>{head}'
            f'<th>Average</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def page_title(title, sentence, scope="", error=False):
    cls = "notice error" if error else "finding"
    scope_html = f'<p class="scope">{scope}</p>' if scope else ""
    return f'<div class="title-row"><h1>{esc(title)}</h1><p class="{cls}">{sentence}</p>{scope_html}</div>'


BRAND = """
<div class="brand">
  <svg width="22" height="26" viewBox="0 0 22 26" aria-hidden="true">
    <path d="M1 1h14l6 6v18H1z" fill="#fff" stroke="#0a0a0a" stroke-width="1.5" stroke-linejoin="round"/>
    <path d="M15 1v6h6" fill="none" stroke="#0a0a0a" stroke-width="1.5" stroke-linejoin="round"/>
    <rect x="5" y="11" width="12" height="2" fill="#0a0a0a"/>
    <rect x="5" y="15" width="7" height="2" fill="#1d5fbf"/>
    <rect x="5" y="19" width="10" height="2" fill="#0a0a0a"/>
  </svg>
  <div><b>10-K Sentiment Analyzer</b><span>Annual report tone, scored with FinBERT</span></div>
</div>
"""

METHOD = (
    '<p class="method">Each section is scored with FinBERT (yiyanghkust/finbert-tone) from its first 512 tokens: '
    "3 × neutral + 5 × positive + 1 × negative, so 1 is negative, 3 neutral and 5 positive. "
    "Filings come from SEC EDGAR via edgar-crawler.</p>"
)


def report(df, ticker, company, cik):
    """The results screen: title and finding, headline figures, trend, changes, the section grid."""
    years = list(df.columns)
    first, last = years[0], years[-1]
    overall = df.mean().round(2)  # one figure per year: the mean of its scored sections

    # Everything below works from the rounded scores on screen, so the figures always add up
    df = df.round(2)
    averages = df.mean(axis=1)  # the same numbers as the grid's "Average" column
    best, worst = averages.idxmax(), averages.idxmin()
    changes = (df[last] - df[first]).dropna().round(2)
    changes = changes[changes.abs() >= 0.045]
    movers = changes.loc[changes.abs().sort_values(ascending=False).index].head(4)

    # The one finding, shown under the title
    sentence = f"The {last} report reads {tone_word(overall[last])} overall ({overall[last]:.2f} of 5)."
    if len(movers):
        top = movers.index[0]
        sentence = esc(sentence) + (f" {esc(SECTIONS[top][1])} changed most since {first}: "
                                    f'<span class="nowrap">{df.loc[top, first]:.2f} → {df.loc[top, last]:.2f}</span>.')
    else:
        sentence = esc(sentence)
    scope = f"{esc(ticker)} · CIK {esc(cik)} · {len(years)} annual reports, {first}–{last}"
    out = [page_title(company, sentence, scope)]

    # Headline figures
    if len(years) > 1:
        delta = round(overall[last] - overall[first], 2)
        cls = "up" if delta >= 0.045 else "down" if delta <= -0.045 else ""
        lead_sub = f'<span class="{cls}">{signed(delta)}</span> vs {first}'
    else:
        lead_sub = "–"
    out.append(f"""
    <div class="card kpis">
      <div class="kpi"><div class="label">Overall tone, {last}</div>
        <div class="value lead num">{overall[last]:.2f}</div><div class="sub num">{lead_sub}</div></div>
      <div class="kpi"><div class="label">Most positive section</div>
        <div class="value name">{esc(SECTIONS[best][1])}</div>
        <div class="sub num">{averages[best]:.2f} average · Item {SECTIONS[best][0]}</div></div>
      <div class="kpi"><div class="label">Most negative section</div>
        <div class="value name">{esc(SECTIONS[worst][1])}</div>
        <div class="sub num">{averages[worst]:.2f} average · Item {SECTIONS[worst][0]}</div></div>
      <div class="kpi"><div class="label">Reports scored</div>
        <div class="value num">{len(years)}</div><div class="sub num">{first}–{last}</div></div>
    </div>""")

    # Trend and biggest changes, side by side
    values = [float(overall[y]) for y in years]
    if len(movers):
        rows = "".join(
            f'<li><div class="name">{section_label(k)}</div>'
            f'<div class="change num"><span class="{"up" if c > 0 else "down"}">{signed(c)}</span>'
            f'<span class="from">{df.loc[k, first]:.2f} → {df.loc[k, last]:.2f}</span></div></li>'
            for k, c in movers.items()
        )
        movers_html = f'<ul class="movers">{rows}</ul>'
    elif len(years) == 1:
        movers_html = '<div class="empty">One report in this range. Changes need at least two years.</div>'
    else:
        movers_html = f'<div class="empty">No section moved by 0.05 or more between {first} and {last}.</div>'
    out.append(f"""
    <div class="row-2">
      <div class="card"><div class="card-head"><h2>Overall tone by year</h2>
        <span class="meta">Mean of all scored sections · dashed line is neutral (3)</span></div>
        <div class="chart">{line_chart(years, values, 680, 8)}{line_chart(years, values, 326, 4)}</div></div>
      <div class="card"><div class="card-head"><h2>Biggest changes</h2>
        <span class="meta">{first} → {last}</span></div>{movers_html}</div>
    </div>""")

    # The 10-K itself: every section by year, tinted only where the tone is clearly off neutral
    moving = [k for k in df.index if any(tint(s) not in ("", "na") for s in df.loc[k])]
    quiet = [k for k in df.index if k not in moving]
    quiet_html = ""
    if quiet:
        quiet_html = (f'<details class="neutral"><summary>{len(quiet)} more sections stay neutral '
                      f"(2.5–3.5) every year</summary>{grid_table(df, quiet)}</details>")
    out.append(f"""
    <div class="card"><div class="card-head"><h2>Tone by section and year</h2>
      <div class="legend"><span><i class="n2"></i>1–1.5</span><span><i class="n1"></i>1.5–2.5</span>
        <span><i></i>neutral</span><span><i class="p1"></i>3.5–4.5</span><span><i class="p2"></i>4.5–5</span></div>
      </div>
      {grid_table(df, moving) if moving else '<div class="empty">Every section stays neutral (2.5–3.5) in every year.</div>'}
      {quiet_html}
    </div>""")
    return f'<div class="stack">{"".join(out)}</div>'


def news_card(company):
    try:
        articles = fetch_news(company)
    except requests.RequestException:
        body = '<div class="empty">Couldn\'t load news from NewsAPI. Try again later.</div>'
    else:
        if articles is None:
            body = ('<div class="empty">Recent news appears here once the NEWSAPI_KEY environment variable '
                    "(or Streamlit secret) is set.</div>")
        elif not articles:
            body = '<div class="empty">NewsAPI has no recent articles about this company.</div>'
        else:
            body = '<ul class="news">' + "".join(
                f'<li><a href="{esc(a["url"])}" target="_blank" rel="noopener">{esc(a["title"])}</a>'
                f'<span>{esc(a["source"]["name"])} · {esc(a["publishedAt"][:10])}</span></li>'
                for a in articles
            ) + "</ul>"
    return (f'<div class="card"><div class="card-head"><h2>Latest news</h2>'
            f'<span class="meta">NewsAPI</span></div>{body}</div>')


# ---------- 4. The screen ----------

def show(target, markup):
    """Render our HTML. (st.html strips SVG, so use markdown, on one line so it stays raw HTML.)"""
    target.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


st.set_page_config(page_title="10-K Sentiment Analyzer", page_icon=":material/description:", layout="wide")
st.html(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>")

this_year = datetime.now().year
params = st.query_params


def year_param():
    """The start year from the link (?from=2019), kept within range."""
    try:
        return min(max(int(st.query_params.get("from", 2019)), 2001), this_year)
    except ValueError:
        return 2019


default_year = year_param()

top = st.columns([1, 1.3], vertical_alignment="bottom")
show(top[0], BRAND)
with top[1].form("search", border=False):
    c1, c2, c3 = st.columns([1.2, 1, 0.8], vertical_alignment="bottom")
    ticker_in = c1.text_input("Stock ticker", params.get("ticker", "GOOG"), max_chars=10)
    year_in = c2.number_input("From year", 2001, this_year, default_year)
    if c3.form_submit_button("Analyze", type="primary", use_container_width=True):
        st.query_params.update({"ticker": ticker_in.strip().upper(), "from": str(int(year_in))})

ticker = st.query_params.get("ticker", "").strip().upper()
start_year = year_param()
screen = st.empty()

if not ticker:
    examples = " · ".join(f'<a href="?ticker={t}&from=2019" target="_self">{t}</a>' for t in ("GOOG", "MSFT", "JPM"))
    show(screen, page_title(
        "Annual report tone",
        "Enter a stock ticker to score every section of the company's 10-K filings, year by year, "
        f"from 1 (negative) to 5 (positive). Try {examples}.",
    ) + METHOD)
    st.stop()

try:
    show(screen, page_title(ticker, "Downloading 10-K filings from SEC EDGAR. The first run for a company "
                                   "takes a minute or two."))
    filings = get_filings(ticker, start_year)
except ValueError:
    show(screen, page_title(ticker, f"{esc(ticker)} isn't in the SEC's list of company tickers. "
                                   "Check the symbol, for example GOOG, MSFT or JPM.", error=True))
    st.stop()
except Exception:
    show(screen, page_title(ticker, "Couldn't download filings from SEC EDGAR. Check the connection and "
                                   "press Analyze to try again.", error=True))
    st.stop()

if not filings:
    show(screen, page_title(ticker, f"No 10-K filings for {esc(ticker)} from {start_year} on. "
                                   "Try an earlier start year.", error=True))
    st.stop()

company = filings[0].get("company", ticker)
show(screen, page_title(company, f"Scoring {len(filings)} annual reports with FinBERT…"))
tokenizer, model = load_finbert()
scores = score_filings(ticker, start_year, filings, tokenizer, model)

show(screen, report(scores, ticker, company, filings[0].get("cik", "–")))
show(st, news_card(company) + METHOD)
