# 📈 10-K Sentiment Analyzer

Measure the **tone of a company's annual reports** (10-K filings) with NLP, and see how it changes over the years.

## The idea

Every US public company files a **10-K** report with the SEC each year. These reports are long and hard to read.
We use **FinBERT**, a language model trained on financial text, to give each section of the report a score:

| Score | Meaning  |
|:-----:|----------|
| 1     | Negative |
| 3     | Neutral  |
| 5     | Positive |

## How it works

```
Ticker (e.g. GOOG)
   │
   ▼
1. Download 10-K filings from SEC EDGAR      (edgar-crawler)
   │
   ▼
2. Split each filing into sections            (Business, Risk Factors, ...)
   │
   ▼
3. Score each section                         (FinBERT)
   │
   ▼
4. Show a trend chart, a colour-coded table and the latest news
```

**The score:** FinBERT returns three probabilities (neutral, positive, negative).
We combine them with a weighted average:

```
score = 3 × neutral + 5 × positive + 1 × negative
```

## Project files

| File | What it is |
|------|------------|
| `streamlit_app.py` | The web app (enter a ticker, click **Analyze**) |
| `analysis_notebook.ipynb` | Step-by-step walkthrough of the same analysis (with results for GOOG) |
| `sec_download.py` | Downloads the filings from SEC EDGAR and loads them |
| `requirements.txt` | Python packages to install |
| `.streamlit/config.toml` | Streamlit settings |

## Run it

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
.venv/Scripts/activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501, enter a ticker (e.g. `GOOG`) and a start year, and click **Analyze**.
A run for GOOG from 2019 takes about 1–2 minutes (the very first run also downloads the FinBERT model).

**Optional:** to show related news, get a free key from [NewsAPI](https://newsapi.org/) and set it before starting the app:

```bash
export NEWSAPI_KEY=your_key        # Windows PowerShell: $env:NEWSAPI_KEY="your_key"
```

## Deploy (Streamlit Community Cloud, free)

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app** and pick the repository.
3. Set **Main file path** to `streamlit_app.py` and click **Deploy**.
4. Optional: under **Advanced settings → Secrets**, add your NewsAPI key:
   ```toml
   NEWSAPI_KEY = "your_key"
   ```

The first start takes a few minutes while the packages and the FinBERT model download.

## Tools used

- **[FinBERT](https://huggingface.co/yiyanghkust/finbert-tone)** – sentiment model for financial text
- **[edgar-crawler](https://github.com/nlpaueb/edgar-crawler)** – downloads and splits 10-K filings (fetched automatically on first run)
- **[Streamlit](https://streamlit.io/)** – web interface
- **[Plotly](https://plotly.com/python/)** – charts
- **[NewsAPI](https://newsapi.org/)** / **[TextRazor](https://www.textrazor.com/)** – news and named-entity recognition (optional)

## Limitations

- BERT reads at most 512 tokens, so only the **start of each section** is scored.
- 10-K reports are written in a careful, neutral tone, so most scores stay close to 3.
- Downloading filings for many years can take a few minutes.
