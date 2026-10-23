"""
Download 10-K filings from SEC EDGAR and split them into sections,
using the open-source edgar-crawler tool (https://github.com/nlpaueb/edgar-crawler).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime

import requests

USER_AGENT = "Student Project (student@example.com)"  # the SEC asks every request for a contact
CRAWLER_DIR = "edgar-crawler"
CRAWLER_URL = "https://github.com/nlpaueb/edgar-crawler.git"
CRAWLER_COMMIT = "84a8d0c5dd7dd6769526e5ccec534c4e0880d56f"  # the version this project was tested with
EXTRACTED_DIR = os.path.join(CRAWLER_DIR, "datasets", "EXTRACTED_FILINGS", "10-K")
INDICES_DIR = os.path.join(CRAWLER_DIR, "datasets", "INDICES")


class UnknownTicker(Exception):
    """The ticker is not in the SEC's list of companies."""


# All downloads share one folder, so only one may run at a time (matters when several people use the app)
LOCK = threading.Lock()


def setup_crawler():
    """Clone edgar-crawler (if needed) and fix its filing links."""
    script = os.path.join(CRAWLER_DIR, "download_filings.py")
    if not os.path.exists(script):
        subprocess.run(["git", "clone", "-q", CRAWLER_URL, CRAWLER_DIR], check=True)
        subprocess.run(["git", "-C", CRAWLER_DIR, "checkout", "-q", CRAWLER_COMMIT], check=True)

    # The SEC now blocks short links like  .../1652044/0001652044-19-000004-index.html
    # and only allows the full form       .../1652044/000165204419000004/0001652044-19-000004-index.html
    with open(script, encoding="utf-8") as f:
        code = f.read()
    old = 'df["html_index"].astype('
    new = ('df["html_index"].str.replace(r"(\\d+)/(\\d{10})-(\\d{2})-(\\d{6})-index", '
           'r"\\1/\\2\\3\\4/\\2-\\3-\\4-index", regex=True).astype(')
    if old in code:
        with open(script, "w", encoding="utf-8") as f:
            f.write(code.replace(old, new))
    elif new not in code:
        raise RuntimeError("edgar-crawler has changed: could not apply the SEC link fix")


def remove_stale_indices():
    """
    edgar-crawler never downloads an index file twice, so a quarter saved before it ended
    would miss every filing made after that day. Remove those files so they are fetched again.
    """
    if not os.path.isdir(INDICES_DIR):
        return
    for name in os.listdir(INDICES_DIR):
        match = re.fullmatch(r"(\d{4})_QTR([1-4])\.tsv", name)
        if match:
            year, quarter = int(match[1]), int(match[2])
            quarter_end = datetime(year + quarter // 4, quarter * 3 % 12 + 1, 1)
            path = os.path.join(INDICES_DIR, name)
            if datetime.fromtimestamp(os.path.getmtime(path)) < quarter_end:
                os.remove(path)


def find_cik(ticker):
    """Look the ticker up in the SEC's official list of companies. Returns its CIK number, or None."""
    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": USER_AGENT}, timeout=30
    )
    response.raise_for_status()
    for company in response.json().values():
        if company["ticker"] == ticker:
            return str(company["cik_str"])
    return None


def download_filings(ticker, start_year):
    """Download the 10-K filings of one company, split them into sections, and return them."""
    with LOCK:
        setup_crawler()

        # Remove results from a previous run
        for folder in ("RAW_FILINGS", "EXTRACTED_FILINGS", "FILINGS_METADATA.csv"):
            path = os.path.join(CRAWLER_DIR, "datasets", folder)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)

        # Important: edgar-crawler downloads EVERY company's 10-K when it can't match the company,
        # so check the ticker first and give it the company's CIK number
        cik = find_cik(ticker)
        if cik is None:
            raise UnknownTicker(ticker)

        remove_stale_indices()
        config = {
            "download_filings": {
                "start_year": start_year,
                "end_year": datetime.now().year,
                "quarters": [1, 2, 3, 4],
                "filing_types": ["10-K"],
                "cik_tickers": [cik],
                "user_agent": USER_AGENT,
                "raw_filings_folder": "RAW_FILINGS",
                "indices_folder": "INDICES",
                "filings_metadata_file": "FILINGS_METADATA.csv",
                "skip_present_indices": True,
            },
            "extract_items": {
                "raw_filings_folder": "RAW_FILINGS",
                "extracted_filings_folder": "EXTRACTED_FILINGS",
                "filings_metadata_file": "FILINGS_METADATA.csv",
                "filing_types": ["10-K"],
                "include_signature": False,
                "items_to_extract": [],
                "remove_tables": True,
                "skip_extracted_filings": True,
            },
        }
        with open(os.path.join(CRAWLER_DIR, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        subprocess.run([sys.executable, "download_filings.py"], cwd=CRAWLER_DIR, check=True)
        subprocess.run([sys.executable, "extract_items.py"], cwd=CRAWLER_DIR, check=True)
        return load_filings()


def load_filings(folder=EXTRACTED_DIR):
    """Read every extracted filing (one JSON file per year), oldest first."""
    filings = []
    if not os.path.exists(folder):
        return filings
    for file_name in sorted(os.listdir(folder)):
        if file_name.endswith(".json"):
            # File names look like: <CIK>_10K_<YEAR>_<ID>.json
            with open(os.path.join(folder, file_name), encoding="utf-8", errors="ignore") as f:
                filing = json.load(f)
            filing["year"] = file_name.split("_")[2]
            filings.append(filing)
    return filings
