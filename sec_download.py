"""
Download 10-K filings from SEC EDGAR and split them into sections,
using the open-source edgar-crawler tool (https://github.com/nlpaueb/edgar-crawler).
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

CRAWLER_DIR = "edgar-crawler"
CRAWLER_URL = "https://github.com/nlpaueb/edgar-crawler.git"
EXTRACTED_DIR = os.path.join(CRAWLER_DIR, "datasets", "EXTRACTED_FILINGS", "10-K")


def setup_crawler():
    """Clone edgar-crawler (if needed) and fix its filing links."""
    script = os.path.join(CRAWLER_DIR, "download_filings.py")
    if not os.path.exists(script):
        subprocess.run(["git", "clone", CRAWLER_URL, CRAWLER_DIR], check=True)

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


def download_filings(ticker, start_year):
    """Download the 10-K filings of one company and split them into sections."""
    setup_crawler()

    # Remove results from a previous run
    for folder in ("RAW_FILINGS", "EXTRACTED_FILINGS", "FILINGS_METADATA.csv"):
        path = os.path.join(CRAWLER_DIR, "datasets", folder)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)

    config = {
        "download_filings": {
            "start_year": start_year,
            "end_year": datetime.now().year,
            "quarters": [1, 2, 3, 4],
            "filing_types": ["10-K"],
            "cik_tickers": [ticker],
            "user_agent": "Student Project (student@example.com)",  # the SEC asks for a contact
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
