# AI-Based Behavioral Finance Analyzer

A minor project for **AI3270** at Manipal University Jaipur (2024–25).

Combines **NLP-based sentiment analysis** with **LSTM stock price forecasting** to study how investor psychology — fear, greed, herd mentality — correlates with real market movements.

---

## What it does

- Collects financial text from **CNBC headlines**, **StockTwits**, and **Twitter**
- Labels sentiment using **VADER**, then fine-tunes a **DistilBERT** classifier on it
- Trains an **LSTM model** on historical stock prices (Yahoo Finance) to forecast adjusted close prices
- Merges both on a time axis to run **behavioral correlation analysis**
- Generates visualizations: sentiment over time, fear/greed distribution, stock price vs sentiment heatmaps, daily returns vs sentiment

---

## Project Structure

```
├── Behavioral_finance_analyser.ipynb   # Main notebook (end-to-end pipeline)
├── AI_Based_Behavioral_Finance_Analyzer.pptx  # Project presentation
├── report_final_minor.pdf              # Full project report
├── README.md
└── .gitignore
```

---

## Pipeline Overview

```
Raw Text Data (CNBC, StockTwits, Twitter)
        ↓
  Text Cleaning & VADER Labeling
        ↓
  DistilBERT Fine-tuning (binary sentiment classifier)
        ↓
  Daily Sentiment Aggregation
        ↓
Stock Price Data (Yahoo Finance)  →  LSTM Forecasting
        ↓
  Time-aligned Merge
        ↓
  Behavioral Correlation & Visualizations
```

---

## Models

### DistilBERT Sentiment Classifier
- Base: `distilbert-base-uncased` from HuggingFace
- Fine-tuned on merged CNBC + StockTwits + Twitter data (~30K samples)
- Labels generated via VADER compound score (positive/negative)
- Test accuracy: **~97% training accuracy**, **76% on held-out test set**

### LSTM Stock Price Predictor
- 2-layer LSTM with Dropout (0.2), trained on 60-day sliding windows
- Input: normalized `Adj Close` prices (MinMaxScaler)
- MSE on test set: **< 0.002**
- Forecasts up to 3 days ahead

---

## Results

| Metric | Value |
|---|---|
| DistilBERT Test Accuracy | 76% |
| DistilBERT F1-Score | 0.76 (macro avg) |
| LSTM MSE | < 0.002 |
| Sentiment–Price Correlation | −0.05 (avg daily sentiment vs Adj Close) |

Key finding: spikes in negative sentiment consistently preceded stock dips; greed periods aligned with inflated prices later corrected by the market.

---

## Datasets

Not included in this repo due to size/licensing. Download from:

| Dataset | Source |
|---|---|
| CNBC Headlines | [Kaggle — CNBC Headlines](https://www.kaggle.com/) |
| StockTwits (AAPL) | [Kaggle — StockTwits AAPL](https://www.kaggle.com/) |
| Twitter Financial Sentiment | [Kaggle — Financial PhraseBank / train_data](https://www.kaggle.com/) |
| Stock Prices | `yfinance` — fetched via Yahoo Finance API |

Update the file paths in the notebook cells (currently hardcoded to local paths like `C:\Users\Aashvi\Downloads\...`).

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install tensorflow transformers vaderSentiment
pip install yfinance
```

Run the notebook top to bottom. Sections are:
1. Data loading & preprocessing (Cells 1–21)
2. DistilBERT training & evaluation (Cells 22–28)
3. LSTM training & evaluation (Cells 29–38)
4. Behavioral correlation analysis & visualizations (Cells 39–64)


--
