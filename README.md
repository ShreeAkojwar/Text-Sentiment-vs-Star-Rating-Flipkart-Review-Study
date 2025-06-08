# Multi-Platform Customer Review Analysis for Flipkart

This project analyzes customer reviews from Flipkart using machine learning and natural language processing techniques to perform sentiment analysis and extract insights.

## Project Components

### Sentiment Analysis
- `sentiment_analysis.py`: Main script for sentiment analysis using TextBlob and Random Forest
- `convert_parquet_to_csv.py`: Script to convert Parquet files to CSV format

### Data Files
- `train.parquet` & `test.parquet`: Original review data
- `flipkart_reviews_full.csv`: Combined dataset from Parquet files
- `flipkart_reviews_with_sentiment.csv`: Final dataset with sentiment analysis

## Features

- Text preprocessing and cleaning
- Sentiment analysis using TextBlob
- Machine learning model (Random Forest) for sentiment prediction
- Feature engineering from review text
- Hybrid approach combining rating-based and text-based sentiment
- Review length analysis (short/medium/long)
- Sentiment distribution analysis
- Feature importance analysis

## Results

The sentiment analysis produces:
- Review length categorization
- Sentiment scores (negative/neutral/positive)
- Model performance metrics
- Feature importance rankings

## Requirements

```
pandas
numpy
textblob
scikit-learn
```

## How to Run

1. Install requirements:
```bash
pip install pandas numpy textblob scikit-learn
```

2. Convert Parquet to CSV:
```bash
python convert_parquet_to_csv.py
```

3. Run sentiment analysis:
```bash
python sentiment_analysis.py
```

## Analysis Results

The sentiment analysis shows:
- Positive reviews: 85.5%
- Negative reviews: 13.1%
- Neutral reviews: 1.4%

Model achieves 86% accuracy in sentiment prediction, with particularly strong performance in identifying positive reviews (93% F1-score).
