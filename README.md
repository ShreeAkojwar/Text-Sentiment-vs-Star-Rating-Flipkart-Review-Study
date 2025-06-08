# Flipkart Reviews Sentiment Analysis

This project analyzes sentiment in Flipkart product reviews using machine learning and natural language processing techniques.

## Project Structure

- `convert_parquet_to_csv.py`: Script to convert Parquet files to CSV format
- `sentiment_analysis.py`: Main script for sentiment analysis using TextBlob and Random Forest
- `train.parquet` & `test.parquet`: Original review data
- `flipkart_reviews_full.csv`: Combined dataset from Parquet files
- `flipkart_reviews_with_sentiment.csv`: Final dataset with sentiment analysis

## Features

- Text preprocessing and cleaning
- Sentiment analysis using TextBlob
- Machine learning model (Random Forest) for sentiment prediction
- Feature engineering from review text
- Hybrid approach combining rating-based and text-based sentiment

## Results

The sentiment analysis produces:
- Review length analysis (short/medium/long)
- Sentiment scores (negative/neutral/positive)
- Feature importance analysis
- Model performance metrics

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