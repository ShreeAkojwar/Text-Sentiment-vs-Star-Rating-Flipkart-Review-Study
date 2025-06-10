# Flipkart Review Analysis

## Regression Analysis: Predicting Ratings from Aspect Sentiments

This project analyzes Flipkart product reviews by attempting to predict star ratings from aspect-based sentiment analysis. The analysis focuses on four key aspects: quality, cost, delivery, and flexibility.

### Key Features
- Multiple advanced ML models (Random Forest, Gradient Boosting, SVR, ElasticNet)
- SMOTE for handling imbalanced data
- Polynomial feature engineering
- Weighted ensemble approach
- Comprehensive error analysis and visualization

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

### Project Structure
```
.
├── README.md
├── aspect_sentiment_analysis.py
├── reression_analysis_predict_rating_from_aspect_sentiments.py
├── processed_data/
│   └── aspect_sentiment_vader.csv
└── outputs/
    ├── final_predictions.png
    ├── final_error_distribution.png
    ├── final_regression_results.csv
    └── final_model_summary.txt
```

### Key Findings
- Aspect sentiments alone are not strong predictors of overall ratings
- Best performing model: SVR (Support Vector Regression)
  - RMSE: 1.6279
  - MAE: 1.4789
- Weighted ensemble of all models:
  - RMSE: 1.6704
  - MAE: 1.5293

### Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run aspect sentiment analysis:
```bash
python aspect_sentiment_analysis.py
```

3. Run regression analysis:
```bash
python reression_analysis_predict_rating_from_aspect_sentiments.py
```

### Results
The analysis reveals that aspect-based sentiments alone cannot reliably predict overall ratings, suggesting that users' overall ratings are influenced by factors beyond the analyzed aspects.

