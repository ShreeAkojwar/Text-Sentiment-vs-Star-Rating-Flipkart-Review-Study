# Flipkart Review Analysis: Text Sentiment vs Star Rating Study

This project analyzes the relationship between aspect-based sentiment analysis and star ratings in Flipkart product reviews, investigating whether textual sentiment can predict numerical ratings.

## Project Overview

The analysis focuses on four key aspects of reviews:
- Quality
- Cost
- Delivery
- Flexibility

We use advanced machine learning techniques to predict star ratings from sentiment analysis results.

### Key Features
- Multiple ML models (Random Forest, Gradient Boosting, SVR, ElasticNet)
- SMOTE for handling imbalanced data
- Polynomial feature engineering
- Weighted ensemble approach
- Comprehensive error analysis

## Project Structure
```
.
├── README.md
├── requirements.txt
├── aspect_sentiment_analysis.py
├── regression_analysis_predict_rating_from_aspect_sentiments.py
├── processed_data/
│   └── aspect_sentiment_vader.csv
└── outputs/
    ├── prediction_analysis.png
    ├── error_distribution.png
    ├── feature_importance.png
    ├── regression_results.csv
    └── model_summary.txt
```

## Requirements
```
pandas==2.3.0
numpy==2.3.0
matplotlib==3.10.3
seaborn==0.13.2
scikit-learn==1.6.1
imbalanced-learn==0.13.0
vaderSentiment==3.3.2
```

## Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/ShreeAkojwar/Text-Sentiment-vs-Star-Rating-Flipkart-Review-Study.git
cd Text-Sentiment-vs-Star-Rating-Flipkart-Review-Study
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
# First, run aspect sentiment analysis
python aspect_sentiment_analysis.py

# Then, run the regression analysis
python regression_analysis_predict_rating_from_aspect_sentiments.py
```

## Results

The analysis reveals several interesting findings:

1. Best Model Performance (SVR):
   - RMSE: 1.6279
   - MAE: 1.4789

2. Ensemble Model Performance:
   - RMSE: 1.6704
   - MAE: 1.5293

Key Finding: The analysis suggests that aspect-based sentiments alone cannot reliably predict star ratings, indicating that users' rating behaviors are influenced by factors beyond the analyzed aspects.

## Output Files

- `prediction_analysis.png`: Scatter plot of predicted vs actual ratings
- `error_distribution.png`: Distribution of prediction errors
- `feature_importance.png`: Top 10 most important features
- `regression_results.csv`: Detailed predictions and errors
- `model_summary.txt`: Complete performance metrics

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License

