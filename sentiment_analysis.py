import pandas as pd
import numpy as np
from textblob import TextBlob #Library for processing textual data: sentiment analysis, text processing
import re # re: Python's built-in regular expression library- regular expressions, pattern matching
from sklearn.model_selection import train_test_split #splitting data into training and testing sets
from sklearn.ensemble import RandomForestClassifier #ensemble learning method
from sklearn.metrics import classification_report #evaluation metrics for classification

def preprocess_text(text):
    """ 1. Convert to lowercase: standardize text 2. Remove special characters: clean noise 3. Handle whitespace: standardize spacing
    Key concepts to learn: - String manipulation, Regular expressions, Text normalization  """ 
# Convert to string if not already (handles non-string inputs)
    text = str(text)
# Convert to lowercase to standardize text
    text = text.lower()
# Remove special characters and digits using regex
# [^a-zA-Z\s] means "match anything that's not a letter or whitespace"
    text = re.sub(r'[^a-zA-Z\s]', '', text)
# Remove extra whitespace and standardize spacing
    text = ' '.join(text.split())
    return text

# SENTIMENT ANALYSIS
def get_textblob_sentiment(text):
    """
    Get sentiment scores using TextBlob.
Sentiment analysis
Polarity scoring: how positive/negative (-1 to 1)
Text classification: how subjective/objective (0 to 1)
    """
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    
# Convert polarity to sentiment categories
    if polarity < -0.1:
        return 0  # negative
    elif polarity > 0.1:
        return 2  # positive
    else:
        return 1  # neutral

def create_sentiment_features(text):
    """
    Create features for sentiment analysis.
    Feature engineering is crucial in ML:
    Extract meaningful characteristics from text
    Create numerical representations
    """
    analysis = TextBlob(str(text))
    
    return {
        'polarity': analysis.sentiment.polarity,      # How positive/negative
        'subjectivity': analysis.sentiment.subjectivity,  # How subjective/objective
        'word_count': len(str(text).split()),         # Length of text
        'has_exclamation': '!' in str(text),          # Presence of excitement
        'has_question': '?' in str(text),             # Presence of questions
        'capital_words': sum(1 for word in str(text).split() if word.isupper()),  # Emphasis through caps
    }

# MAIN PROCESSING
# Load the data
print("Loading data...")
df = pd.read_csv("flipkart_reviews_full.csv")

# Preprocess text
print("Preprocessing text...")
df['processed_text'] = df['text'].apply(preprocess_text)

# Get initial sentiment using TextBlob
print("Calculating TextBlob sentiment...")
df['textblob_sentiment'] = df['processed_text'].apply(get_textblob_sentiment)

# Create features for ML model
print("Creating features...")
features = df['processed_text'].apply(create_sentiment_features).apply(pd.Series)
# Converts the dictionary output from the previous .apply() into a DataFrame.
# Each dictionary becomes a row, and the keys become column names.

# MACHINE LEARNING
# Create target variable based on rating
# Map ratings to sentiment categories
df['rating_sentiment'] = df['Rate'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))

# Prepare data for ML
# Combine numerical features with categorical features (one-hot encoded)
X = pd.concat([
    features,
    pd.get_dummies(df['review_type'])  # Convert categorical to numerical
], axis=1)

y = df['rating_sentiment']

# Split data into training and testing sets
# Key concepts: train-test split, cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
print("Training Random Forest model...")
# Random Forest: ensemble learning method using multiple decision trees
# Key concepts: ensemble learning, decision trees, random forests
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = rf_model.predict(X_test)

# MODEL EVALUATION
# Print model performance metrics
print("\nModel Performance:")
# Classification report shows precision, recall, f1-score
# Key concepts: precision, recall, F1 score, classification metrics
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))


# FINAL SENTIMENT GENERATION
# Generate final sentiment labels
print("Generating final sentiment labels...")
df['sentiment_code'] = rf_model.predict(X)

# Combine rating and textblob sentiment for final labels
# This creates a hybrid approach using both methods
df['labels'] = df.apply(lambda row: 
    2 if (row['Rate'] >= 4 or row['textblob_sentiment'] == 2) else
    (0 if row['Rate'] <= 2 or row['textblob_sentiment'] == 0 else 1), 
    axis=1
)

# SAVE AND SUMMARIZE RESULTS
# Save the updated dataset
print("Saving updated dataset...")
output_columns = [
    'product_name', 'product_price', 'Rate', 'Review', 'text',
    'review_length', 'review_type', 'rating_sentiment',
    'sentiment_code', 'labels'
]
df[output_columns].to_csv("flipkart_reviews_with_sentiment.csv", index=False)

# Print sentiment distribution statistics
print("\nSentiment Distribution:")
sentiment_counts = df['labels'].value_counts()
for sentiment, count in sentiment_counts.items():
    sentiment_name = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}[sentiment]
    percentage = (count / len(df)) * 100
    print(f"  - {sentiment_name}: {count:,} reviews ({percentage:.1f}%)")

# Print feature importance analysis
print("\nTop Features for Sentiment Prediction:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
for _, row in feature_importance.head(10).iterrows():
    print(f"  - {row['feature']}: {row['importance']:.3f}")

print("\nDone! Updated dataset saved as 'flipkart_reviews_with_sentiment.csv'") 