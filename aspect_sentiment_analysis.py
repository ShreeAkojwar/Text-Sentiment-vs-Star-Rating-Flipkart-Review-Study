import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('flipkart_reviews_with_sentiment.csv')

# Refined aspect keywords
aspect_keywords = {
    "quality": [
        "quality", "durability", "reliable", "reliability", "build", "sturdy", "material",
        "solid", "premium", "design", "performance", "defective", "broken", "damaged"
    ],
    "cost": [
        "price", "cheap", "expensive", "affordable", "value", "worth", "cost", "overpriced",
        "reasonable", "deal", "budget", "money", "money's worth", "rip off", "steal"
    ],
    "delivery": [
        "delivery", "shipping", "courier", "delivered", "arrival", "late", "delay",
        "on time", "fast", "slow", "packaging", "return window", "damaged during shipping"
    ],
    "flexibility": [
        "return", "replace", "exchange", "adapt", "modify", "customize", "adjust", "change",
        "cancellation", "refund", "reschedule", "policy"
    ]
}

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Updated sentiment extraction function
def extract_aspect_sentiment_vader(review):
    review = str(review).lower()
    results = {}

    for aspect, keywords in aspect_keywords.items():
        relevant_sentences = []
        for kw in keywords:
            pattern = rf'([^.]*\b{re.escape(kw)}\b[^.]*)\.?'
            relevant_sentences += re.findall(pattern, review)

        if relevant_sentences:
            compound_scores = [
                analyzer.polarity_scores(sent)["compound"]
                for sent in relevant_sentences
            ]
            avg_score = np.mean(compound_scores)

            if avg_score > 0.1:
                results[aspect] = 1
            elif avg_score < -0.1:
                results[aspect] = -1
            else:
                results[aspect] = 0
        else:
            results[aspect] = 0

    return results

# Apply the function to reviews
aspect_scores = df['Review'].apply(extract_aspect_sentiment_vader)
aspect_df = pd.DataFrame(aspect_scores.tolist())

# Merge with original data
df_final = pd.concat([df[['Review', 'Rate', 'product_name']], aspect_df], axis=1)

# Save result
df_final.to_csv('processed_data/aspect_sentiment_vader.csv', index=False)
print("Updated VADER-based aspect sentiment saved to 'processed_data/aspect_sentiment_vader.csv'")
