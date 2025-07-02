# 🎯 What is it?
# We'll extract the most frequent words from reviews where the star rating and the text sentiment disagree (sentiment_match == False).
# 🧠 Why is This Insightful?
# Language reveals emotions better than numbers.
# We can detect confusion, sarcasm, or subtle negativity in mismatched reviews.
# Repeated terms like "but," "however," or "disappointed" in mismatched 5★ reviews can expose false positivity
# 💼 Business Value
# Flipkart can use these terms to flag suspicious reviews.
# Brands get a clearer picture of what bothers customers despite high ratings.
# Great for automated sentiment correction, moderation, or quality filtering.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

try:
    # Load the dataset
    df_full = pd.read_csv('flipkart_reviews_with_sentiment.csv')
    print("Data loaded successfully.")

    # Check required columns
    required_columns = {'Rate', 'sentiment_code', 'text'}
    if not required_columns.issubset(df_full.columns):
        raise Exception(f"The DataFrame must contain the following columns: {required_columns}")
    
    # Define function to map ratings to sentiment categories
    def map_rating_to_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'

    # Map sentiment codes to labels for text sentiment
    sentiment_map = {1: 'positive', 0: 'neutral', -1: 'negative'}
    df_full['label_sentiment'] = df_full['sentiment_code'].map(sentiment_map)
    
    # Calculate rating sentiment
    df_full['rating_sentiment'] = df_full['Rate'].apply(map_rating_to_sentiment)
    
    # Calculate sentiment match
    df_full['sentiment_match'] = df_full['rating_sentiment'] == df_full['label_sentiment']

# Step 1: Filter mismatched reviews
    mismatched_texts = df_full[~df_full['sentiment_match']]['text'].dropna().astype(str)
    print(f"\nAnalyzing {len(mismatched_texts)} mismatched reviews.")

# Step 2: Tokenize and clean text
words = []
for review in mismatched_texts:
        # Remove punctuation, digits, and special characters, lowercase everything
    review = re.sub(r'[^a-zA-Z\s]', '', review).lower()
    words.extend(review.split())

    # Enhanced stopwords list
stopwords = set([
    'the', 'this', 'and', 'was', 'with', 'for', 'not', 'that', 'have', 'you',
        'but', 'its', 'are', 'very', 'too', 'had', 'been', 'from', 'they', 'all', 'my',
        'can', 'will', 'just', 'any', 'has', 'more', 'now', 'than', 'then', 'who',
        'what', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'get', 'got', 'getting', 'use', 'using'
])

    # Filter words and get counts
filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    word_counts = Counter(filtered_words)

    # Get top 15 words
    top_words = word_counts.most_common(15)
words, counts = zip(*top_words)

    print("\nTop 15 most frequent words in mismatched reviews:")
    for word, count in zip(words, counts):
        print(f"{word}: {count}")

    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(words)), counts, color=sns.color_palette('viridis', len(words)))
    
    # Customize the plot
    plt.title('Most Common Words in Reviews with Rating-Sentiment Mismatch', fontsize=14, pad=20)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Rotate x-labels for better readability
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_8_Top_Words_in_Mismatched_Reviews.png', 
                bbox_inches='tight', dpi=300)
    print("\nPlot saved as 'Figure_8_Top_Words_in_Mismatched_Reviews.png'")
    
plt.show()

except FileNotFoundError:
    print("Error: The data file could not be found. Please check if 'flipkart_reviews_with_sentiment.csv' exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

