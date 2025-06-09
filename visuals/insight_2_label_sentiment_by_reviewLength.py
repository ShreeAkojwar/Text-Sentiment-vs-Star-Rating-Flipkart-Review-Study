# 🎯 What is it?
# We're analyzing how different types of reviews (short, medium, long) relate to sentiment — based on what users wrote (label_sentiment).
# 🧠 Why is this Insightful? (First Principles Thinking)
# People write differently when they're angry, satisfied, or confused.
# A long review may signal strong emotion (positive or negative), while short ones are often quick reactions.
# By analyzing this, we learn how deeply customers express their emotions, and what kind of feedback is more common.
# 💼 Business Value
# Helps identify which kind of reviews (short/long) are more emotionally charged.
# Customer support teams can prioritize longer negative reviews.
#Platforms can encourage detailed feedback if it's shown to offer clearer sentiment.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set(style="whitegrid")

try:
    # Load the dataset
    df = pd.read_csv('/Users/shreeakojwar/Downloads/IIMN_Final /flipkart_reviews_with_sentiment.csv')
    
    # Create sentiment labels based on rating_sentiment
    df['sentiment'] = df['rating_sentiment'].map({
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    })
    
    # Create a grouped countplot
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='review_type', hue='sentiment',
                    order=['short', 'medium', 'long'], 
                    hue_order=['positive', 'neutral', 'negative'],
                    palette='Set2')

    # Add value labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', fontsize=10)

    # Add labels and title
    plt.title('Sentiment Distribution by Review Length', fontsize=14)
    plt.xlabel('Review Length Category')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: Could not find the CSV file at '/Users/shreeakojwar/Downloads/IIMN_Final /flipkart_reviews_with_sentiment.csv'")
    print("Please check if the file exists and the path is correct.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check if the CSV file contains the required columns: 'review_type' and 'rating_sentiment'")

