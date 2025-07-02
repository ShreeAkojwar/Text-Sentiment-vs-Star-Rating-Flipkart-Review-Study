# 🎯 What is it?
# We explore how often short, medium, and long reviews express positive, neutral, or negative sentiments based on their text content (label_sentiment).
# You've seen review length and sentiment separately.
# Now we're analyzing how sentiment changes within each type of review length.
# 🧠 Why is This Insightful?
# Shows how emotional complexity increases with review length.
# Short reviews: Often positive ("nice", "good") or vague.
# Medium/Long reviews: Might carry more complaints, sarcasm, or detailed emotion.
# 💼 Business Value
# Sellers can prioritize long negative reviews as they usually contain actionable feedback.
# Short reviews may inflate positivity, which affects reputation unfairly.
# Encourages platforms to prompt for detailed reviews where appropriate.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    # Load the data
    df_full = pd.read_csv('flipkart_reviews_with_sentiment.csv')
    
    # Map numeric sentiment labels to text
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df_full['sentiment'] = df_full['labels'].map(sentiment_map)
    
    # We already have review_length and review_type columns, so we can skip those calculations
    
    # Print statistics about the categorization
    print("\nReview Length Categories Distribution:")
    print(df_full['review_type'].value_counts().to_string())
    
    # Set the style
sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))

# Group and count sentiment within each review type
    sentiment_by_length = df_full.groupby(['review_type', 'sentiment']).size().reset_index(name='count')
    
    # Calculate percentages within each review type
    total_by_type = sentiment_by_length.groupby('review_type')['count'].sum().reset_index()
    sentiment_by_length = sentiment_by_length.merge(total_by_type, on='review_type', suffixes=('', '_total'))
    sentiment_by_length['percentage'] = (sentiment_by_length['count'] / sentiment_by_length['count_total'] * 100).round(6)

# Plot
    ax = sns.barplot(data=sentiment_by_length, 
                     x='review_type', 
                     y='count', 
                     hue='sentiment',
                     hue_order=['positive', 'neutral', 'negative'], 
                     order=['short', 'medium', 'long'],
                 palette='Set2')

    # Add value annotations with both count and percentage
for p in ax.patches:
    height = p.get_height()
    if height > 0:
            review_type = p.get_x() + p.get_width() / 2
            sentiment_idx = int(p.get_x() / p.get_width())
            review_type_name = ['short', 'medium', 'long'][int(review_type)]
            sentiment_name = ['positive', 'neutral', 'negative'][sentiment_idx % 3]
            percentage = sentiment_by_length[
                (sentiment_by_length['review_type'] == review_type_name) & 
                (sentiment_by_length['sentiment'] == sentiment_name)
            ]['percentage'].values[0]
            
            ax.annotate(f'{int(height):,}\n({percentage:.6f}%)', 
                    (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=8)

# Customize
    plt.title('Sentiment Distribution Across Review Lengths', fontsize=14, pad=20)
    plt.xlabel('Review Type', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.legend(title='Sentiment', title_fontsize=10, fontsize=9)
    
    # Adjust layout
plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_12_Sentiment_Distribution_by_Review_Length.png', 
                bbox_inches='tight', dpi=300)
    print("\nPlot saved as 'Figure_12_Sentiment_Distribution_by_Review_Length.png'")
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("\nSentiment distribution within each review type (percentages):")
    pivot_table = sentiment_by_length.pivot(
        index='review_type', 
        columns='sentiment', 
        values='percentage'
    ).round(6)
    print(pivot_table.to_string())

except FileNotFoundError:
    print("Error: Could not find the data file. Please ensure 'flipkart_reviews_with_sentiment.csv' exists in the root directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
