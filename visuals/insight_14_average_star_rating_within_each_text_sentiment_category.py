# 🎯 What is it?
# We compute the average numeric star rating (Rate) for each category of text-based sentiment (label_sentiment).
# 🧠 Why is This Insightful?
# Helps detect rating inflation or deflation based on how people feel.
# For example:
# If the average rating for positive text reviews is 4.8, that's consistent.
# But if the average rating for negative reviews is 3.5, that's a serious sentiment mismatch trend
# 💼 Business Value
# Useful for brands to understand bias in user ratings.
# Shows whether people overrate or underrate relative to how they write.
# Can feed into building a sentiment-adjusted rating score, a smarter metric for product trust.

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
    
    # Compute statistics for each sentiment category
    sentiment_stats = df_full.groupby('sentiment').agg({
        'Rate': ['count', 'mean', 'std', 'min', 'max']
    }).round(6)
    
    # Flatten column names
    sentiment_stats.columns = ['count', 'mean', 'std', 'min', 'max']
    
    # Print detailed statistics
    print("\nDetailed Statistics by Sentiment Category:")
    print("\nNumber of reviews in each category:")
    for sentiment in ['positive', 'neutral', 'negative']:
        count = sentiment_stats.loc[sentiment, 'count']
        mean = sentiment_stats.loc[sentiment, 'mean']
        std = sentiment_stats.loc[sentiment, 'std']
        min_val = sentiment_stats.loc[sentiment, 'min']
        max_val = sentiment_stats.loc[sentiment, 'max']
        print(f"\n{sentiment.capitalize()} Sentiment:")
        print(f"Count: {int(count):,} reviews")
        print(f"Average Rating: {mean:.6f} ± {std:.6f}")
        print(f"Rating Range: {min_val:.1f} to {max_val:.1f} stars")
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Create bar plot
    ax = sns.barplot(data=df_full, 
                    x='sentiment', 
                    y='Rate',
                    order=['positive', 'neutral', 'negative'],
                    palette='Set2',
                    ci='sd')  # Show standard deviation instead of confidence interval
    
    # Add value labels with mean and std dev
    for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
        mean = sentiment_stats.loc[sentiment, 'mean']
        std = sentiment_stats.loc[sentiment, 'std']
        count = sentiment_stats.loc[sentiment, 'count']
        
        ax.text(i, mean + 0.1,
                f'Mean: {mean:.6f}\n±{std:.6f}\n(n={int(count):,})',
                ha='center', va='bottom', fontsize=9)
    
    # Customize the plot
    plt.title('Average Star Rating by Text Sentiment Category', fontsize=14, pad=20)
    plt.xlabel('Text Sentiment', fontsize=12)
    plt.ylabel('Star Rating', fontsize=12)
    
    # Set y-axis limits with some padding
    plt.ylim(0, 5.5)
    
    # Add a horizontal line at rating 3 for reference
    plt.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_14_Average_Rating_by_Sentiment.png', 
                bbox_inches='tight', 
                dpi=300)
    print("\nPlot saved as 'Figure_14_Average_Rating_by_Sentiment.png'")

except FileNotFoundError:
    print("Error: Could not find the data file. Please ensure 'flipkart_reviews_with_sentiment.csv' exists in the root directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
