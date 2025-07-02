# 🎯 What is it?
# This visualization shows how sentiment mismatches are distributed across different review types (short, medium, long).
# A mismatch occurs when the sentiment from the text doesn't align with the rating sentiment.
# 🧠 Why is This Insightful?
# Helps identify which review lengths are more prone to sentiment-rating mismatches.
# May reveal patterns in how review length relates to sentiment consistency.
# 💼 Business Value
# Guides review collection strategy (e.g., encouraging longer/shorter reviews).
# Helps in prioritizing which reviews to analyze for customer satisfaction issues.

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
    
    # Calculate rating sentiment
    def get_rating_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    df_full['rating_sentiment'] = df_full['Rate'].apply(get_rating_sentiment)
    
    # Calculate sentiment match
    df_full['sentiment_match'] = df_full['sentiment'] == df_full['rating_sentiment']

# Filter mismatched reviews
    mismatched_df = df_full[~df_full['sentiment_match']]

# Count mismatches by review type
    mismatch_by_type = mismatched_df['review_type'].value_counts().reindex(['short', 'medium', 'long'])
    mismatch_percentages = (mismatch_by_type / mismatch_by_type.sum() * 100).round(6)
    
    # Print statistics
    print("\nTotal number of mismatched reviews:", len(mismatched_df))
    print("\nMismatch distribution by review type:")
    for review_type, count in mismatch_by_type.items():
        percentage = mismatch_percentages[review_type]
        print(f"{review_type}: {count:,} reviews ({percentage:.6f}%)")

# Plot as pie chart
    plt.figure(figsize=(10, 8))
colors = sns.color_palette('pastel')
    
    # Create pie chart with exact percentages (6 decimal places)
    patches, texts, autotexts = plt.pie(mismatch_percentages, 
                                      labels=mismatch_percentages.index, 
                                      autopct=lambda pct: f'{pct:.6f}%',
                                      colors=colors, 
                                      startangle=140)
    
    # Adjust font sizes
    plt.setp(autotexts, size=8)
    plt.setp(texts, size=10)
    
    plt.title('Proportion of Total Sentiment Mismatches by Review Type', 
              fontsize=14, 
              pad=20)
    
    # Add a legend with counts
    legend_labels = [f'{type_}: {count:,} reviews' 
                    for type_, count in mismatch_by_type.items()]
    plt.legend(patches, 
              legend_labels, 
              title='Review Counts',
              loc='center left', 
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Adjust layout to prevent label cutoff
plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_13_Sentiment_Mismatch_Proportion.png', 
                bbox_inches='tight', 
                dpi=300)
    print("\nPlot saved as 'Figure_13_Sentiment_Mismatch_Proportion.png'")

except FileNotFoundError:
    print("Error: Could not find the data file. Please ensure 'flipkart_reviews_with_sentiment.csv' exists in the root directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
