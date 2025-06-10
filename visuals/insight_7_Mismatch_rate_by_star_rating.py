# 🎯 What is it?
# We measure the percentage of mismatched reviews for each numeric star rating (from Rate column: 1★ to 5★).
# 🧠 Why is This Insightful?
# It reveals which ratings are most emotionally confusing.
# For example:
# 3★ is often chosen when people are confused, indifferent, or lazy.
# 5★ may be chosen out of habit or reward pressure, even if the review is critical.
# 1★ might include sarcastic positives — e.g., "Amazing phone… if you want it to die in a week."
# 💼 Business Value
# Platforms can improve rating system UX: highlight when users contradict themselves.
# Sellers can interpret 3★ reviews with caution, especially if text is overly negative.
# Helps build an "honesty index" for each rating level.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the dataset
    df_full = pd.read_csv('flipkart_reviews_with_sentiment.csv')
    print("Data loaded successfully.")

    # Check required columns
    required_columns = {'Rate', 'sentiment_code'}
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

    # Step 1: Group by numeric star rating (Rate), calculate mismatch %
    mismatch_by_rating = df_full.groupby('Rate')['sentiment_match'].apply(lambda x: 100 * (~x).mean())
    print("\nMismatch rates by star rating:")
    print(mismatch_by_rating)

    # Step 2: Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=mismatch_by_rating.index, y=mismatch_by_rating.values, palette='coolwarm')

    # Annotate bars
    for i, val in enumerate(mismatch_by_rating.values):
        ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', fontsize=11)

    # Labels and styling
    plt.title('Sentiment Mismatch Rate by Star Rating', fontsize=14, pad=20)
    plt.xlabel('Star Rating')
    plt.ylabel('Mismatch Rate (%)')
    plt.ylim(0, mismatch_by_rating.max() + 5)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_7_Mismatch_Rate_by_Star_Rating.png', bbox_inches='tight', dpi=300)
    print("\nPlot saved as 'Figure_7_Mismatch_Rate_by_Star_Rating.png'")
    
    plt.show()

except FileNotFoundError:
    print("Error: The data file could not be found. Please check if 'flipkart_reviews_with_sentiment.csv' exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

