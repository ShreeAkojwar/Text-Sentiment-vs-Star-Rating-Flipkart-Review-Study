# 🎯 What is it?
# We analyze how often the rating sentiment and text sentiment align for each numeric star rating (1 to 5).
# 🧠 Why is This Insightful?
# Not all star ratings carry the same trustworthiness.
# A 3-star rating might be a mixed bag: some are neutral, some are subtly angry or overly nice.
# A 5-star rating might look positive but hide long complaints in text.
# By analyzing match % by star rating, you can detect where users are most emotionally honest
# 💼 Business Value
# Flipkart can treat certain ratings (like 3★) with caution — and maybe flag them for deeper review.
# Sellers can prioritize mismatched 5★ reviews (they may carry hidden dissatisfaction).
# Useful in developing an "emotional trust score" for reviews.

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

# Step 1: Group by Rate and calculate match percentage
    match_rate_by_rating = df_full.groupby('Rate')['sentiment_match'].agg(['mean', 'count']).reset_index()
    match_rate_by_rating['match_percentage'] = match_rate_by_rating['mean'] * 100
    
    print("\nSentiment match rates by star rating:")
    print(match_rate_by_rating[['Rate', 'match_percentage', 'count']].round(6))

# Step 2: Plot
    plt.figure(figsize=(12, 7))
    
    # Create bar plot
    x = range(len(match_rate_by_rating))
    bars = plt.bar(x, match_rate_by_rating['match_percentage'],
                   color=sns.color_palette('Blues', len(match_rate_by_rating)))

    # Add value labels on the bars with 6 decimal points
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = match_rate_by_rating.iloc[i]['count']
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.6f}%\n(n={int(count):,})',
                ha='center', va='bottom',
                fontsize=8)  # Slightly reduced font size to accommodate longer numbers

    # Customize the plot
    plt.title('Sentiment Match Rate by Star Rating', fontsize=14, pad=20)
    plt.xlabel('Star Rating', fontsize=12)
    plt.ylabel('Match Rate (%)', fontsize=12)
    
    # Set x-ticks to show star ratings
    plt.xticks(x, [f"{rate:.0f}★" for rate in match_rate_by_rating['Rate']])
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits with padding
    plt.ylim(0, max(match_rate_by_rating['match_percentage']) * 1.2)  # Add 20% padding
    
    # Adjust layout to prevent label cutoff
plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_11_Sentiment_Match_Rate_by_Rating.png', 
                bbox_inches='tight', dpi=300)
    print("\nPlot saved as 'Figure_11_Sentiment_Match_Rate_by_Rating.png'")
    
    # Print insights with 6 decimal points
    print("\nKey Insights:")
    max_idx = match_rate_by_rating['match_percentage'].idxmax()
    min_idx = match_rate_by_rating['match_percentage'].idxmin()
    
    print(f"Most emotionally consistent rating: {match_rate_by_rating.iloc[max_idx]['Rate']:.0f}★ "
          f"(Match rate: {match_rate_by_rating.iloc[max_idx]['match_percentage']:.6f}%, "
          f"Count: {int(match_rate_by_rating.iloc[max_idx]['count']):,})")
    print(f"Least emotionally consistent rating: {match_rate_by_rating.iloc[min_idx]['Rate']:.0f}★ "
          f"(Match rate: {match_rate_by_rating.iloc[min_idx]['match_percentage']:.6f}%, "
          f"Count: {int(match_rate_by_rating.iloc[min_idx]['count']):,})")
    
plt.show()

except FileNotFoundError:
    print("Error: The data file could not be found. Please check if 'flipkart_reviews_with_sentiment.csv' exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
