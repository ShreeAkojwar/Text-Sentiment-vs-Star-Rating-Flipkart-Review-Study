# 🎯 What is it?
# We visualize how many reviews of each sentiment type (positive, neutral, negative) exist for each star rating (1–5).
# 🧠 Why is This Insightful?
# Not every 5-star review is textually positive, and not every 1-star is harsh.
# This chart reveals which star ratings are most ambiguous in terms of actual emotion.
# If 3★ is heavily mixed, it suggests emotional confusion or forced neutrality.
# 💼 Business Value
# Flipkart can fine-tune review sorting algorithms (e.g., highlight "textually negative" 5★).
# Brands get clarity on where the perception gap lies — are 4★ reviews actually critical?
# Encourages better rating calibration on the platform.

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
    
    # Map sentiment codes to labels for text sentiment
    sentiment_map = {1: 'positive', 0: 'neutral', -1: 'negative'}
    df_full['label_sentiment'] = df_full['sentiment_code'].map(sentiment_map)

# Step 1: Group data by Rate and label_sentiment
rating_sentiment_group = df_full.groupby(['Rate', 'label_sentiment']).size().reset_index(name='count')
    
    print("\nDistribution of sentiments across ratings:")
    print(rating_sentiment_group.pivot(index='Rate', columns='label_sentiment', values='count').fillna(0))

# Step 2: Plot
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=rating_sentiment_group, 
                     x='Rate', 
                     y='count', 
                     hue='label_sentiment',
                     hue_order=['positive', 'neutral', 'negative'], 
                     palette='Set2')

    # Add value labels on the bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
            ax.annotate(f'{int(height):,}', 
                    (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        fontsize=8)

    # Customize the plot
    plt.title('Distribution of Review Sentiments Across Star Ratings', fontsize=14, pad=20)
    plt.xlabel('Star Rating', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.legend(title='Text Sentiment', bbox_to_anchor=(1.05, 1))
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_10_Sentiment_Distribution_by_Rating.png', 
                bbox_inches='tight', dpi=300)
    print("\nPlot saved as 'Figure_10_Sentiment_Distribution_by_Rating.png'")
    
    # Print some insights
    total_reviews = rating_sentiment_group['count'].sum()
    print(f"\nKey Insights:")
    print(f"Total number of reviews analyzed: {total_reviews:,}")
    
    # Calculate percentage of each sentiment type for 5-star reviews
    five_star = rating_sentiment_group[rating_sentiment_group['Rate'] == 5]
    five_star_total = five_star['count'].sum()
    if five_star_total > 0:
        print("\n5-star reviews breakdown:")
        for sentiment in ['positive', 'neutral', 'negative']:
            count = five_star[five_star['label_sentiment'] == sentiment]['count'].values
            if len(count) > 0:
                percentage = (count[0] / five_star_total) * 100
                print(f"- {sentiment.capitalize()}: {percentage:.1f}%")
    
plt.show()

except FileNotFoundError:
    print("Error: The data file could not be found. Please check if 'flipkart_reviews_with_sentiment.csv' exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
