# 🎯 What is it?
# We compare the number of reviews by:
# rating_sentiment → what the user clicked (based on stars)
# label_sentiment → what the text really says (based on NLP)
# 🧠 Why is This Insightful?
# This checks whether people rate emotionally or rationally.
# You'll often find mismatches like:
# People giving 4★ but writing complaints.
# Or giving 2★ but praising the product (maybe due to delivery delay, etc.)
# 💼 Business Value
# Helps brands detect false positivity (good ratings with bad experience).
# Improves trust scoring for reviews.
# Suggests whether customers are sugarcoating or overreacting with ratings.

import pandas as pd
import matplotlib.pyplot as plt

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

    # Print value counts to verify
    print("\nRating Sentiment Distribution:")
    print(df_full['rating_sentiment'].value_counts())
    print("\nText Sentiment Distribution:")
    print(df_full['label_sentiment'].value_counts())
    
    # Step 1: Get value counts with percentages
    label_counts = df_full['label_sentiment'].value_counts(normalize=True).reindex(['positive', 'neutral', 'negative'])
    rating_counts = df_full['rating_sentiment'].value_counts(normalize=True).reindex(['positive', 'neutral', 'negative'])

    # Step 2: Combine into a single DataFrame
    stacked_df = pd.DataFrame({
        'Text Sentiment': label_counts * 100,  # Convert to percentages
        'Rating Sentiment': rating_counts * 100
    }).T  # Transpose for stacking
    
    # Step 3: Plot stacked percentage bar chart
    ax = stacked_df.plot(kind='bar', stacked=True, figsize=(10, 6), 
                        color=['#66c2a5', '#fc8d62', '#8da0cb'])
    
    # Add percentage labels on the bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')

    # Add labels and customize
    plt.title('Proportion of Sentiment: Text Analysis vs Rating', pad=20)
    plt.xlabel('Sentiment Source')
    plt.ylabel('Proportion of Reviews (%)')
    plt.legend(title='Sentiment Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_5_Sentiment_Comparison.png', bbox_inches='tight', dpi=300)
    print("\nPlot saved as 'Figure_5_Sentiment_Comparison.png'")
    
plt.show()

except FileNotFoundError:
    print("Error: The data file could not be found. Please check if 'flipkart_reviews_with_sentiment.csv' exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
