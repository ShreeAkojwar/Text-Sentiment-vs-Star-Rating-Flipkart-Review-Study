# 🎯 What is it?
# You already calculated whether each review's rating sentiment matches the label sentiment (sentiment_match).
# Now, we go a level deeper: among mismatched cases, what specific transitions are happening?
# E.g.:
# People give a positive rating but their review text is negative
# Or they give neutral ratings but write like they're very happy
# 🧠 Why is This Insightful?
# You're not just checking if mismatches exist — you're analyzing how and where they occur.
# This shows emotional contradictions in user behavior or even platform pressure
# 💼 Business Value
# Flipkart can flag products where customers rate high under pressure but complain in text.
# Sellers can identify when buyers are mildly frustrated but not expressing it through ratings.
# Helps design better survey and feedback UX — users shouldn't have to "fake" their feelings.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Filter only mismatched reviews
    mismatch_df = df_full[~df_full['sentiment_match']]
    print(f"\nFound {len(mismatch_df)} mismatched reviews.")

# Count combinations of rating_sentiment → label_sentiment
pattern_counts = mismatch_df.groupby(['rating_sentiment', 'label_sentiment']).size().reset_index(name='count')
    print("\nMismatch patterns:")
    print(pattern_counts)

# Plot grouped bar chart
    plt.figure(figsize=(12, 7))
ax = sns.barplot(data=pattern_counts, x='rating_sentiment', y='count', hue='label_sentiment',
                 order=['positive', 'neutral', 'negative'],
                 hue_order=['positive', 'neutral', 'negative'],
                 palette='Set2')

# Annotate bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
            ax.annotate(f'{int(height):,}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=10)

# Labels and legend
    plt.title('Sentiment Mismatch Patterns: Rating vs. Text Analysis', fontsize=14, pad=20)
    plt.xlabel('Rating-Based Sentiment')
plt.ylabel('Number of Mismatched Reviews')
    plt.legend(title='Text-Based Sentiment', bbox_to_anchor=(1.05, 1))
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_6_Sentiment_Mismatch_Patterns.png', bbox_inches='tight', dpi=300)
    print("\nPlot saved as 'Figure_6_Sentiment_Mismatch_Patterns.png'")
    
plt.show()

except FileNotFoundError:
    print("Error: The data file could not be found. Please check if 'flipkart_reviews_with_sentiment.csv' exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
