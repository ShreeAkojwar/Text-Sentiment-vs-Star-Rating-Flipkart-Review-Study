# 🔍 Insight #9: Sentiment Mismatch by Product Category
# 🎯 What is it?
# We analyze how sentiment mismatches vary across different product categories
# 🧠 Why is This Insightful?
# Different product types may trigger different emotional responses
# Some categories might be more prone to rating-sentiment mismatches
# Helps identify which product areas need more attention in review analysis
# 💼 Business Value
# Category-specific review monitoring strategies
# Identify product categories that need improved rating systems
# Help sellers understand category-specific customer behavior

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the dataset
    df_full = pd.read_csv('flipkart_reviews_with_sentiment.csv')
    print("Data loaded successfully.")

    # Check required columns
    required_columns = {'Rate', 'sentiment_code', 'product_name'}
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

    # Extract product category from product_name (taking the first word as category)
    df_full['category'] = df_full['product_name'].str.split().str[0]
    
    # Calculate mismatch rate by category
    category_stats = df_full.groupby('category').agg({
        'sentiment_match': ['count', lambda x: (~x).mean() * 100]
    }).reset_index()
    
    # Rename columns
    category_stats.columns = ['category', 'total_reviews', 'mismatch_rate']
    
    # Filter categories with at least 100 reviews for meaningful analysis
    category_stats = category_stats[category_stats['total_reviews'] >= 100].sort_values('mismatch_rate', ascending=True)
    
    print("\nMismatch rates by product category:")
    print(category_stats)

    # Create the visualization
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(category_stats)), 
                   category_stats['mismatch_rate'],
                   color=sns.color_palette('viridis', len(category_stats)))
    
    # Customize the plot
    plt.title('Sentiment Mismatch Rate by Product Category', fontsize=14, pad=20)
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Mismatch Rate (%)', fontsize=12)
    
    # Rotate x-labels for better readability with smaller font
    plt.xticks(range(len(category_stats)), 
               category_stats['category'], 
               rotation=45, 
               ha='right',
               fontsize=8)  # Decreased font size for category labels
    
    # Add value labels on top of bars with smaller font
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=8)  # Decreased font size for value labels
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff with more bottom margin
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Added more space at the bottom
    
    # Save the plot
    plt.savefig('visual_images/Figure_9_Mismatch_Rate_by_Category.png', 
                bbox_inches='tight', dpi=300)
    print("\nPlot saved as 'Figure_9_Mismatch_Rate_by_Category.png'")
    
    # Add some statistics
    print(f"\nSummary Statistics:")
    print(f"Average mismatch rate across categories: {category_stats['mismatch_rate'].mean():.1f}%")
    print(f"Category with highest mismatch rate: {category_stats.iloc[-1]['category']} ({category_stats.iloc[-1]['mismatch_rate']:.1f}%)")
    print(f"Category with lowest mismatch rate: {category_stats.iloc[0]['category']} ({category_stats.iloc[0]['mismatch_rate']:.1f}%)")
    
    plt.show()

except FileNotFoundError:
    print("Error: The data file could not be found. Please check if 'flipkart_reviews_with_sentiment.csv' exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
