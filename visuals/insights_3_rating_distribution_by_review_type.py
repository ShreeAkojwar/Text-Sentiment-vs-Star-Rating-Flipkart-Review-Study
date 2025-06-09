# 🎯 What is it?
# We analyze the distribution of Rate (1 to 5 stars) across different review_type categories — short, medium, and long reviews.
# 🧠 Why is This Insightful?
# Does the length of a review affect how a person rates the product?
# For example:
# Are short reviews overly generous (5 stars with "Good")?
# Are long reviews more critical or balanced?
# This helps you uncover rating behavior patterns that numbers alone won't show
# 💼 Business Value
# Helps platforms identify superficial or impulsive reviews (short + 5 stars).
# Reveals how much effort people put into rating fairly.
# Can improve product feedback weighting: long reviews might carry more honest context.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style
sns.set(style="whitegrid")

try:
    # Load the dataset
    df = pd.read_csv('flipkart_reviews_with_sentiment.csv')
    
    # Clean the Rate column: fill NA with median and ensure values are between 1-5
    df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')  # Convert to numeric, invalid values become NA
    median_rate = df['Rate'].median()  # Get median for filling NA values
    df['Rate'] = df['Rate'].fillna(median_rate)  # Fill NA with median
    df['Rate'] = df['Rate'].clip(1, 5)  # Ensure values are between 1-5
    df['Rate'] = df['Rate'].round().astype(int)  # Round and convert to integer
    
    # Plot: Count of ratings grouped by review type
    plt.figure(figsize=(12, 7))
    ax = sns.countplot(data=df, x='review_type', hue='Rate',
                      order=['short', 'medium', 'long'], 
                      hue_order=[1, 2, 3, 4, 5],
                      palette='RdYlGn')

    # Add value labels to bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10)

    # Customize labels
    plt.title('Rating Distribution by Review Length', fontsize=14, pad=20)
    plt.xlabel('Review Length Category', labelpad=10)
    plt.ylabel('Number of Reviews', labelpad=10)
    plt.legend(title='Star Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: Could not find the CSV file. Please check if the file exists in the current directory:")
    print("flipkart_reviews_with_sentiment.csv")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check if the CSV file contains the required columns: 'review_type' and 'Rate'")
