import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Ensure df_full is defined (example: loading from a CSV file)
try:
    # Load the Flipkart reviews dataset
    df_full = pd.read_csv('../flipkart_reviews_with_sentiment.csv')  # Using the correct file name
    print("Data loaded successfully.")
except FileNotFoundError:
    raise Exception("The data file could not be found. Please check the file path.")

# Check if required columns exist
required_columns = {'review_type', 'sentiment_match'}
if not required_columns.issubset(df_full.columns):
    raise Exception(f"The DataFrame must contain the following columns: {required_columns}")
print("Required columns are present.")

# Ensure 'sentiment_match' is boolean
if not pd.api.types.is_bool_dtype(df_full['sentiment_match']):
    print(f"Column 'sentiment_match' dtype: {df_full['sentiment_match'].dtype}")
    raise Exception("'sentiment_match' column must contain boolean values (True/False).")
print("'sentiment_match' column is boolean.")

# Step 1: Group by review type and calculate mismatch rate
try:
    # Ensure 'review_type' contains the expected categories
    expected_categories = ['short', 'medium', 'long']
    actual_categories = df_full['review_type'].unique()
    print(f"Actual categories in 'review_type': {actual_categories}")
    missing_categories = set(expected_categories) - set(actual_categories)
    if missing_categories:
        print(f"Warning: Missing categories in 'review_type': {missing_categories}")

    # Calculate mismatch rate
    mismatch_by_type = df_full.groupby('review_type')['sentiment_match'].apply(
        lambda x: 100 * (~x).mean()
    ).reindex(expected_categories, fill_value=0)  # Fill missing categories with 0
    print("Mismatch rates calculated successfully.")
    print(mismatch_by_type)
except KeyError as e:
    raise Exception(f"Error in grouping or reindexing: {e}")

# Step 2: Plot the mismatch percentages
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=mismatch_by_type.index, y=mismatch_by_type.values, palette='pastel')

# Add value labels
for i, val in enumerate(mismatch_by_type.values):
    ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', fontsize=11)

# Titles and labels
plt.title('Sentiment Mismatch Rate by Review Type', fontsize=14)
plt.xlabel('Review Type')
plt.ylabel('Mismatch Rate (%)')
plt.ylim(0, mismatch_by_type.max() + 5)
plt.tight_layout()
plt.show()