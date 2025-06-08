import pandas as pd
import os

# Load the parquet files
print("Loading Parquet files...")
try:
    df_train = pd.read_parquet("train.parquet")
    df_test = pd.read_parquet("test.parquet")
except FileNotFoundError as e:
    print(f"Error: Could not find parquet files: {e}")
    raise
except Exception as e:
    print(f"Error loading parquet files: {e}")
    raise

# Combine them into a single dataset
print("Combining datasets...")
df_full = pd.concat([df_train, df_test], ignore_index=True)

# Print original column names
print("\nOriginal columns in the dataset:")
for col in df_full.columns:
    print(f"  - {col}") #The f"..." is a formatted string literal (called an "f-string").It lets you insert variables directly into strings using curly braces {}

# Calculate review length using the correct column name 'text'
# .apply(...)	Applies a function to each row in the text column
# lambda x: ...	Anonymous function (takes each review as x)
# str(x)	Ensures the input is a string (in case of None/NaN)
# .split()	Splits the text into words (by spaces)
# len(...)	Counts how many words are in the list 
df_full['review_length'] = df_full['text'].apply(lambda x: len(str(x).split()))

# Review type
def review_type(length):
    if length < 10:
        return 'short'
    elif length < 50:
        return 'medium'
    else:
        return 'long'

df_full['review_type'] = df_full['review_length'].apply(review_type)

# 'rating' is the star rating (1 to 5) given directly by the user.
# 'labels' are sentiment classes (0 = negative, 1 = neutral, 2 = positive) assigned by humans or a model.
# While 'rating' reflects the user's chosen score, 'labels' reflect the tone of the review text.

# Rating sentiment
def map_rating_to_sentiment(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

df_full['rating_sentiment'] = df_full['Rate'].apply(map_rating_to_sentiment)

# Map labels to sentiment categories
def map_label_to_sentiment(label):
    if label == 0:
        return 'negative'
    elif label == 1:
        return 'neutral'
    else:
        return 'positive'

df_full['label_sentiment'] = df_full['labels'].apply(map_label_to_sentiment)

# Print review type statistics
print("\nReview Type Distribution:")
review_type_counts = df_full['review_type'].value_counts() # Counts how many times each category appears in the review_type column.
for review_type, count in review_type_counts.items(): # Iterates over each review type and its count (e.g., 'short' with 15,000 reviews)
    percentage = (count / len(df_full)) * 100 # Calculates the percentage share of that type andlen(df_full) = total number of reviews.
    print(f"  - {review_type}: {count:,} reviews ({percentage:.1f}%)")
# Example output - medium    18000  ## short     15000  # long       7000

# Print rating sentiment statistics
print("\nRating Sentiment Distribution:")
sentiment_counts = df_full['rating_sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df_full)) * 100
    print(f"  - {sentiment}: {count:,} reviews ({percentage:.1f}%)")

# Print average rating for each sentiment
print("\nAverage Rating by Sentiment:")
for sentiment in ['negative', 'neutral', 'positive']:
    avg_rating = df_full[df_full['rating_sentiment'] == sentiment]['Rate'].mean()
    print(f"  - {sentiment}: {avg_rating:.1f}")

#  This is filtering the DataFrame to only include rows where the rating_sentiment 
# (derived earlier from 1-5 stars) matches the current sentiment.
# Prints the sentiment and its average rating in a clean format, rounded to 1 decimal place.

# Print some example reviews of each type
print("\nExample reviews of each type:")
for type_ in ['short', 'medium', 'long']:
    example = df_full[df_full['review_type'] == type_].iloc[0]
    print(f"\n{type_.upper()} review example:")
    print(f"Length: {example['review_length']} words")
    print(f"Text: {example['text'][:200]}...")
# Loop through each review type: short, medium, long
# Filter the DataFrame to only reviews of the current type and get the first one using iloc[0]
# Print review type in uppercase (e.g., SHORT, MEDIUM, LONG)
# Print the number of words in the selected review
# Print the first 200 characters of the review text (to keep it concise)

# Sentiment match analysis -Create a new column 'sentiment_match' to check if the label sentiment matches the sentiment derived from the user rating
df_full['sentiment_match'] = df_full['label_sentiment'] == df_full['rating_sentiment']

# Calculate and print mismatch rate
#Use ~ (tilde) to invert the Boolean values: True → False, False → True
# So ~df_full['sentiment_match'] selects the mismatched rows
# .mean() gives proportion of mismatches, then multiply by 100 for %
mismatch_rate = 100 * (~df_full['sentiment_match']).mean()
print(f"\nMismatch rate between ratings and labels: {mismatch_rate:.2f}%")

# Print sentiment match statistics
print("\nSentiment Match Distribution:")
sentiment_match_counts = df_full['sentiment_match'].value_counts() # Count how many reviews are matching (True) vs mismatching (False)
for match, count in sentiment_match_counts.items(): # Loop through each match type (True/False) and count
    percentage = (count / len(df_full)) * 100 # Calculate percentage of total dataset
    print(f"  - {'Matching' if match else 'Mismatching'}: {count:,} reviews ({percentage:.1f}%)") # Print results in clean format

# Print mismatch analysis
print("\nMismatch Analysis:")
mismatch_df = df_full[~df_full['sentiment_match']] # Create a new DataFrame that contains only mismatched sentiment rows
print(f"Total mismatches: {len(mismatch_df):,}") # Print total number of mismatched reviews
print("\nMismatch patterns (Rating Sentiment → Label Sentiment):")
pattern_counts = mismatch_df.groupby(['rating_sentiment', 'label_sentiment']).size() # Group the mismatches by (rating_sentiment, label_sentiment) and count how many in each group
for (rating_sent, label_sent), count in pattern_counts.items(): # Loop through each mismatch pair and print how many cases occurred
    percentage = (count / len(mismatch_df)) * 100 # Calculate what percentage each pattern makes up of all mismatches
    print(f"  - {rating_sent} → {label_sent}: {count:,} cases ({percentage:.1f}%)") # Print in format: positive → negative: 1,200 cases (35.4%)

# Save to CSV
print("\nSaving to CSV...")
try:
    df_full.to_csv("flipkart_reviews_full.csv", index=False)
    
    # Print information about the final dataset
    print("\n✅ CSV file saved as 'flipkart_reviews_full.csv'")
    print(f"Total number of rows: {len(df_full):,}")
    print("\nFinal columns in the dataset:")
    for col in df_full.columns:
        print(f"  - {col}")
except Exception as e:
    print(f"Error saving CSV file: {e}")
    raise
