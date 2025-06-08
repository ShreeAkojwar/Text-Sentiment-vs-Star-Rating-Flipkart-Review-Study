import pandas as pd

# Load the parquet files
print("Loading Parquet files...")
df_train = pd.read_parquet("/Users/shreeakojwar/Downloads/IIMN_Final /train.parquet")
df_test = pd.read_parquet("/Users/shreeakojwar/Downloads/IIMN_Final /test.parquet")

# Combine them into a single dataset
print("Combining datasets...")
df_full = pd.concat([df_train, df_test], ignore_index=True)

# Print original column names
print("\nOriginal columns in the dataset:")
for col in df_full.columns:
    print(f"  - {col}")
"""The f"..." is a formatted string literal (called an "f-string").
It lets you insert variables directly into strings using curly braces {}."""

# Calculate review length using the correct column name 'text'
df_full['review_length'] = df_full['text'].apply(lambda x: len(str(x).split()))
"""Part	Meaning
df_full['text']	Accesses the review text column
.apply(...)	Applies a function to each row in the text column
lambda x: ...	Anonymous function (takes each review as x)
str(x)	Ensures the input is a string (in case of None/NaN)
.split()	Splits the text into words (by spaces)
len(...)	Counts how many words are in the list """

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
# They may not always match — for example, a user may give 4 stars but write a negative review.

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
review_type_counts = df_full['review_type'].value_counts()
for review_type, count in review_type_counts.items():
    percentage = (count / len(df_full)) * 100
    print(f"  - {review_type}: {count:,} reviews ({percentage:.1f}%)")

"""Just prints a heading with a line break (\n) before it.
 df_full['review_type'].value_counts()
Counts how many times each category appears in the review_type column.
Example output
medium    18000  
short     15000  
long       7000
Iterates over each review type and its count (e.g., 'short' with 15,000 reviews)
Calculates the percentage share of that type.
len(df_full) = total number of reviews."""

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
""" This is filtering the DataFrame to only include rows where the rating_sentiment 
(derived earlier from 1-5 stars) matches the current sentiment.
Prints the sentiment and its average rating in a clean format, rounded to 1 decimal place.
"""

# Print some example reviews of each type
print("\nExample reviews of each type:")
for type_ in ['short', 'medium', 'long']:
    example = df_full[df_full['review_type'] == type_].iloc[0]
    print(f"\n{type_.upper()} review example:")
    print(f"Length: {example['review_length']} words")
    print(f"Text: {example['text'][:200]}...")

# Sentiment match analysis
df_full['sentiment_match'] = df_full['label_sentiment'] == df_full['rating_sentiment']

# Calculate and print mismatch rate
mismatch_rate = 100 * (~df_full['sentiment_match']).mean()
print(f"\nMismatch rate between ratings and labels: {mismatch_rate:.2f}%")

# Print sentiment match statistics
print("\nSentiment Match Distribution:")
sentiment_match_counts = df_full['sentiment_match'].value_counts()
for match, count in sentiment_match_counts.items():
    percentage = (count / len(df_full)) * 100
    print(f"  - {'Matching' if match else 'Mismatching'}: {count:,} reviews ({percentage:.1f}%)")

# Print mismatch analysis
print("\nMismatch Analysis:")
mismatch_df = df_full[~df_full['sentiment_match']]
print(f"Total mismatches: {len(mismatch_df):,}")
print("\nMismatch patterns (Rating Sentiment → Label Sentiment):")
pattern_counts = mismatch_df.groupby(['rating_sentiment', 'label_sentiment']).size()
for (rating_sent, label_sent), count in pattern_counts.items():
    percentage = (count / len(mismatch_df)) * 100
    print(f"  - {rating_sent} → {label_sent}: {count:,} cases ({percentage:.1f}%)")

# Save to CSV
print("\nSaving to CSV...")
df_full.to_csv("flipkart_reviews_full.csv", index=False)

# Print information about the final dataset
print("\n✅ CSV file saved as 'flipkart_reviews_full.csv'")
print(f"Total number of rows: {len(df_full):,}")
print("\nFinal columns in the dataset:")
for col in df_full.columns:
    print(f"  - {col}")

