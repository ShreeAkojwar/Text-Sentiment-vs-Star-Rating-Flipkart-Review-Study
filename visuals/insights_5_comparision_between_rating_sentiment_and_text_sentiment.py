# 🎯 What is it?
# We compare the number of reviews by:
# rating_sentiment → what the user clicked (based on stars)
# label_sentiment → what the text really says (based on NLP)
# 🧠 Why is This Insightful?
# This checks whether people rate emotionally or rationally.
# You’ll often find mismatches like:
# People giving 4★ but writing complaints.
# Or giving 2★ but praising the product (maybe due to delivery delay, etc.)
# 💼 Business Value
# Helps brands detect false positivity (good ratings with bad experience).
# Improves trust scoring for reviews.
# Suggests whether customers are sugarcoating or overreacting with ratings.
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Prepare counts
label_counts = df_full['label_sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'])
rating_counts = df_full['rating_sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'])

# Create side-by-side bar chart
plt.figure(figsize=(9, 5))
bar_width = 0.35
x = range(len(label_counts))

# Plotting
plt.bar(x, label_counts.values, width=bar_width, label='Text Sentiment', color='skyblue')
plt.bar([p + bar_width for p in x], rating_counts.values, width=bar_width, label='Rating Sentiment', color='salmon')

# Add labels and legend
plt.xticks([p + bar_width / 2 for p in x], ['Positive', 'Neutral', 'Negative'])
plt.ylabel('Number of Reviews')
plt.title('Comparison of Text vs Rating-Based Sentiment')
plt.legend()
plt.tight_layout()
plt.show()
