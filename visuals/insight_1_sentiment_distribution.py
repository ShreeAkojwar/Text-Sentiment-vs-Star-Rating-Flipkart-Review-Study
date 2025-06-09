# 🎯 What is it?
# We analyze the distribution of label_sentiment across the entire dataset — i.e., how many reviews are labeled as positive, neutral, or negative based on their text.
# 💡 Why is it important? (First Principles Thinking)
# Any e-commerce platform needs to understand the general mood of its customers.
# If most reviews are positive, that signals good product quality or buyer satisfaction.
# If there's a high number of neutral or negative reviews, this may hint at recurring issues or unmet expectations.
# This gives a baseline for all further analysis — you must know where you stand before asking "why."


import pandas as pd                  # Import pandas for data manipulation
import matplotlib.pyplot as plt      # Imports matplotlib for plotting charts
import seaborn as sns               # Imports seaborn, a high-level plotting library built on top of matplotlib

# Load the dataset
df_full = pd.read_csv('flipkart_reviews_with_sentiment.csv')

# Convert sentiment codes to labels for better visualization
sentiment_map = {2: 'positive', 1: 'neutral', 0: 'negative'}
df_full['label_sentiment'] = df_full['labels'].map(sentiment_map)

sns.set(style="whitegrid")          # Sets a clean background grid for the plot (optional aesthetic choice)

# Create a new figure of size 8 inches wide and 6 inches tall
plt.figure(figsize=(8, 6))

# Plot a bar chart (countplot) using seaborn
# - data: DataFrame to use
# - x: the column to count values of 'label_sentiment'
# - palette: color scheme (Set2 is a pastel color palette)
# - order: sets the order of bars so it's always ['positive', 'neutral', 'negative']
ax = sns.countplot(data=df_full, x='label_sentiment', palette='Set2',
                   order=['positive', 'neutral', 'negative'])

# Add value labels above each bar
for p in ax.patches:                               # Loop through each bar in the chart
    height = p.get_height()                        # Get the height of the current bar (i.e., its count)
    ax.annotate(f'{height:,}',                     # Annotate with the count, comma-formatted
                (p.get_x() + p.get_width() / 2., height),  # Position the label at center-top of the bar
                ha='center', va='bottom', fontsize=11)     # Center align horizontally; place just above the bar

# Add a title to the plot
plt.title('Sentiment Distribution Based on Review Text', fontsize=14)

# Label the x-axis
plt.xlabel('Label Sentiment')

# Label the y-axis
plt.ylabel('Number of Reviews')

# Adjust layout to prevent overlap and auto-scale spacing
plt.tight_layout()

# Display the plot
plt.show()
