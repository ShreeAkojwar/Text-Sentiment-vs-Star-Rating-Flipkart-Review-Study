# 🎯 What is it?
# This script performs aspect-based sentiment analysis on Flipkart reviews
# It identifies specific aspects (like quality, price, delivery) and analyzes sentiment for each
# 🧠 Why is This Insightful?
# - Breaks down overall sentiment into specific product/service aspects
# - Shows which aspects drive positive/negative reviews
# - Helps identify specific areas for improvement
# 💼 Business Value
# - Targeted improvement areas for products/services
# - Better understanding of customer priorities
# - Data-driven decision making for product development

import pandas as pd
import re
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the data
    print("Loading data...")
    df_full = pd.read_csv('flipkart_reviews_with_sentiment.csv')
    
    # Step 1: Define comprehensive aspect keywords
    aspect_keywords = {
        "quality": [
            "quality", "durability", "build", "material", "sturdy", "reliable", "construction",
            "solid", "durable", "build quality", "finish", "workmanship", "craftsmanship"
        ],
        "performance": [
            "performance", "speed", "fast", "slow", "efficient", "powerful", "works", "working",
            "functionality", "functions", "features", "capability"
        ],
        "cost": [
            "price", "cost", "value", "expensive", "cheap", "worth", "affordable", "pricing",
            "budget", "money", "costly", "priced", "investment"
        ],
        "delivery": [
            "delivery", "shipping", "arrived", "packaging", "courier", "shipment", "delivered",
            "package", "arrival", "on time", "late", "quick delivery", "fast delivery"
        ],
        "customer_service": [
            "service", "support", "customer service", "help", "response", "assistance",
            "communication", "customer care", "helpline", "contact"
        ],
        "usability": [
            "easy", "difficult", "simple", "complicated", "user friendly", "convenient",
            "hassle", "intuitive", "straightforward", "usage", "using"
        ],
        "appearance": [
            "look", "design", "style", "color", "appearance", "beautiful", "attractive",
            "elegant", "aesthetic", "looks", "pretty", "ugly"
        ]
    }

    # Step 2: Create a function to assign sentiment per aspect for a given review
    def get_aspect_sentiment(review):
        if not isinstance(review, str):
            return {aspect: 0 for aspect in aspect_keywords}
            
        review = review.lower()
        aspect_scores = {}
        
        for aspect, keywords in aspect_keywords.items():
            score = 0
            matched = False
            matched_sentences = []
            
            for keyword in keywords:
                sentences = re.findall(r'([^.]*?\b' + re.escape(keyword) + r'\b[^.]*\.)', review)
                if sentences:
                    matched = True
                    matched_sentences.extend(sentences)
            
            if matched:
                # Calculate average sentiment across all matched sentences
                sentiments = [TextBlob(sent).sentiment.polarity for sent in matched_sentences]
                avg_sentiment = np.mean(sentiments)
                
                if avg_sentiment > 0.1:
                    score = 1
                elif avg_sentiment < -0.1:
                    score = -1
                else:
                    score = 0
                    
            aspect_scores[aspect] = score
            
        return aspect_scores

    print("Analyzing aspects in reviews...")
    # Step 3: Apply sentiment analysis to each review
    aspect_sentiments = df_full['Review'].apply(get_aspect_sentiment)

    # Step 4: Convert results to DataFrame
    aspect_df = pd.DataFrame(aspect_sentiments.tolist())
    
    # Combine with original data
    df_aspect = pd.concat([df_full[['Review', 'Rate', 'product_name']], aspect_df], axis=1)
    
    # Calculate aspect statistics
    print("\nAspect Sentiment Analysis Results:")
    for aspect in aspect_keywords.keys():
        total = len(df_aspect)
        mentioned = (df_aspect[aspect] != 0).sum()
        positive = (df_aspect[aspect] == 1).sum()
        negative = (df_aspect[aspect] == -1).sum()
        neutral = (df_aspect[aspect] == 0).sum()
        
        mention_rate = (mentioned / total * 100)
        if mentioned > 0:
            positive_rate = (positive / mentioned * 100)
            negative_rate = (negative / mentioned * 100)
            neutral_rate = (neutral / mentioned * 100)
        else:
            positive_rate = negative_rate = neutral_rate = 0
            
        print(f"\n{aspect.title()} Aspect:")
        print(f"Mention Rate: {mention_rate:.6f}% ({mentioned:,} reviews)")
        if mentioned > 0:
            print(f"Sentiment Distribution (among mentioned):")
            print(f"- Positive: {positive_rate:.6f}% ({positive:,} reviews)")
            print(f"- Negative: {negative_rate:.6f}% ({negative:,} reviews)")
            print(f"- Neutral: {neutral_rate:.6f}% ({neutral:,} reviews)")
    
    # Create visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(15, 8))
    
    # Prepare data for plotting
    aspects = list(aspect_keywords.keys())
    mention_counts = [(df_aspect[aspect] != 0).sum() for aspect in aspects]
    positive_counts = [(df_aspect[aspect] == 1).sum() for aspect in aspects]
    negative_counts = [(df_aspect[aspect] == -1).sum() for aspect in aspects]
    
    # Create grouped bar chart
    x = np.arange(len(aspects))
    width = 0.35
    
    plt.bar(x - width/2, positive_counts, width, label='Positive', color='green', alpha=0.6)
    plt.bar(x + width/2, negative_counts, width, label='Negative', color='red', alpha=0.6)
    
    plt.xlabel('Aspects')
    plt.ylabel('Number of Reviews')
    plt.title('Sentiment Distribution Across Different Aspects')
    plt.xticks(x, [aspect.title() for aspect in aspects], rotation=45)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(positive_counts):
        plt.text(i - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate(negative_counts):
        plt.text(i + width/2, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visual_images/Figure_15_Aspect_Sentiment_Analysis.png', 
                bbox_inches='tight', 
                dpi=300)
    print("\nPlot saved as 'Figure_15_Aspect_Sentiment_Analysis.png'")
    
    # Save the results
    df_aspect.to_csv('processed_data/aspect_sentiment_results.csv', index=False)
    print("\nDetailed results saved to 'processed_data/aspect_sentiment_results.csv'")

except FileNotFoundError:
    print("Error: Could not find the data file. Please ensure 'flipkart_reviews_with_sentiment.csv' exists in the root directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
