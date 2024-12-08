# Stock-Movement-Analysis-Based-on-Social-Media-Sentiment


# Steps to Run code

1. Replace YOUR_API_ID and YOUR_API_HASH with your Telegram API credentials.
2. Replace @yourchannel with the actual Telegram channel or group username.
3. ave the script as stock_prediction.py and run it
4. Prerequisites
   Make sure you have:
   Python 3.8 or higher
   Necessary libraries installed: telethon, nltk, pandas, scikit-learn, and matplotlib
5. Ensure your phone number is in the correct international format:

    Example: For a phone number in the US: +1234567890
    Include the country code (+ followed by the numeric code).
   
# Key Features

1. Telegram Scraping:
   Uses Telethon to fetch messages from a specified Telegram channel or group.
2. Sentiment Analysis:
   Applies NLTK's VADER SentimentIntensityAnalyzer to calculate sentiment scores.
   Labels sentiment as positive (1), neutral (0), or negative (-1).
3. Stock Movement Prediction:
   Trains a RandomForestClassifier to predict stock movement based on sentiment.
   Generates fake stock labels for demonstration (replace with real stock price movement data).
4. Visualization:
   Plots actual vs. predicted stock movements for analysis.
