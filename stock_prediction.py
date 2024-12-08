import asyncio
from telethon import TelegramClient
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Download NLTK Data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Telegram API credentials (replace with your actual API ID and Hash)
API_ID = "23823060"
API_HASH = "c3a0761896ba9d59cdfa08110492de17"

# Telegram channel or group to scrape
CHANNEL_NAME = "@Nadd8050"  # Replace with the actual channel name

# 1. Scraping Data from Telegram
async def scrape_telegram_messages(channel_name, limit=100):
    """Scrape messages from a Telegram channel."""
    messages = []
    async with TelegramClient("stock_sentiment", API_ID, API_HASH) as client:
        async for message in client.iter_messages(channel_name, limit=limit):
            if message.text:
                messages.append(message.text)
    return messages

# 2. Data Preprocessing and Sentiment Analysis
def preprocess_and_analyze(messages):
    """Preprocess and perform sentiment analysis on the scraped messages."""
    data = pd.DataFrame(messages, columns=["Message"])
    data["Sentiment"] = data["Message"].apply(lambda x: sia.polarity_scores(x)["compound"])
    data["Label"] = data["Sentiment"].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
    return data

# 3. Prediction Model for Stock Movement
def train_model(data):
    """Train a machine learning model to predict stock movements based on sentiment."""
    # Generate fake stock movement labels for demonstration (1: Up, 0: Neutral, -1: Down)
    data["Stock_Movement"] = data["Label"]  # In real scenarios, use real stock movement data
    
    # Prepare training and test sets
    X = data[["Sentiment"]]
    y = data["Stock_Movement"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, X_test, y_test, y_pred

# 4. Visualization
def plot_results(y_test, y_pred):
    """Plot the actual vs predicted results."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual", marker="o")
    plt.plot(y_pred, label="Predicted", marker="x")
    plt.title("Actual vs Predicted Stock Movements")
    plt.xlabel("Data Point")
    plt.ylabel("Movement (1: Up, 0: Neutral, -1: Down)")
    plt.legend()
    plt.grid()
    plt.show()

# Main Program
if __name__ == "__main__":
    try:
        print("Scraping Telegram messages...")
        messages = asyncio.run(scrape_telegram_messages(CHANNEL_NAME, limit=200))
        print(f"Scraped {len(messages)} messages.")
        
        print("Preprocessing and analyzing sentiment...")
        sentiment_data = preprocess_and_analyze(messages)
        print(sentiment_data.head())
        
        print("Training prediction model...")
        model, X_test, y_test, y_pred = train_model(sentiment_data)
        
        print("Visualizing results...")
        plot_results(y_test, y_pred)
        
    except Exception as e:
        print(f"An error occurred: {e}")
