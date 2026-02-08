import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import urllib.request
import zipfile
import os

class SpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        self.model = MultinomialNB()
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess the text"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def train(self, messages, labels):
        """Train the spam detector"""
        print("Preprocessing messages...")
        processed_messages = [self.preprocess_text(msg) for msg in messages]
        
        print("Converting text to features...")
        X = self.vectorizer.fit_transform(processed_messages)
        
        print("Training the model...")
        self.model.fit(X, labels)
        self.is_trained = True
        print("Training completed!")
        
    def evaluate(self, messages, labels):
        """Evaluate model performance"""
        if not self.is_trained:
            print("Model not trained yet!")
            return
        
        processed_messages = [self.preprocess_text(msg) for msg in messages]
        X = self.vectorizer.transform(processed_messages)
        predictions = self.model.predict(X)
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        print(f"\nAccuracy: {accuracy_score(labels, predictions):.2%}")
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=['Ham', 'Spam']))
        
        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
        plt.show()
        
    def predict(self, message):
        """Predict if a message is spam"""
        if not self.is_trained:
            return {"error": "Model not trained yet!"}
        
        processed = self.preprocess_text(message)
        features = self.vectorizer.transform([processed])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'message': message,
            'is_spam': bool(prediction),
            'spam_probability': float(probability[1]),
            'ham_probability': float(probability[0])
        }
    
    def save_model(self, filepath='spam_detector.pkl'):
        """Save the trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer, 
                'model': self.model,
                'is_trained': self.is_trained
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='spam_detector.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
            self.is_trained = data.get('is_trained', True)
        print(f"Model loaded from {filepath}")


def download_spam_dataset():
    """Download SMS Spam Collection dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "smsspamcollection.zip"
    
    if not os.path.exists("SMSSpamCollection"):
        print("Downloading SMS Spam Collection dataset...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            os.remove(zip_path)
            print("Dataset downloaded and extracted successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Creating sample dataset instead...")
            create_sample_dataset()
            return "sample_spam_data.csv"
        
        return "SMSSpamCollection"
    else:
        print("Dataset already exists!")
        return "SMSSpamCollection"


def create_sample_dataset():
    """Create a sample dataset if download fails"""
    sample_data = {
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'] * 50,
        'message': [
            "URGENT! You have won a $1000 gift card. Click here to claim now!",
            "Hey, are we still meeting for lunch tomorrow?",
            "Congratulations! You've been selected for a FREE iPhone. Act now!",
            "Can you send me the project report by end of day?",
            "WINNER! Claim your prize money now. Limited time offer!!!",
            "Don't forget to pick up milk on your way home",
            "You have been chosen to receive a cash prize. Reply YES to confirm",
            "Meeting rescheduled to 3pm in conference room B",
            "Make money fast! Work from home. No experience needed!!!",
            "Thanks for your help with the presentation today"
        ] * 50
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_spam_data.csv', index=False)
    print("Sample dataset created!")


def load_dataset(filepath):
    """Load and prepare the dataset"""
    if filepath == "SMSSpamCollection":
        # Load the SMS Spam Collection format
        df = pd.read_csv(filepath, sep='\t', names=['label', 'message'], encoding='utf-8')
    else:
        df = pd.read_csv(filepath)
    
    # Convert labels to binary (0 for ham, 1 for spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"\nDataset loaded: {len(df)} messages")
    print(f"Spam messages: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    print(f"Ham messages: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
    
    return df


def interactive_demo(detector):
    """Interactive demonstration"""
    print("\n" + "="*60)
    print("INTERACTIVE SPAM DETECTION DEMO")
    print("="*60)
    print("Enter messages to check if they're spam (type 'quit' to exit)\n")
    
    while True:
        message = input("Enter a message: ")
        
        if message.lower() == 'quit':
            break
        
        if not message.strip():
            continue
            
        result = detector.predict(message)
        
        print("\n" + "-"*60)
        print(f"Message: {result['message']}")
        print(f"Classification: {'🚫 SPAM' if result['is_spam'] else '✅ HAM (Not Spam)'}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")
        print(f"Ham Probability: {result['ham_probability']:.2%}")
        print("-"*60 + "\n")


def batch_demo(detector):
    """Demonstrate with predefined test messages"""
    test_messages = [
        "Congratulations! You've won a free vacation to Hawaii! Click here now!",
        "Hey mom, I'll be home late tonight. Don't wait up.",
        "URGENT: Your bank account has been compromised. Verify your identity immediately!",
        "Can we reschedule our meeting to next Tuesday?",
        "Get rich quick! Make $5000 per week working from home!!!",
        "Thanks for dinner last night. Had a great time!",
        "You have been selected for a special offer. Limited time only!",
        "Please review the attached document and send me your feedback",
        "FREE MONEY! No strings attached! Act now before it's too late!",
        "Happy birthday! Hope you have an amazing day!"
    ]
    
    print("\n" + "="*60)
    print("BATCH PREDICTION DEMO")
    print("="*60 + "\n")
    
    for i, msg in enumerate(test_messages, 1):
        result = detector.predict(msg)
        print(f"{i}. Message: {msg[:60]}...")
        print(f"   Classification: {'🚫 SPAM' if result['is_spam'] else '✅ HAM'} "
              f"(Spam: {result['spam_probability']:.2%})\n")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("SPAM MESSAGE DETECTOR - TRAINING & DEMONSTRATION")
    print("="*60)
    
    # Step 1: Download or create dataset
    dataset_path = download_spam_dataset()
    
    # Step 2: Load dataset
    df = load_dataset(dataset_path)
    
    # Step 3: Split data
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    print(f"Training set: {len(X_train)} messages")
    print(f"Testing set: {len(X_test)} messages")
    
    # Step 4: Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    detector = SpamDetector()
    detector.train(X_train, y_train)
    
    # Step 5: Evaluate model
    detector.evaluate(X_test, y_test)
    
    # Step 6: Save model
    detector.save_model()
    
    # Step 7: Batch demonstration
    batch_demo(detector)
    
    # Step 8: Interactive demo
    print("\nWould you like to try the interactive demo? (yes/no)")
    choice = input("> ").lower()
    
    if choice in ['yes', 'y']:
        interactive_demo(detector)
    
    print("\n" + "="*60)
    print("Demo completed! Model saved as 'spam_detector.pkl'")
    print("You can load this model later using detector.load_model()")
    print("="*60)