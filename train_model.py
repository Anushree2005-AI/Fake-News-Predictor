import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or not isinstance(text, str):
        return ''
    
    # Remove non-letters, lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    
    # Stem words and remove stopwords
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = [ps.stem(word) for word in text if word not in stop_words]
    
    return ' '.join(text)

print("Loading datasets...")
# Load dataset (example using Fake & True CSVs)
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

fake['label'] = 'fake'
real['label'] = 'real'

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Combine title & text if present
print("Combining text columns...")
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Clean content
print("Cleaning text data (this may take a while)...")
df['content_clean'] = df['content'].apply(clean_text)

# Remove empty rows
df = df[df['content_clean'].str.strip() != '']

# Prepare features and labels
X = df['content_clean']
y = df['label']

print(f"Total samples: {len(df)}")
print(f"Fake news: {sum(y == 'fake')}, Real news: {sum(y == 'real')}")

# Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
accuracy = model.score(X_test_vec, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Save model + vectorizer
print("Saving model and vectorizer...")
joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model trained and saved successfully!")