from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import warnings
import os

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Download stopwords once
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load model + vectorizer
try:
    model = joblib.load('models/model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    print("✓ Model and vectorizer loaded successfully")
except FileNotFoundError as e:
    print(f"ERROR: Could not load model files. Make sure 'models/model.pkl' and 'models/vectorizer.pkl' exist.")
    print(f"Run train_model.py first to create the model files.")
    exit(1)

def clean_text(text):
    """Clean and preprocess text for prediction"""
    if not text or not isinstance(text, str):
        return ''
    
    # Remove non-letters, lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    
    # Stem words and remove stopwords
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = [ps.stem(word) for word in text if word not in stop_words]
    
    return ' '.join(text)

@app.route("/", methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            news = request.form.get('news', '').strip()
            
            if not news:
                error = "Please enter some text to analyze."
            elif len(news) < 10:
                error = "Please enter a longer text (at least 10 characters)."
            else:
                # Clean and predict
                clean = clean_text(news)
                
                if not clean:
                    error = "Could not extract meaningful text. Please try again."
                else:
                    vect = vectorizer.transform([clean])
                    pred = model.predict(vect)[0]
                    prediction = pred.upper()
                    
        except Exception as e:
            error = f"An error occurred: {str(e)}"
            print(f"Error during prediction: {e}")
    
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == "__main__":
    # Check if templates folder exists
    if not os.path.exists('templates'):
        print("WARNING: 'templates' folder not found. Creating it...")
        os.makedirs('templates')
        print("Please add index.html to the templates folder.")
    
    app.run(debug=True, host='127.0.0.1', port=5000)