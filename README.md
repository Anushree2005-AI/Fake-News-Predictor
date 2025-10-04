# Fake News Detection using Machine Learning & Flask  

#Overview  
This project is a Fake News Detection Web App that predicts whether a given news article is real or fake.  
It uses Natural Language Processing (NLP) techniques and Machine Learning to classify news articles based on their content.  
The app is built with **Flask** for the backend and provides an easy-to-use web interface.  

---

Features  
- Input a news headline or article text  
- Predicts whether the news is Fake or Real  
- Machine Learning pipeline with TF-IDF and Logistic Regression
- Flask-based web application for deployment  

---

Dataset  
We used the Fake News Dataset available on Kaggle:  
- `title` → The headline of the news  
- `text` → The body/content of the news  
- `label` → `1` = Fake news, `0` = Real news  

---

Tech Stack  
- Python 3 
- Scikit-learn → ML model & evaluation  
- Pandas, Numpy → Data handling  
- NLTK → Text preprocessing (stopwords, stemming, etc.)  
- TF-IDF Vectorizer → Convert text to numerical features  
- Flask → Web app framework  
- HTML→ Frontend UI  

---

How It Works  
1. Preprocessing
   - Remove punctuation, lowercase text, and remove stopwords  
   - Apply TF-IDF to convert text into numerical features  

2. Model Training  
   - Logistic Regression classifier is trained on the dataset  
   - Model is saved using `joblib`  

3. Prediction 
   - User enters news text in the Flask app  
   - Text is preprocessed and transformed  
   - Model predicts: **Fake (1)** or **Real (0)**  

4. Web App
   - Flask serves a form where the user can enter text  
   - Displays prediction result instantly  

---

Run Locally  
Clone the repo and install dependencies:  

```bash
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
