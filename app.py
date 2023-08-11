from flask import Flask, render_template, request
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model_path = 'models/fake_news_model.pkl'
vectorizer_path = 'models/fake_news_tfidfvect.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))
loaded_tfidfvect = pickle.load(open(vectorizer_path, 'rb'))

# Initialize natural language processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text[0]
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\n', ' ', text)  # Replace newline characters with a space
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    words = nltk.word_tokenize(text)  # Tokenize
    corpus = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    preprocessed_text = ' '.join(corpus)
    return preprocessed_text


def detect_fake_news(news_text):
    preprocessed_news = preprocess_text(news_text)
    vectorized_input = loaded_tfidfvect.transform([preprocessed_news])
    prediction = loaded_model.predict(vectorized_input)[0]
    return prediction


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['message']
        prediction = detect_fake_news(news_text)
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html', prediction="Something went wrong")


if __name__ == '__main__':
    app.run(debug=True)
