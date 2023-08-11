# AI Fake News Detector (Experimental) (!!! Trained on very limited dataset !!!)

![image](https://github.com/ahsann455/AI-Fake-News-Detector/assets/97152316/67c7656f-e59a-498a-8781-05d9ca8747d5)

##  Methodology:
Several strategies can be employed, including supervised learning, unsupervised learning, or deep learning. For counterfeit news identification, the supervised learning approach is frequently used. In this approach, the model is trained using labeled data, where the true classification of each article (real or fake) is known. The aim is to create a model that can reliably classify new, unseen articles as genuine or fake.

## Data Collection:
The initial step entails acquiring data for training and model evaluation. Multiple sources of data are available, such as online news outlets, social media platforms, and historical news archives. Data can be collected from these sources using web scraping methods, or you can directly download the data from the repository's dataset folder.

## Data Exploration:
Following data preparation, it's crucial to explore the dataset to uncover its underlying characteristics. Utilizing libraries like matplotlib, seaborn, and wordcloud, you can analyze the data and identify potential trends or patterns.

## Data Preprocessing and Cleansing:
After data exploration, potential data quality issues may emerge that need to be addressed. The subsequent step involves preparing the data for analysis. This encompasses data cleaning and preprocessing tasks, such as eliminating duplicates, handling missing values, and rectifying data inaccuracies.

## Feature Crafting:
Feature engineering entails selecting and transforming the data's variables (features) that the machine learning algorithm will use. In the given dataset, only the news article's title and body were used, with the author column omitted.

## Model Training:
Following feature engineering, the next phase is to train the machine learning model. Multiple algorithms can be employed for model training. In this instance, logistic regression, decision tree classification, and the PassiveAggressiveClassifier were utilized.

## Constructing the Model:
Upon model training, the subsequent step involves constructing the final model. This includes fine-tuning model hyperparameters and selecting the best-performing model. From the tested models, the PassiveAggressiveClassifier was chosen as the final model for predicting new data.

## Model Assessment:
Subsequent to model construction, it's imperative to evaluate its performance. For this purpose, accuracy was adopted as the evaluation metric.

## Predictions:
Once the model has been evaluated, it can be employed to predict the category of new, unseen articles.

## Project Startup:
-- Clone the repository. <br>
-- Launch the command prompt in the designated directory. <br>
-- Execute pip install -r requirements.txt. <br>
-- Run Fake_News_Detector.py and navigate to http://127.0.0.1:5000/. <br>

## Legal Disclaimer:
Predictions made by the model may occasionally be imprecise, and the repository's owner does not guarantee their accuracy, particularly when applied to the most recent news developments.
