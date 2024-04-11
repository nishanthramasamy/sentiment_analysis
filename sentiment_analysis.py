import string
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import streamlit as st

ylp = pd.read_csv(rf"yelp_labelled.txt", sep='\t', names=['reviews', 'label'] )
print(ylp.head())
print(ylp.isnull().sum())
print(ylp.shape)

amz = pd.read_csv(rf"amazon_cells_labelled.txt", sep='\t', names=['reviews', 'label'])
print(amz.isnull().sum())
print(amz.head())
print(amz.shape)

imdb = pd.read_csv(rf"imdb_labelled.txt", sep='\t', names=['reviews', 'label'])
print(imdb.isnull().sum())
print(imdb.head())
print(imdb.shape)

df= pd.concat([ylp, amz, imdb], ignore_index=True, axis=0)
print(df.shape)


nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words
punctuations = string.punctuation

def preprocessing(sentences):
  text = nlp(sentences)
  tokens = []
  for words in text:
    if words.lemma_ != '-PRON-':
      temp = words.lemma_.lower().strip()
    else:
      temp = words.lower_
    tokens.append(temp)
  cleaned_tokens = []
  for word in tokens:
    if word not in punctuations and word not in stopwords:
      cleaned_tokens.append(word)
  return cleaned_tokens

X = df['reviews']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train.shape,y_test.shape)

tfidf  = TfidfVectorizer(tokenizer = preprocessing)
svm = LinearSVC()
steps = [('tfidf',tfidf),('svm',svm)]
pipe = Pipeline(steps)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test,y_pred))
print("\n\n")
print(confusion_matrix(y_test,y_pred))

st.title("Welcome!!!")
st.write("Let's see if you are in a positive/negative state")
input = st.text_input("Enter your thoughts")
senti = pipe.predict([input])

button = st.button("Check Mood")
if button:
    if senti == 1:
        st.write("Yay! You are happy and positive")
    if senti == 0:
        st.write("Sorry! It is sad to see you sad. Just give a Smile. Sadness gets into madness")
