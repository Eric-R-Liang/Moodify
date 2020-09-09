import joblib
import pickle
import re
import pandas as pd
#from textblob import Word

def predict_mood(inputs):
    #data = pickle.load(open("Mood/merged_training.pkl", "rb"))
    #data['text'] = data['text'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))
    model = joblib.load("Model/Mood/mood_model.pkl")
    inputs = pd.Series(inputs)
    inputs = inputs.str.replace('[^\w\s]', ' ')

    stop = open("Model/Mood/english.txt", 'r')
    inputs = inputs.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    #inputs = inputs.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    count_vect =joblib.load("Model/Mood/Vectorizer.pkl")
    inputs_count = count_vect.transform(inputs)
    inputs_pred = model.predict(inputs_count)
    pred = inputs_pred[0]
    if (pred == 0):
        mood = "anger"
    elif (pred == 1):
        mood = "fear"
    elif (pred == 2):
        mood = "joy"
    elif (pred == 3):
        mood = "love"
    elif (pred == 4):
        mood = "sad"
    elif (pred == 5):
        mood = "surprised"
    return mood




