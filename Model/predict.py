import tensorflow as tf
import csv
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()


def construct_prereq_TS(filename):
    corpus = []
    with open(filename, errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            text = row[4].translate(str.maketrans('', '', string.punctuation))
            corpus.append(text)
        corpus = corpus[1:]

    tokenizer.fit_on_texts(corpus)
    word_index = tokenizer.word_index
    total_words = len(word_index) + 1


    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    return max_sequence_len

def construct_prereq(filename):
    data = open(filename).read()
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    word_index = tokenizer.word_index
    total_words = len(word_index) + 1


    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    return max_sequence_len


def predict(class_choice,data):
    next_words = 20
    seed_text = data
    if (class_choice == "Taylor Swift"):
        max_sequence_len = construct_prereq_TS("Model/Taylor_Swift_Model/TSwift.csv")
        model = tf.keras.models.load_model("Model/Taylor_Swift_Model/TaylorSwift.h5")
    elif (class_choice == "ABBA"):
        max_sequence_len = construct_prereq("Model/ABBA_Model/ABBA.txt")
        model = tf.keras.models.load_model("Model/ABBA_Model/ABBA.h5")
    elif (class_choice == "Drake"):
        max_sequence_len = construct_prereq("Model/Drake_Model/Drake.txt")
        model = tf.keras.models.load_model("Model/Drake_Model/Drake.h5")
    elif (class_choice == "Ed Sheeran"):
        max_sequence_len = construct_prereq("Model/Ed_Sheeran_Model/Ed_Sheeran.txt")
        model = tf.keras.models.load_model("Model/Ed_Sheeran_Model/Ed_Sheeran.h5")
    elif (class_choice == "Jonas Brothers"):
        max_sequence_len = construct_prereq("Model/Jonas_Brothers_Model/Jonas_Brothers.txt")
        model = tf.keras.models.load_model("Model/Jonas_Brothers_Model/Jonas_Brothers.h5")
    elif (class_choice == "Justin Bieber"):
        max_sequence_len = construct_prereq("Model/Justin_Bieber_Model/Justin_Bieber.txt")
        model = tf.keras.models.load_model("Model/Justin_Bieber_Model/Justin_Bieber.h5")
    elif (class_choice == "Katy Perry"):
        max_sequence_len = construct_prereq("Model/Katy_Perry_Model/Katy_Perry.txt")
        model = tf.keras.models.load_model("Model/Katy_Perry_Model/Katy_Perry.h5")
    elif (class_choice == "Kendrick Lamar"):
        max_sequence_len = construct_prereq("Model/Kendrick_Lamar_Model/Kendrick_Lamar.txt")
        model = tf.keras.models.load_model("Model/Donald_Trump_Model/DonaldTrump.h5")
    elif (class_choice == "Lady Gaga"):
        max_sequence_len = construct_prereq("Model/Lady_Gaga_Model/Lady_Gaga.txt")
        model = tf.keras.models.load_model("Model/Lady_Gaga_Model/Lady_Gaga.h5")
    elif (class_choice == "Maroon5"):
        max_sequence_len = construct_prereq("Model/Maroon5_Model/Maroon5.txt")
        model = tf.keras.models.load_model("Model/Maroon5_Model/Maroon5.h5")
    elif (class_choice == "Michael Jackson"):
        max_sequence_len = construct_prereq("Model/MJ_Model/MJ.txt")
        model = tf.keras.models.load_model("Model/MJ_Model/MJ.h5")
    elif (class_choice == "Rihanna"):
        max_sequence_len = construct_prereq("Model/Rihanna_Model/Rihanna.txt")
        model = tf.keras.models.load_model("Model/Rihanna_Model/Rihanna.h5")
    elif (class_choice == "The Weekend"):
        max_sequence_len = construct_prereq("Model/The_Weekend_Model/The_Weekend.txt")
        model = tf.keras.models.load_model("Model/The_Weekend_Model/The_Weekend.h5")

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list],maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list,verbose=0)
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word
    return seed_text


