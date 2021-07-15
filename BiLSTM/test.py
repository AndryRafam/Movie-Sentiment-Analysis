import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

max_words = 20000
max_len = 200

model_best = tf.keras.models.load_model("model_bilstm.hdf5")
sentiment = ["Negative","Positive"]

with open("tockenizer.pickle","rb") as handle:
	tokenizer = pickle.load(handle)

string = input("\n❯❯ ")
print("\n")
sequence = tokenizer.texts_to_sequences([string])
test = pad_sequences(sequence,maxlen=max_len)
predictions_data1= model_best.predict(test)
score_data1 = predictions_data1[0]
print("\n")
print("{} ({:.2f} %)".format(sentiment[np.around(model_best(test),decimals=0).argmax(axis=1)[0]],100*np.max(score_data1)))
