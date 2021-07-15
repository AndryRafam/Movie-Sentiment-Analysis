import tensorflow as tf
import numpy as np
import pandas as pd
import re
import nltk
import string
import random

random.seed(0)
np.random.seed(0)
tf.random.set_seed(42)
tf.random.set_seed(42)
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from nltk.tokenize.treebank import TreebankWordDetokenizer
print("Done")

#nltk.download("stopwords")
#nltk.download("wordnet")
# Directory to the dataset. (User can change it at will according their own download directory.)
directory = "imdb.csv"

df = pd.read_csv(directory)
#stop_words = set(stopwords.words("english"))
#wordnet = WordNetLemmatizer()

def text_preproc(x):
	x = x.lower()
	#x = " ".join([word for word in x.split(" ") if word not in stop_words])
	x = x.encode("ascii", "ignore").decode()
	x = re.sub("https*\S+", " ", x)
	x = re.sub("@\S+", " ", x)
	x = re.sub("#\S+", " ", x)
	x = re.sub("\'\w+", "", x)
	x = re.sub("[%s]" % re.escape(string.punctuation), " ", x)
	x = re.sub("\w*\d+\w*", "", x)
	x = re.sub("\s{2,}", " ", x)
	return x
	
temp = []
data_to_list = df["review"].values.tolist()
for i in range(len(data_to_list)):
	temp.append(text_preproc(data_to_list[i]))

def tokenize(y):
	for x in y:
		yield(simple_preprocess(str(x)))

data_words = list(tokenize(temp))

def detokenize(txt):
	return TreebankWordDetokenizer().detokenize(txt)
	
final_data = []
for i in range(len(data_words)):
	final_data.append(detokenize(data_words[i]))
print(final_data[:5])
final_data = np.array(final_data)

labels = np.array(df["sentiment"])
l = []
for i in range(len(labels)):
	if labels[i]=="negative":
		l.append(0)
	elif labels[i]=="positive":
		l.append(1)
l = np.array(l)
labels = tf.keras.utils.to_categorical(l,2,dtype="int32")
del l

print(len(labels))

import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

max_words = 20000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(final_data)
sequences = tokenizer.texts_to_sequences(final_data)
tweets = pad_sequences(sequences, maxlen=max_len)
with open("tockenizer.pickle","wb") as handle:
	pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)	
print(tweets)
print(labels)

x_train, x_test, y_train, y_test = train_test_split(tweets,labels,random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
print(len(x_train),len(x_val),len(x_test),len(y_train),len(y_val),len(y_test))

model = Sequential([
	layers.Embedding(max_words,128,input_length=max_len),
	layers.Bidirectional(layers.LSTM(64,return_sequences=True)),
	layers.Bidirectional(layers.LSTM(64)),
	layers.Dense(2,activation="softmax"),
])
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
checkpoint = ModelCheckpoint("model_bilstm.hdf5", monitor="val_accuracy", verbose=1, save_best_only=True, save_weights_only=False)
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_val,y_val), verbose=2, callbacks=[checkpoint])

model_bilstm = tf.keras.models.load_model("model_bilstm.hdf5")
test_loss, test_acc, = model_bilstm.evaluate(x_test, y_test)
print("Test accuracy: {:.2f} %".format(100*test_acc))
print("Test loss: {:.2f} %".format(100*test_loss))
