from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("home.html")

@app.route("/Predict_Sentiment",methods=["POST","GET"])
def get_pred():
	max_words = 20000
	max_len = 200

	model_best = tf.keras.models.load_model("Main/model_best.hdf5")
	sentiment = ["Negative","Positive"]

	with open("Main/tockenizer.pickle","rb") as handle:
		tokenizer = pickle.load(handle)

	if request.method == "POST":
		text = request.form['sentiment']
		sequence = tokenizer.texts_to_sequences([text])
		test = pad_sequences(sequence,maxlen=max_len)
		predictions_data = model_best.predict(test)
		score_data = predictions_data[0]
		res = sentiment[np.around(model_best(test),decimals=0).argmax(axis=1)[0]]
		acc = " ({:.2f} % of accuracy)".format(100*np.max(score_data))
	#return redirect(url_for('home', result='baba'))
	return render_template('home.html',text=text,result=res+acc)

if __name__ == '__main__':
    app.run(debug= True)
