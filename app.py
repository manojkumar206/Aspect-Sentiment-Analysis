#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import libraries
import pandas as pd 
import numpy 

import tensorflow as tf
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification

from flask import Flask, request, render_template


# In[4]:


app = Flask(__name__)


# In[5]:


loaded_model = TFDistilBertForSequenceClassification.from_pretrained("label_prediction_h5")


# In[ ]:


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# In[ ]:


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['post'])
def predict():
    test_sentence, aspect = request.form.values()
    test_sentence = str(test_sentence) + str(aspect )
    predict_input = tokenizer.encode(test_sentence, truncation=True,padding=True,return_tensors="tf")
    tf_output = loaded_model.predict(predict_input)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    labels = [0, 1, 2]
    label = tf.argmax(tf_prediction, axis=1)
    label = label.numpy()
    if labels[label[0]] == 0.0:
        sentiment = 'Negative'
    elif  labels[label[0]] ==1.0:
        sentiment = 'Neutral'
    else:
        sentiment = 'Positive'
      
    return render_template('index.html', prediction_text = 'Predicted sentiment is {}'.format(sentence))


if __name__ == '__main__':
    app.run(debug=True)
    

