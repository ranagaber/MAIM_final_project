from tkinter import *
import subprocess
import sys
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification ,  AutoTokenizer, AutoModelForSequenceClassification
import torch
import kagglehub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer_s = AutoTokenizer.from_pretrained(model_name)
model_s = AutoModelForSequenceClassification.from_pretrained(model_name)
path = kagglehub.model_download("ranagaber1111/bert-finetuned-faketrue/tensorFlow2/default")
model_path = path
tokenizer = BertTokenizer.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(model_path)

def get_fake_or_real(news):
    inputs = tokenizer(news, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1)
    pred_class = tf.argmax(probs, axis=-1).numpy()[0]
    if pred_class == 1:
        return 'Fake'
    else:
        return 'Real'
def predict_sentiment(news):
    inputs = tokenizer_s(news, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_s(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]
def display():
    news = text.get("1.0", END).strip()
    result_label1.config(text = f"Is it fake or true?{get_fake_or_real(news)}")
    result_label2.config(text = f"Sentiment: {predict_sentiment(news)}")

#2 frames, 1 for the button and the text box and one for the output so it's clear enough
root = Tk()
frame1 = Frame(root)
frame1.pack()
frame2 = Frame(root)
frame2.pack(side = BOTTOM)
text = Text(frame1 , width=20 , height=5)
text.pack()
button = Button(frame2 , text = 'Enter' , fg = 'pink' , bg = 'black' , command=display)
button.pack(fill  =  X )
result_label1 = Label(frame2, text="", fg="black", bg="white")
result_label2 = Label(frame2 , text = "" ,fg="black", bg="white" )
result_label1.pack(pady=20)
result_label2.pack(pady=20)
root.mainloop()
