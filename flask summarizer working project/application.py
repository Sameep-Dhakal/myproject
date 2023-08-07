from flask import Flask , render_template, request
import requests
import transformers

import torch

from model import summarizer

app=Flask(__name__)








@app.route('/',methods=["GET","POST"])
def index():
    if request.method=="POST":
        print("route called")
        src_text=request.form.get("original")
        # list_sent=src_text.replace('\n',' ')
        summarize=summarizer(src_text)
        # output=' '.join(summarize)
        # summarizer=pegasus(list_text)

        return render_template('index.html',src_text=summarize)

    return render_template('index.html')

 

if __name__=="__main__":
    app.run(debug=True)    

