import pickle

import numpy as np
import logging
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM


def get_top_prediction_with_proba(model,X_test,k=1):
    
    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:,-k:]
    
    # GET CATEGORY OF PREDICTIONS
    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
    
    preds=[ item[::-1] for item in preds]
    
    return [preds[0][0], probs[0][0]]
   

model_path="loadedmodel/model_lr.pkl"
transformer_path="loadedmodel/transformer_lr.pkl"

loaded_model = pickle.load(open(model_path, 'rb'))
loaded_transformer = pickle.load(open(transformer_path, 'rb'))


# test_features=loaded_transformer.transform(["INPUT TEXT"])
# get_top_prediction_with_proba(loaded_model,test_features,1)

def detect_hate_speech_ml(input_text):
    test_features=loaded_transformer.transform([input_text])
    return 1-get_top_prediction_with_proba(loaded_model,test_features,1)[1]

# model_path = "loadedmodel/hateBERT"
# # tokenizer = AutoTokenizer.from_pretrained(model_path)
# # model = AutoModelForSequenceClassification.from_pretrained(model_path)

# tokenizer = AutoTokenizer.from_pretrained("raphaesq/autotrain-filipino-hate-speech-roberta-tagalog-base-66889136741")

# model = AutoModelForSequenceClassification.from_pretrained("raphaesq/autotrain-filipino-hate-speech-roberta-tagalog-base-66889136741")

# def detect_hate_speech_dl(input_text):
#     inputs = tokenizer.encode_plus(
#         input_text,
#         add_special_tokens=True,
#         return_tensors="pt",
#         padding="longest",
#         truncation=True
#     )
#     outputs = model(**inputs)
#     logits = outputs.logits

#     probabilities = torch.softmax(logits, dim=1)
#     confidence_score = probabilities[0][0].item()

#     # print(probabilities[0][0].item())
    
#     return 1 - confidence_score
    


