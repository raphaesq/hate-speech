import pickle
import numpy as np
import logging
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM


def get_top_prediction_with_proba(model,X_test,k=1):
    
    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:,-k:]
    
    # GET CATEGORY OF PREDICTIONS
    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
    
    preds=[ item[::-1] for item in preds]
    
    return preds[0][0]

# model_path="loadedmodel/model_svm.pkl"
# transformer_path="loadedmodel/transformer_svm.pkl"
# loaded_model = pickle.load(open(model_path, 'rb'))
# loaded_transformer = pickle.load(open(transformer_path, 'rb'))





def calculate_metrics_ml(benchmark_csv):
    # Load the benchmark dataset
    benchmark_df = pd.read_csv(benchmark_csv)

    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    model_path="loadedmodel/model_lr.pkl"
    transformer_path="loadedmodel/transformer_lr.pkl"
    loaded_model = pickle.load(open(model_path, 'rb'))
    loaded_transformer = pickle.load(open(transformer_path, 'rb'))

    # Iterate through the benchmark dataset
    for _, row in benchmark_df.iterrows():
        label = row['label']
        text = row['text']

        test_features=loaded_transformer.transform([text])

        # Get the predicted label using the detect_hate_speech_dl function
        predicted_label = get_top_prediction_with_proba(loaded_model,test_features,1)

        # Append true and predicted labels to the lists
        true_labels.append(label)
        predicted_labels.append(predicted_label)

        print(text)
        print('true_label', label)
        print('predicted_label', predicted_label)


    # Calculate accuracy and F1 score
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.show()
    # cm = confusion_matrix(true_labels, predicted_labels)

    print("Confusion Matrix:")
    for row in cm:
        print(row)

    return accuracy, precision, recall, f1

# tokenizer = AutoTokenizer.from_pretrained("raphaesq/autotrain-filipino-hate-speech-roberta-tagalog-base-66889136741")

# model = AutoModelForSequenceClassification.from_pretrained("raphaesq/autotrain-filipino-hate-speech-roberta-tagalog-base-66889136741")


# model_path="loadedmodel/model_svm.pkl"
# transformer_path="loadedmodel/transformer_svm.pkl"


accuracy, precision, recall, f1  = calculate_metrics_ml('ELECTION BENCHMARK1.csv')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# print(get_top_prediction_with_proba(loaded_model,test_features,1))

# def detect_hate_speech_ml(input_text):
#     test_features=loaded_transformer.transform([input_text])
#     return 1-get_top_prediction_with_proba(loaded_model,test_features,1)[1]

# model_path = "loadedmodel/hateBERT"
# # tokenizer = AutoTokenizer.from_pretrained(model_path)
# # model = AutoModelForSequenceClassification.from_pretrained(model_path)

# def detect_hate_speech_dl(model, tokenizer, input_text):
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
    
#     if probabilities[0][0].item() >= probabilities[0][1].item():
#         pred = 0
#     else:
#         pred = 1

#     return pred
    

# def calculate_metrics_dl(model, tokenizer, benchmark_csv):
#     # Load the benchmark dataset
#     benchmark_df = pd.read_csv(benchmark_csv)

#     # Initialize lists to store true labels and predicted labels
#     true_labels = []
#     predicted_labels = []

#     # Iterate through the benchmark dataset
#     for _, row in benchmark_df.iterrows():
#         label = row['label']
#         text = row['text']

#         # Get the predicted label using the detect_hate_speech_dl function
#         predicted_label = detect_hate_speech_dl(model, tokenizer, text)

#         # Append true and predicted labels to the lists
#         true_labels.append(label)
#         predicted_labels.append(predicted_label)

#         # print(text)
#         # print('true_label', label)
#         # print('predicted_label', predicted_label)


#     # Calculate accuracy and F1 score
#     accuracy = accuracy_score(true_labels, predicted_labels)
#     precision = precision_score(true_labels, predicted_labels)
#     recall = recall_score(true_labels, predicted_labels)
#     f1 = f1_score(true_labels, predicted_labels)
#     cm = confusion_matrix(true_labels, predicted_labels)

#     # Plot confusion matrix
#     # plt.figure(figsize=(8, 6))
#     # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
#     # plt.title('Confusion Matrix')
#     # plt.xlabel('Predicted Labels')
#     # plt.ylabel('True Labels')
#     # plt.show()
#     # cm = confusion_matrix(true_labels, predicted_labels)

#     print("Confusion Matrix:")
#     for row in cm:
#         print(row)

#     return accuracy, precision, recall, f1

# tokenizer = AutoTokenizer.from_pretrained("raphaesq/autotrain-filipino-hate-speech-roberta-tagalog-base-66889136741")

# model = AutoModelForSequenceClassification.from_pretrained("raphaesq/autotrain-filipino-hate-speech-roberta-tagalog-base-66889136741")


# accuracy, precision, recall, f1  = calculate_metrics_dl(model, tokenizer, 'ELECTION BENCHMARK1.csv')
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)