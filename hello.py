from flask import Flask, render_template, request
import hatespeechdetectionmodule
import nermodule
import spacy


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ''
    confidence_score = None
    targets = None
    if request.method == 'POST':
        input_text = request.form['input_text']
    # Get the selected model option from the form data
    model_selection = request.form.get('model_selection', 'machine_learning')

    # Call your Python script to calculate the confidence score based on the selected model
    if model_selection == 'machine_learning':
        confidence_score = calculate_confidence_score_machine_learning(input_text)
        targets = get_targets(input_text)
    elif model_selection == 'deep_learning':
        confidence_score = calculate_confidence_score_deep_learning(input_text)
        targets = get_targets(input_text)
    return render_template('form.html', model_selection=model_selection, confidence_score=confidence_score, targets=targets, input_text=input_text)

def clean_string(s):
    # Transform into lowercase
    s = s.lower()

    # Remove non-alphanumeric characters
    s.replace(r'(@[A-Za-z0-9_]+)|([^A-Za-z0-9_ \t])|(\w+:\/\/\S+)', '')
    s.replace('"', '')
    s.replace("'", '')

    return s

def calculate_confidence_score_machine_learning(input_text):
    input_text = clean_string(input_text)
    probability_of_hate_speech = hatespeechdetectionmodule.detect_hate_speech_ml(input_text)
    return probability_of_hate_speech

def calculate_confidence_score_deep_learning(input_text):
    input_text = clean_string(input_text)
    probability_of_hate_speech = hatespeechdetectionmodule.detect_hate_speech_dl(input_text)
    return probability_of_hate_speech

def get_targets(input_text):
    nlp = spacy.load('en_core_web_sm')
    possible_subjects = nermodule.identify_entities(input_text, nlp)
    return possible_subjects

if __name__ == '__main__':
    app.run(debug=True)

