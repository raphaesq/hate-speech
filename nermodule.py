import pickle
import spacy
import numpy as np
import logging



def identify_entities(s, nlp):
    # NER - ['PERSON', 'NORP', 'ORG', 'GPE']
#     raw_text = "The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru. It operates under Department of Space which is directly overseen by the Prime Minister of India while Chairman of ISRO acts as executive of DOS as well."
    token = nlp(s)

    person_entities = [t.text for t in token.ents if t.label_ == 'PERSON']
    norp_entities = [t.text for t in token.ents if t.label_ == 'NORP']
    org_entities = [t.text for t in token.ents if t.label_ == 'ORG']
    gpe_entities = [t.text for t in token.ents if t.label_ == 'GPE']
    targets = person_entities+norp_entities+org_entities+gpe_entities

    return(targets)
