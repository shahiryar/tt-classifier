import streamlit as st
import csv
import json
import requests
import os

RESOURCE_CLASSES_JSON_PATH = "resources/classes.json"

def query(payload):
	response = requests.post(st.session_state['API_URL'], headers=st.session_state['HEADER'], json=payload)
	return response.json()

@st.cache_data
def load_cache():
    with open(RESOURCE_CLASSES_JSON_PATH, 'r') as file:
        st.session_state['label_code'] =  json.load(file)
    st.session_state["add_new_class_active"] = False

    token = ACCESS_TOKEN = os.environ['HUGGINGFACE_HUB_ACCESS_CODE']
    st.session_state['API_URL'] = "https://api-inference.huggingface.co/models/shahiryar/tt_abstract_classifier"
    st.session_state['HEADER'] = {"Authorization": f"Bearer {token}"}
    _ = query("Test text to start the model")

classifier = query

def save_log():
    log = f"{abstract}\t{label}\t{score}\n"
    #abstract_input.text_area("Abstract of the Paper", value="")
    with open('dataset/log.csv', 'a') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([abstract, label, score])
        #log_file.write(log)
    st.session_state.abstract = ''

def save_corrected_log():
    log = f"{abstract}\t{actual_label}\t{''}\n"
    #abstract_input.text_area("Abstract of the Paper", value="")
    with open('dataset/log.csv', 'a') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([abstract, actual_label.split(':')[0].strip(),''])
        #log_file.write(log)
    st.session_state.abstract = ''

def save_new_class():
    st.session_state.label_code[st.session_state.new_class_code] = st.session_state.new_class_label
    with open(RESOURCE_CLASSES_JSON_PATH, 'w') as file:
        json.dump(st.session_state.label_code, file)
    st.session_state.new_class_code = ""
    st.session_state.new_class_label = ""
    deactivate_add_new_class()

def activate_add_new_class():
    st.session_state["add_new_class_active"] = True
def deactivate_add_new_class():
    st.session_state["add_new_class_active"] = False

def clean_abstract(abstract):
    abstract = abstract.replace("\n", "")
    abstract = abstract.lower()
    return abstract


#================================================
#================================================

load_cache()

label_code = st.session_state['label_code']
label_options = [f"{code} : {label}" for code, label in st.session_state['label_code'].items()]

#================================================
#================================================

st.title("Welcome to your Personal Research Classifier")

abstract_input = st.empty()
abstract = abstract_input.text_area("Abstract of the Paper", height=200, key='abstract')

if abstract:
    abstract = clean_abstract(abstract)
    classified = classifier(abstract)
    actual_label = label = classified[0][0]["label"]
    score = classified[0][0]["score"]
    s = f"This abstract can be classified as :orange[{label} ({label_code[label]})] with confidence of :green[`{round(score*100,2)}`]"
    st.markdown(s)
    st.button(label="Prediciton is Correct", on_click=save_log)# save to log and move on
    st.markdown(":orange[If the Prediction is incorrect]")
    option = st.selectbox("Select the Actual Label",options=label_options)
    actual_label = option
    col1, col2 = st.columns(2)
    with col1:
        save_button = st.button("Save Label", on_click=save_corrected_log)
    with col2:
        st.button("Add new class", key='add_class_button', on_click=activate_add_new_class)

if st.session_state['add_new_class_active']:
    new_code = st.text_input("Enter Class Code", placeholder="TT1", key="new_class_code")
    new_label = st.text_input("Enter Class Label", placeholder="Explaining Theory", key="new_class_label")
    st.button("Save Changes", key='save_new_class_button' ,on_click=save_new_class)



## TODO
# [x] Option to input an abstract
# [x] input abstract be classified
# [x] show classification
# [x] option to correct the classification
# [x] new classification be stored in a log file
# if the log file reaches a certain level retrain the model
# Give option to add new classes
# if new class is added retrain the whole model

## TODO
# Get classes
# Engineer prompt
# integrate API


#TAKE A pdf and summarise it
# user the  summary to classify the document
