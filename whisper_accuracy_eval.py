import os
import json
import string
from pymongo import MongoClient
from tqdm import tqdm

mongo_client = MongoClient('mongodb://ia_admin:CureMD123@172.16.101.189:27017/?authSource=admin&readPreference=primary&directConnection=true&ssl=false')
asr_predictions_db= mongo_client.asr_predictions
whisper_predictions = asr_predictions_db.whisper_predictions_en

dir_path = "/home/cmdadmin/Data Ambient Intelligence/scripts" 
labeled_text_file = "text_sentiment_tasks.json"

labeled_text_filepath = os.path.join(dir_path , labeled_text_file)

labeled_text_dict = None
with open(labeled_text_filepath, 'r') as f:
    txt = f.read()
    labeled_text_dict = json.loads(txt)
    f.close()

total_error_tokens = 0
total_tokens = 0
for sample in tqdm(labeled_text_dict):
    id = sample["id"]
    labeled_txt = sample["data"]["text"]
    filename = sample["data"]["call_id"].replace(".txt",".wav")

    document = whisper_predictions.find_one( { "Filename" : filename} )
    if document is not None:
        #converting to lower chars because label text is in lower chars
        #plus label text does not contain punctuation so we remove that too
        predicted_text = document["Predicted Text"].lower()
        predicted_text = predicted_text.translate(str.maketrans('','',string.punctuation))
        label_txt_tokens = labeled_txt.split()
        pred_txt_tokens = predicted_text.split()
        
        error_tokens = 0
        tokens = 0
        for token in label_txt_tokens:
            tokens += 1
            if token not in pred_txt_tokens:
                error_tokens += 1

        for token in pred_txt_tokens:
            tokens += 1
            if token not in label_txt_tokens:
                error_tokens += 1

    total_error_tokens += error_tokens
    total_tokens += tokens


print("WER : ", (total_error_tokens/total_tokens)*100)
