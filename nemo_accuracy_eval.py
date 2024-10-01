import os
import json
from pymongo import MongoClient
from tqdm import tqdm
import nemo.collections.asr as asr 

mongo_client = MongoClient('mongodb://ia_admin:CureMD123@172.16.101.189:27017/?authSource=admin&readPreference=primary&directConnection=true&ssl=false')
asr_predictions_db= mongo_client.asr_predictions
nemo_cs_predictions = asr_predictions_db.nemo_pre_citrinet_preds


asr_model_path = "/home/cmdadmin/Datalake Pusher/models/finetuned_model_70_citrinet.nemo"
dir_path = "/home/cmdadmin/Data Ambient Intelligence/scripts" 
labeled_text_file = "text_sentiment_tasks.json"

#asr_model = asr.models.EncDecCTCModelBPE.restore_from(asr_model_path)
asr_model = asr.models.ASRModel.from_pretrained(model_name='stt_en_citrinet_1024')
asr_model.eval()

labeled_text_filepath = os.path.join(dir_path , labeled_text_file)

labeled_text_dict = None
with open(labeled_text_filepath, 'r') as f:
    txt = f.read()
    labeled_text_dict = json.loads(txt)
    f.close()

total_error_tokens = 0
total_tokens = 0
total_documents_processed = 0
for sample in tqdm(labeled_text_dict):
    id = sample["id"]
    labeled_txt = sample["data"]["text"]
    filename = sample["data"]["call_id"].replace(".txt",".wav")

    document = nemo_cs_predictions.find_one( { "Filename" : filename} )
    if document is not None:
        total_documents_processed += 1
        predicted_text = document["Predicted Text"]

        label_txt_tokens = labeled_txt.split()
        pred_txt_tokens = predicted_text[0].split()
        
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


print("WER : {} on {} documents".format( (total_error_tokens/total_tokens)*100 , total_documents_processed) )
