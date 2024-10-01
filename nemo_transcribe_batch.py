import os
import json
import nemo.collections.asr as asr  
from tqdm import tqdm
from pymongo import MongoClient

mongo_client = MongoClient('mongodb://ia_admin:CureMD123@172.16.101.189:27017/?authSource=admin&readPreference=primary&directConnection=true&ssl=false')
asr_predictions_db= mongo_client.asr_predictions
nemo_cs_predictions = asr_predictions_db.nemo_pre_citrinet_preds


#asr_model_path = "/home/cmdadmin/Datalake Pusher/models/finetuned_model_70_citrinet.nemo"
#asr_model = asr.models.EncDecCTCModelBPE.restore_from(asr_model_path)
asr_model = asr.models.ASRModel.from_pretrained(model_name='stt_en_citrinet_1024')
asr_model.eval()


AUDIO_TRANSCRIPT_DIR = "/home/cmdadmin/Data Ambient Intelligence/Auto Transcribed Files"

for dir_name in tqdm(os.listdir(AUDIO_TRANSCRIPT_DIR)):

    dir_path = os.path.join(AUDIO_TRANSCRIPT_DIR, dir_name)
    files = os.listdir(dir_path)
    for _file in files:
        if '.wav' in _file:
            audio_path = os.path.join(dir_path, _file)
            count = nemo_cs_predictions.count_documents( {"Filepath": audio_path} )
            if count == 0:
                txt = asr_model.transcribe([audio_path])
                document = { 
                	'Filename' : _file,
                	'Filepath' : audio_path,
                         'Predicted Text' : txt
                        }
                nemo_cs_predictions.insert_one(document)
        else:
            continue



