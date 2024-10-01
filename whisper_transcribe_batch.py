import whisper
import os
from pymongo import MongoClient
from tqdm import tqdm

mongo_client = MongoClient('mongodb://ia_admin:CureMD123@172.16.101.189:27017/?authSource=admin&readPreference=primary&directConnection=true&ssl=false')
asr_predictions_db= mongo_client.asr_predictions
whisper_predictions = asr_predictions_db.whisper_predictions_en
model = whisper.load_model("base.en") #whisper.load_model("base")

AUDIO_TRANSCRIPT_DIR = "/home/cmdadmin/Data Ambient Intelligence/Auto Transcribed Files"

for dir_name in tqdm(os.listdir(AUDIO_TRANSCRIPT_DIR)):

    dir_path = os.path.join(AUDIO_TRANSCRIPT_DIR, dir_name)
    files = os.listdir(dir_path)
    for _file in files:
        if '.wav' in _file:
            audio_path = os.path.join(dir_path, _file)
            count = whisper_predictions.count_documents( {"Filepath": audio_path} )
            if count == 0:
                result = model.transcribe(audio_path)
                document = { 
                	'Filename' : _file,
                	'Filepath' : audio_path,
                         'Predicted Text' : result["text"]
                        }
                whisper_predictions.insert_one(document)
        else:
            continue

