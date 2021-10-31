import matplotlib
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import numpy as np
from tqdm import tqdm
import soundfile as sf
from pydub import AudioSegment



#set paths for each directory for the different genres(insert yours here)
folders = ["/Volumes/external/songs/folk"]

for fd in folders:
    print(f"working in dir {fd}")
    for file in tqdm(os.listdir(fd), position=0, leave=True):
        audio_path = os.path.abspath(fd+"/"+file)
        song = AudioSegment.from_wav(audio_path)
        #print(audio_path)
        fol_name = os.path.basename(fd)
        fourty_five = 45 * 1000
        song = song[:fourty_five]
        #create a new directory to hold each shortened file and add it after the first '"'
        song.export(f"/Volumes/external/song45/{fol_name}/{file}", format="wav")
        #insert the same directory after the first '"'
        audio_path_short = f"/Volumes/external/song45/{fol_name}/{file}"
        x, sr = librosa.load(audio_path_short, sr= 44100)
        plt.figure(figsize=(14,5))
        plt.title(file)
        librosa.display.waveshow(x, sr = 44100)
        #insert the directory you want to save in here
        plt.savefig(f"/Volumes/external/raw_wav/{fol_name}/{file}.png")

