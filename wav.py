import matplotlib
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os



#set paths for each directory (insert yours here)
folders = ["/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs/country",
           "/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs/hiphop",
           "/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs/pop"]

for fd in folders:
    for file in os.listdir(fd):
        audio_path = os.path.abspath(fd+"/"+file)
        #print(audio_path)
        fol_name = os.path.basename(fd)
        x, sr = librosa.load(audio_path)
        plt.figure(figsize=(14,5))
        plt.title(file)
        librosa.display.waveshow(x, sr = sr)
        #insert the directory you want to save in here
        plt.savefig(f"/{fol_name}/{file}.png")

