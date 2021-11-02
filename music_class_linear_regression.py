import warnings
import numpy as np
import os
import random
import pandas as pd
import sklearn
import scipy.io.wavfile as wav
from os import path

from matplotlib import pyplot as plt
from python_speech_features import mfcc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn.model_selection import train_test_split
from openpyxl import Workbook



#########################################################################################################################
#########################################################################################################################

# create a list to hold all the downloaded files ########################################################################

#########################################################################################################################
#########################################################################################################################

from tqdm import tqdm

warnings.filterwarnings("ignore")
main_directory = "/Volumes/external/song45"
songlist = []
# base = os.path.basename("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/country.xlsx")
# print(os.path.splitext(base)[0])


#########################################################################################################################
#########################################################################################################################

# count the total number of files in the directory ######################################################################

#########################################################################################################################
#########################################################################################################################

for path, subdirs, files in os.walk(main_directory):
    for file in files:
        if (file.endswith('.wav')):
            songlist.append(os.path.join(path,file))
num_files = len(songlist)
print("Total songs: ",num_files)

#########################################################################################################################
#########################################################################################################################

# use mfcc to extract features from each .wav file ######################################################################
# mfcc extracts 13 features from each file ##############################################################################

#########################################################################################################################
#########################################################################################################################

def feat_extract(file):
    features = []
    (sr, data) = wav.read(file)
    mffc_feature = mfcc(data, sr, winlen=0.045, nfft= 2160, appendEnergy= False)
    meanMatrix = mffc_feature.mean(0)

    for x in meanMatrix:
        features.append(x)
    return features


#########################################################################################################################
#########################################################################################################################

# extract features and class labels #####################################################################################

#########################################################################################################################
#########################################################################################################################

featureSet = []
i = 0
for folder in os.listdir(main_directory):
    # mac thing to skip over hidden files
    if not folder.startswith("."):
        print(f"working in directory {folder}")
        i+=1
    if not folder.startswith("."):
        for files in tqdm(os.listdir(main_directory+"/"+folder), desc= "songs", position= 0, leave= True):
            song = main_directory+"/"+folder+"/"+files
            features = feat_extract(song)
            j = 0
            for x in features:
                featureSet.append(x)
                j += 1
                if(j%13 == 0):
                    featureSet.append(i)

#########################################################################################################################
#########################################################################################################################

# create a data frame from featurelist for processing. Listed are all 13 extracted features from each song ##############

#########################################################################################################################
#########################################################################################################################


df = pd.DataFrame(columns=['m1','m2','m3','m4','m5','m6','m7',
                           'm8','m9','m10','m11','m12','m13','target'])

i = 1
n = []

for m in featureSet:
    n.append(m)
    if (i % 14 == 0):
        df = df.append({'m1':n[0],'m2':n[1],'m3':n[2],'m4':n[3],'m5':n[4],
                   'm6':n[5],'m7':n[6],'m8':n[7],'m9':n[8],'m10':n[9],
                   'm11':n[10],'m12':n[11],'m13':n[12],'target':n[13]},
                  ignore_index=True)
        n =[]
    i += 1

#########################################################################################################################
#########################################################################################################################

# seperate the features and class labels ################################################################################

#########################################################################################################################
#########################################################################################################################

x1=df[['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13']]

Y = df[['target']]


# save the data to a new excel file
df.to_excel("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/data.xlsx")


#########################################################################################################################
#########################################################################################################################

# plotting all the data #################################################################################################

#########################################################################################################################
#########################################################################################################################


#
# # try and plot all the things
# def PCA(exlfile):
#     from sklearn.decomposition import PCA
#     df = pd.read_excel(exlfile, index_col=0)
#
#     df_scaled  = pd.DataFrame()
#
#     for col in df.loc[:, "m1":"m13"]:
#         df_scaled[col] = (df[col] - df[col].mean() / df[col].std())
#
#     df_scaled
#
#     pca = PCA(n_components = len(df_scaled.columns))
#
#     pca_series = pca.fit_transform(df_scaled).T
#
#     df_pca = pd.DataFrame({"PC1": pca_series[0], "PC2": pca_series[1]})
#     base = os.path.basename(exlfile)
#     tag = os.path.splitext(base)[0]
#     df_pca.to_excel(f"/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/DF_PCA_{tag}.xlsx")
#
#     explained_variance = pca.explained_variance_ratio_
#
#     print(f' \n explained variance = {explained_variance}')
#
#     eigenvector_loadings = pca.singular_values_
#     print(f' \n eigenvector_loadings = {eigenvector_loadings}')
#
#
# PCA("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/country.xlsx")
# PCA("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/folk.xlsx")
# PCA("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/hiphop.xlsx")
# PCA("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/pop.xlsx")
#
# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111)
#
# ax.set_xlabel("First Principal Component", fontsize=15)
# ax.set_ylabel("Second Principal Component", fontsize=15)
# ax.set_title("Principal Components of both genres", fontsize=18)
#
# country_components = pd.read_excel("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/DF_PCA_country.xlsx", index_col=0)
# folk_components = pd.read_excel("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/DF_PCA_folk.xlsx", index_col=0)
# hiphop_components = pd.read_excel("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/DF_PCA_hiphop.xlsx", index_col=0)
# pop_components = pd.read_excel("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/dataframe/DF_PCA_pop.xlsx", index_col=0)
#
# ax = plt.scatter(x = country_components["PC1"], y= country_components["PC2"], label="country", color="darkcyan")
# ax = plt.scatter(x = folk_components["PC1"][:1861], y= folk_components["PC2"][:1861], label="folk", color="crimson")
# ax = plt.scatter(x = hiphop_components["PC1"][:1861], y= hiphop_components["PC2"][:1861], label="hiphop", color="black")
# ax = plt.scatter(x = pop_components["PC1"][:1861], y= pop_components["PC2"][:1861], label="pop", color="yellow")
#
# plt.xlim(-250000,400000)
#
# plt.grid(True)
# plt.legend(prop = {"size":18}, loc="upper right")
# plt.savefig("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/figures/PCAfig1.png")


#########################################################################################################################
#########################################################################################################################

# split into train and test sets ########################################################################################

#########################################################################################################################
#########################################################################################################################

X_train, X_test, y_train, y_test = train_test_split(x1, Y, test_size=0.35, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

pred_val = clf.predict(X_test)

sklearn.metrics.plot_confusion_matrix(clf, X_test, y_test)

print(classification_report(y_test, pred_val))
plt.savefig("/Volumes/external/School/Fall_2021/CSCI_6352/Projects/CSCI6352_Project/figures/fig1.png")


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
# The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
#
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
# The recall is intuitively the ability of the classifier to find all the positive samples.
#
# The f1 score (also called worstF-score or F-measure) is a measure of a model’s accuracy.
# It reaches its best value at 1 and  score at 0.
#
# The support is the number of occurrences of each class in y_test

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################







#########################################################################################################################
#########################################################################################################################

# make new predictions ##################################################################################################

#########################################################################################################################
#########################################################################################################################

ran_songs = ["/Volumes/external/songs/country/‘Letting You Go’ - Remy Garrison-5R5rodPfV44.wav",
         "/Volumes/external/songs/folk/James Blunt - Carry You Home (Video).wav",
         "/Volumes/external/songs/hiphop/21 Savage x Metro Boomin - Runnin (Official Music Video)-ZZ6VhTBcc-c.wav",
         "/Volumes/external/songs/pop/▷ NO COPYRIGHT MUSIC ➜ RYYZN - Miss You (Instrumental)✔️-X-ahaV5oFbI.wav"]
correct = 0
results=defaultdict(int)
i=1

for folder in os.listdir("/Volumes/external/song45"):
    if not folder.startswith("."):
        results[i]=folder
        i+=1


for song in ran_songs:
    audio_feat = feat_extract(song)
    pred = clf.predict([audio_feat])
    print(f"{os.path.basename(song)} is predicted to be {results[int(pred)]} and should be {os.path.basename(os.path.dirname(song))}")
    if(os.path.basename(os.path.dirname(song)) == results[int(pred)]):
        correct += 1

total_acc = correct / len(ran_songs)
print(f"The model predicted {correct}/{len(ran_songs)}: ({total_acc}%) songs correctly")


