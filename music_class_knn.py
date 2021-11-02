from python_speech_features import mfcc
import scipy.io.wavfile as wav
from tqdm import tqdm
import os
import pickle
import random
import operator
from collections import defaultdict
import numpy as np

########################
########################
########################

##    dataset manip   ##

########################
########################
########################

directory = "/Volumes/external/songs"

#ONLY RUN THIS SECTION ONCE#############################################################################################
# f= open("my.dat" ,'wb')
# i=0
# for folder in os.listdir(directory):
#     #mac thing to skip over hidden files
#     if not folder.startswith("."):
#         print(f"Working for songs in directory: {folder}")
#         i+=1
#     #weird mac thing to filter out hidden files and directories. (may not need this on other OS)
#     if not folder.startswith("."):
#         for file in tqdm(os.listdir(directory+"/"+folder), desc= "songs", position= 0, leave= True):
#             (rate,sig) = wav.read(directory+"/"+foldirectory+"/"+folder+"/"+fileder+"/"+file)
#             mfcc_feat = mfcc(sig,rate, appendEnergy = False, nfft= 2400, winlen = 0.045)
#             covariance = np.cov(np.matrix.transpose(mfcc_feat))
#             mean_matrix = mfcc_feat.mean(0)
#             feature = (mean_matrix, covariance , i)
#             pickle.dump(feature , f)
# f.close()
########################################################################################################################


dataset = []
def loadDataset(filename , split , trSet , teSet):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split :
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])


trainingSet = []
testSet = []
loadDataset("my.dat" , 0.35, trainingSet, testSet)

trainingSet = np.array(trainingSet, dtype='object')
testSet = np.array(testSet, dtype='object')


########################
########################
########################

##    calculations    ##

########################
########################
########################

def distance(instance1, instance2, k ):

    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 ))
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance

def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1
        else:
            classVote[response]=1
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range (len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return 1.0*correct/len(testSet)


# ########################
# ########################
# ########################
#
# ## prediction section ##
#
# ########################
# ########################
# ########################

leng = len(testSet)
predictions = []

for x in range (leng):
    predictions.append(nearestClass(getNeighbors(trainingSet ,testSet[x] , 3)))
accuracy1 = getAccuracy(testSet , predictions)

print("Accuracy:", accuracy1)


results=defaultdict(int)
i=1

for folder in os.listdir("/Volumes/external/song45"):
    if not folder.startswith("."):
        results[i]=folder
        i+=1

songs = ["/Volumes/external/songs/country/‘Letting You Go’ - Remy Garrison-5R5rodPfV44.wav",
         "/Volumes/external/songs/folk/James Blunt - Carry You Home (Video).wav",
         "/Volumes/external/songs/hiphop/21 Savage x Metro Boomin - Runnin (Official Music Video)-ZZ6VhTBcc-c.wav",
         "/Volumes/external/songs/pop/▷ NO COPYRIGHT MUSIC ➜ RYYZN - Miss You (Instrumental)✔️-X-ahaV5oFbI.wav"]

correct = 0

for item in songs:

    path = os.path.abspath(item)
    (rate,sig)=wav.read(path)
    mfcc_feat= mfcc(sig, appendEnergy=False, nfft=2400, samplerate= 44100, winlen=0.045)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature=(mean_matrix,covariance,0)

    pred = nearestClass(getNeighbors(dataset,feature, 3))
    print(os.path.basename(item), "should be:", os.path.basename(os.path.dirname(item)), "and was predicted:", results[pred])

    if(os.path.basename(os.path.dirname(item)) == results[pred]):
        correct += 1

total_acc = correct / len(songs)
print(f"The model predicted {correct}/{len(songs)}: ({total_acc}%) songs correctly")


