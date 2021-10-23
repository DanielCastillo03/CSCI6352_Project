#removes any files of a specified type

import os

#copy paths of each folder into here
folders = ["/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs/country",
           "/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs/hiphop",
           "/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs/pop"]


for fd in folders:
    for filename in os.listdir(fd):
        check = filename.endswith(".wav")
        if(check != True):
            os.remove(os.path.join(fd, filename))
