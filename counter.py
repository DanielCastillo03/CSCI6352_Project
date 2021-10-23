#counts how many files are in the current directory

import os
size = 0
#copy paths of each folder into here
folders = ["/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs/country",
           "/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs/hiphop",
           "/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs/pop",
           "/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/raw_wav/country",
           "/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/raw_wav/hiphop",
           "/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/raw_wav/pop"]
for fd in folders:
    base = os.path.basename(fd)
    print(base, len(os.listdir(fd)))



def get_size(start_path = '/Volumes/external/School/Fall_2021/CSCI_6352/Projects/Project/songs'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)


    return total_size / 1e+9

print(get_size(), 'Gbs')