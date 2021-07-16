# Mahdiar Nekoui
# coding = utf-8
import os
import  subprocess

import cv2


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

extensions = ['.avi']
home_dir = '/home/mahdiar/Projects/pytorch-i3d/AQA_Dataset/GymVault_raw/'
out_dir = '/home/mahdiar/Projects/pytorch-i3d/AQA_Dataset/GymVault/'

for root, dirs, files in os.walk(home_dir):
    for filename in files:
        for i in extensions:
            str_wo_ext = filename.replace(i, '')
        os.mkdir(out_dir + str_wo_ext)
        vidcap = cv2.VideoCapture(home_dir + filename)
        success, image = vidcap.read()
        count = 1
        while success:
            dir = out_dir + str_wo_ext + '/' + "%d.jpg" % count
            cv2.imwrite(dir, image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
