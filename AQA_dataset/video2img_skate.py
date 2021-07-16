# coding = utf-8
import os
import numpy as np
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
home_dir = '/home/mahdiar/Projects/pytorch-i3d/AQA_Dataset/Figure_skating_raw/'
out_dir = '/home/mahdiar/Projects/pytorch-i3d/AQA_Dataset/Figure_skating/'
T = 5824


for root, dirs, files in os.walk(home_dir):
    for filename in files:
        for i in extensions:
            str_wo_ext = filename.replace(i, '')
        print("Working on:", filename)
        os.mkdir(out_dir + str_wo_ext)
        vidcap = cv2.VideoCapture(home_dir + filename)

        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        black_img = np.zeros((height,width,3))

        success, image = vidcap.read()
        count = 1

        # for i in range(T - int(length/2)):
        for i in range(T - length):
            dir = out_dir + str_wo_ext + '/' + "%d.jpg" % (i+1)
            image = black_img
            cv2.imwrite(dir, image)  # save frame as JPEG file


        while success:
            dir = out_dir + str_wo_ext + '/' + "%d.jpg" % (count + T - length)
            cv2.imwrite(dir, image)  # save frame as JPEG file

            # if count%2 == 0:
                # dir = out_dir + str_wo_ext + '/' + "%d.jpg" % (count/2 + T - int(length/2))
                # cv2.imwrite(dir, image)  # save frame as JPEG file

            success, image = vidcap.read()
            count += 1

