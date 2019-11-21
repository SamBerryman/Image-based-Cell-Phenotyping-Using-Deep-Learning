# Step 1: Database
# Samuel Berryman
# Multi-scale Design Lab
# University of British Columbia
# 2019

import numpy as np
import matplotlib.pyplot as plot
import cv2

from PIL import Image, ImageSequence
import math as math
import os, os.path
import sklearn.preprocessing as pre
from scipy import signal
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.morphology as morph
from skimage.morphology import disk
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from skimage import color
from skimage import io
import tensorflow as tf


import matplotlib.pyplot as plt
import matplotlib
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import random
import pickle

#Create a list of all of the images/labels in the training folder
def ListofImages(path):
    pathlist = []
    labels = []
    #Look in all subfolders
    for root, dirs, files in os.walk(path):
        for name in files:
            #If the file is a tif file
            if name.endswith('.tif'):
                p = os.path.join(root, name)
                pathlist.append(p)
                #Append the label from the folder it is in
                labels.append(os.path.basename(os.path.dirname(p)))
    return pathlist, labels


#Create an array of the training data with numerical labels
def create_training_data(pathlist, labels, IMG_SIZE):
    ##Make array of data
    training_data = []

    for i in range(len(pathlist)):
        try:
            img_array = cv2.imread(pathlist[i])
            new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
            training_data.append([new_array,labels[i]])
        except Exception as e:
            print('error in creating data set..continuing')
            pass
    return training_data

##Open TIFF image
def ReadTiff(path, offset, npages):
    imagestack = []
    temp = Image.open(path)
    for i in range(npages):
        image = temp
        image.seek(offset+i)
        image = np.array(image)
        image = image.astype(np.int32)* 255/image.astype(np.int32).max()
        image = Image.fromarray(image)
        #image.show()
        image = np.array(image)
        imagestack.append(image)
    return imagestack

#Thresholding an image
def ChanneltoBinary(img):
    #background = morph.opening(img,disk(15))
    #img = img - background
    #a = Image.fromarray(img)
    #a.show()
    try:
        glob_thresh = threshold_otsu(img)
    except:
        glob_thresh = 50
    binary_global = img
    binary_global[binary_global < glob_thresh] = 0
    binary_global[binary_global > glob_thresh] = 1
    binary_global = binary_global.astype(bool)
    #binary_global = ndimage.binary_fill_holes(binary_global)
    binary_global = morph.remove_small_objects(binary_global, 15)
    #b = Image.fromarray(binary_global)
    #b.show()
    return binary_global

#Crop a subsection of image
def cropSave(img,target_image,labeled,x,y,s,ObjectCounter, ch1_Otsu, ch2_Otsu, ch3_Otsu):
    #Check if we are near an edge
    width, height = img.size
    if width-x <(s/2):
        return 0, False, ObjectCounter
    if height-y <(s/2):
        return 0, False, ObjectCounter
    if x<s/2+1:
        return 0, False, ObjectCounter
    if y<s/2+1:
        return 0, False, ObjectCounter

    box = (x-s/2,y-s/2,x+1+s/2,y+1+s/2)
    c = img.crop(box)
    
    #Image gather point---------------------------------------------------------|>
    #image 1
    c_testing = np.array(c)
    c_testing[...,0] = (c_testing[...,0])
    c_testing[...,1] = (c_testing[...,1])
    c_testing[...,2] = (c_testing[...,2])

    #TEMP
    temp = c_testing
    low_values = temp[...,0] < ch1_Otsu
    temp[low_values,0] = 0
    low_values =temp[...,1] < ch2_Otsu
    temp[low_values,1] = 0
    low_values =temp[...,2] < ch3_Otsu
    temp[low_values,2] = 0
    temp = Image.fromarray(temp)
    #temp.show()
    #temp.save('D://Documents//Deep Learning Paper//Figures//Cells//SampleImages//LnCAP//' + dataset_name + '_' + str(ObjectCounter) + '_RGB.png')
    #input('pause')
    #END OF TEMP
    #c_testing.save('D:/Documents/Deep Learning Paper/Figures/Cells/' + dataset_name + '_Fluo.png')
    #--------------------------------
    ##Crop in numpy
    targetCrop = target_image[(int)(y-s/2):(int)(y+1+s/2), (int)(x-s/2):(int)(x+1+s/2),:]

    #targetCrop = target_image.crop(box)
    labeled = Image.fromarray(labeled)
    d = labeled.crop(box)
    d = np.array(d)
    
    ##Quality Checks
    
    #Check the number of nuclei
    Blue = np.array(c)[...,2]
    Blue = Blue > ch2_Otsu
    distance = ndimage.distance_transform_edt(Blue)
    local_maxi = peak_local_max(distance, labels=Blue,indices=False)
    markers = ndimage.label(local_maxi)[0]
    labeled = watershed(-distance, markers)
    regions = regionprops(labeled)
    if len(regions) >1:
        return 0, False, ObjectCounter


    #Check the green channel for live cells
    Green = np.array(c)[...,1]
    try:
        ch3_Otsu = threshold_otsu(Green)
    except:
        ch3_Otsu = 0
    Green = Green > ch3_Otsu
    Green[Green>0] = 255


    #Check the borders for pixels in contact
    clearedregionsGreen = clear_border(Green)
    if np.sum(clearedregionsGreen) != np.sum(Green):
        return 0, False, ObjectCounter

    Red = np.array(c)[...,0]
    Red = Red > ch1_Otsu
    clearedregionsRed = clear_border(Red)

    if np.sum(clearedregionsRed) != np.sum(Red):
        print("Too many Red objects")
        return 0, False, ObjectCounter
    
    clearedregionsBlue = clear_border(Blue)
    if np.sum(clearedregionsBlue) != np.sum(Blue):
        print("Too many Blue objects")
        return 0, False, ObjectCounter
    
    labeledregionsGreen, cleaned_objects = ndimage.label(clearedregionsGreen)
    regionsGreen = regionprops(labeledregionsGreen)
    if len(regionsGreen) >1 or len(regionsGreen) == 0:
        print("Too many Green objects")
        return 0, False, ObjectCounter
    
    if regionsGreen[0].area <50:
        print("Green area is too small")
        return 0, False, ObjectCounter
    #print("Green Area: " + str(regionsGreen[0].area))


    
    #Check in greyscale for number of objects
    grayscale = c.convert('L')
    grayscale = np.array(grayscale)
    glob_thresh = threshold_otsu(grayscale)
    binary_global = grayscale > glob_thresh

    ##watershed
    distance = ndimage.distance_transform_edt(binary_global)
    local_maxi = peak_local_max(distance, labels=binary_global,indices=False)
    markers = ndimage.label(local_maxi)[0]
    labeled = watershed(-distance, markers)
    
    regions = regionprops(labeled)
    if len(regions) >1:
        return 0, False, ObjectCounter
    
    labeled, nr_objects = ndimage.label(binary_global)
    cleared = clear_border(binary_global)
    labeled, cleaned_objects = ndimage.label(cleared)
    regions2 = regionprops(labeled)
    #Image gather point---------------------------------------------------------|>
    #image 2
    #print(d.shape)
    Brightfield = Image.fromarray((targetCrop[...,0]*255/targetCrop[...,0].max()).astype(np.uint8))
    #Brightfield.show()
    #Brightfield.save('D:/Documents/Deep Learning Paper/Figures/Cells/' + dataset_name + '_BF.png')

    #image 4
    im = Image.fromarray((labeled*125).astype(np.uint8))
    #im.show()
    #im.save('D:/Documents/Deep Learning Paper/Figures/Cells/' + dataset_name + '_mask.png')
    #input('stuff')
    #--------------------------------
    if len(regions) != cleaned_objects or nr_objects != cleaned_objects:
        return 0, False, ObjectCounter
    #print(nr_objects)
    if nr_objects >2:
        print("Too many objects")
        return 0, False, ObjectCounter
    if regions2[0].area <30:
        print("Object is too small")
        return 0, False, ObjectCounter
    
    #print("Area: " + str(regions2[0].area))
    #c = np.array(c)
    l = int(s/2+1)
    level = d[l,l]
    if level != 0:
        #c[d[:] != level] = 0
        #print(targetCrop.shape)


        #Try saving the FFT's
        #FFT = targetCrop
        #FFT[...,0] = np.fft.fft2(targetCrop[...,0])
        #FFT[...,1] = np.fft.fft2(targetCrop[...,1])
        #FFT[...,2] = np.fft.fft2(targetCrop[...,2])
        #FFT[...,3] = np.fft.fft2(targetCrop[...,3])
        return targetCrop, True, ObjectCounter + 1
        #targetCrop.save("D:\\SamImaging\\20x\\TrainingData\\PC3\\PC3_%d.jpg" % ObjectCounter)
    return 0, False, ObjectCounter


def CreateDatabase(pathlist, labels, dataset_name):
    training_data = []
    number_of_tiff_files = len(pathlist)
    tiff_file_counter = 1
    ObjectCounter = 0
    for path, image_label in zip(pathlist, labels):
        print("Tiff File Number: ", tiff_file_counter, "out of ", number_of_tiff_files, " files")
        print("Objects: ", ObjectCounter)
        #Check for number of images in tiff (Sets of 4)
        img = Image.open(path)
        pages = (int)(img.n_frames/4)
        training_data = []
        offset = 0
        for tiffs in range(pages):
            imagestack = ReadTiff(path,offset,4)
            offset = offset + 4
            shape = imagestack[0].shape
            stains = np.zeros((shape[0],shape[1],3),'uint8')
            stains[...,0] = imagestack[2]*255.0/(imagestack[2].max()) #RED
            stains[...,1] = imagestack[3]*255.0/(imagestack[3].max()) #GREEN
            stains[...,2] = imagestack[1]*255.0/(imagestack[1].max()) #BLUE
            stains.astype(np.uint8)
            #Image.fromarray(stains).show()
        
            target_image = np.zeros((shape[0],shape[1],4),'float')
            target_image[...,0] = imagestack[0]*255.0/(imagestack[0].max()) #Brightfield
            #Image.fromarray(target_image[...,0]).show()
            target_image[...,1] = imagestack[1]*255.0/(imagestack[1].max()) #Blue
            target_image[...,2] = imagestack[2]*255.0/(imagestack[2].max()) #RED
            target_image[...,3] = imagestack[3]*255.0/(imagestack[3].max()) #Green
            image = Image.fromarray(stains)

            ch1_Otsu = threshold_otsu(target_image[...,1])
            ch2_Otsu = threshold_otsu(target_image[...,2])
            ch3_Otsu = threshold_otsu(target_image[...,3])

            low_values = stains[...,0] < ch2_Otsu
            stains[low_values,0] = 0
            low_values = stains[...,1] < ch3_Otsu
            stains[low_values,1] = 0
            low_values = stains[...,2] < ch1_Otsu
            stains[low_values,2] = 0

            target = imagestack[1]*255.0/(imagestack[1].max())
            binary = ChanneltoBinary(target)
            binary = clear_border(binary)
        
        
            binary = np.uint8(binary*255)
            binary = Image.fromarray(binary)
            #print('4')
            labeled, nr_objects = ndimage.label(binary)
            b = list(range(1,nr_objects))
            centers = ndimage.measurements.center_of_mass(labeled, labeled, b)
            for i in range(1,nr_objects -1):
                #try:
                flag = False
                good_image, flag, ObjectCounter = cropSave(image,target_image,labeled,centers[i][1],centers[i][0],IMG_SIZE-1,ObjectCounter, ch1_Otsu, ch2_Otsu, ch3_Otsu)
                #except:
                #    pass
                if(flag == True):
                    training_data.append([good_image])
                    saveimg = np.zeros((75,75,3),'uint8')
                    saveimg[...,0]=good_image[...,3]
                    saveimg[...,1]=good_image[...,2]
                    saveimg[...,2]=good_image[...,1]
                    saveimg = Image.fromarray(saveimg)
                    saveimg.save(DATADIR + '//' + dataset_name + "//Images//" + str(ObjectCounter+1) + '.png')
            t = np.uint8(labeled)
            t = Image.fromarray(t)
        if ObjectCounter >1:
            random.shuffle(training_data)
            X = []
            X = np.array(training_data)
            #SAVE THE DATABASE for use later on
            X = tf.keras.utils.normalize(X)
            location = save_folder + "//" + dataset_name + '//' + dataset_name + "_" + str(tiff_file_counter) + ".pickle"
            pickle_out = open(location,"wb")
            pickle.dump(X, pickle_out)
            pickle_out.close()
        tiff_file_counter = tiff_file_counter + 1
            #t.show()
    return ObjectCounter

    
##-----------------MAIN---------------------

#-----------Input Variables-----------------
#Training image size
IMG_SIZE = 75
#Dataset name aka MCF7 (will not be used as label)
dataset_name = 'PC3'
#Data Location
DATADIR = "D://SamImaging//Cell-Line Project//Training"
#Database Directory
save_folder = "D://SamImaging//Cell-Line Project//Training//Database"
#--------End of Input Variables-------------

#Look in all subfolders
#for root, dirs, files in os.walk(DATADIR):
classes = os.listdir(DATADIR)
print('Found Classes: ' , classes)
    #Create a list of all of the images/labels in the training folder
objects = []
for cell_line in classes:
    print('Working on: ' , cell_line)
    pathlist, labels = ListofImages(DATADIR + '//' + cell_line)
    #Multi-thread in future iteration
    cells = CreateDatabase(pathlist, labels, cell_line)
    objects.append(cell_line + ': ' + str(cells))
print('FINISHED')
print(objects)



























