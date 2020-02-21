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

##Open TIFF image
def ReadTiff(path, offset, npages):
    imagestack = []
    temp = Image.open(path)
    for i in range(npages):
        image = temp
        #Seek the start of the next 4 channel tiff image in the stack
        image.seek(offset+i)
        image = np.array(image)
        #Remap the pixel values
        image = image.astype(np.int32)* 255/image.astype(np.int32).max()
        #Ensure numpy format
        image = np.array(image)
        imagestack.append(image)
    return imagestack

#Thresholding an image
def ChanneltoBinary(img):
    #Try to get an otsu threshold
    try:
        glob_thresh = threshold_otsu(img)
    #Else set it to 50 (low number)
    except:
        glob_thresh = 50
    #Threshold the image converting it to 0 or 1
    binary_global = img
    binary_global[binary_global < glob_thresh] = 0
    binary_global[binary_global > glob_thresh] = 1
    binary_global = binary_global.astype(bool)
    #Remove any objects from the binray that are < 15 pixels to allow for noise
    binary_global = morph.remove_small_objects(binary_global, 15)
    return binary_global

#Crop a subsection of image
def cropSave(img,target_image,labeled,x,y,s,ObjectCounter, ch1_Otsu, ch2_Otsu, ch3_Otsu):
    
    #Get the width and height of the image
    width, height = img.size
    
    #Check if we are near an edge, Reject if cropspace overlaps with edge of image
    if width-x <(s/2):
        return 0, False, ObjectCounter
    if height-y <(s/2):
        return 0, False, ObjectCounter
    if x<s/2+1:
        return 0, False, ObjectCounter
    if y<s/2+1:
        return 0, False, ObjectCounter



    #Define the bounds of the cropped image, centered on the queried position
    box = (x-s/2,y-s/2,x+1+s/2,y+1+s/2)
    #Grab the cropped version of the fluorescent channels
    c = img.crop(box)
    
    #Convert to a numpy array for analysis
    c_testing = np.array(c)
    
    ##Get the target four channel image
    targetCrop = target_image[(int)(y-s/2):(int)(y+1+s/2), (int)(x-s/2):(int)(x+1+s/2),:]

    labeled = Image.fromarray(labeled)
    d = labeled.crop(box)
    d = np.array(d)


    
    ##------Pass/Fail Tests-----##

    #Test 1: Check the number of nuclei

    #grab the blue channel and determine the number of objects present using watershed
    Blue = np.array(c)[...,2]
    Blue = Blue > ch2_Otsu
    distance = ndimage.distance_transform_edt(Blue)
    local_maxi = peak_local_max(distance, labels=Blue,indices=False)
    markers = ndimage.label(local_maxi)[0]
    labeled = watershed(-distance, markers)
    regions = regionprops(labeled)
    
    #If more than one object is present, reject it
    if len(regions) >1:
        return 0, False, ObjectCounter


    #Test 2: Check if the cell body is touching the border of the image

    #Grab the green channel
    Green = np.array(c)[...,1]
    #Attempt to detrmine a local otsu threshold
    try:
        ch3_Otsu = threshold_otsu(Green)
    except:
        ch3_Otsu = 0
    #Threshold the image
    Green = Green > ch3_Otsu
    Green[Green>0] = 255

    #Check the borders for pixels in contact
    clearedregionsGreen = clear_border(Green)
    if np.sum(clearedregionsGreen) != np.sum(Green):
        return 0, False, ObjectCounter

    #Grab the actin channel
    Red = np.array(c)[...,0]
    Red = Red > ch1_Otsu
    clearedregionsRed = clear_border(Red)

    #Reject if object is in contact with the border
    if np.sum(clearedregionsRed) != np.sum(Red):
        print("Too many Red objects")
        return 0, False, ObjectCounter
    
    #Reject if object is in contact with the border
    clearedregionsBlue = clear_border(Blue)
    if np.sum(clearedregionsBlue) != np.sum(Blue):
        print("Too many Blue objects")
        return 0, False, ObjectCounter


    #Test 3: Check that only a single cell is present
    
    labeledregionsGreen, cleaned_objects = ndimage.label(clearedregionsGreen)
    regionsGreen = regionprops(labeledregionsGreen)
    if len(regionsGreen) >1 or len(regionsGreen) == 0:
        print("Too many Green objects")
        return 0, False, ObjectCounter


    #Test 4: Check that the cell is viable in the cytoplasm channel
    
    if regionsGreen[0].area <50:
        print("Green area is too small")
        return 0, False, ObjectCounter


    #Test 5: Check there is only one object present

    #Convert the 3 fluorescent channels to greyscale
    grayscale = c.convert('L')
    grayscale = np.array(grayscale)
    
    #Determine an otsu threshold
    glob_thresh = threshold_otsu(grayscale)

    #Threshold the image into a binary
    binary_global = grayscale > glob_thresh

    #Determine the bounds of objects in the image using watershed
    distance = ndimage.distance_transform_edt(binary_global)
    local_maxi = peak_local_max(distance, labels=binary_global,indices=False)
    markers = ndimage.label(local_maxi)[0]
    labeled = watershed(-distance, markers)
    regions = regionprops(labeled)

    #Reject if more than 1 object was found in the greyscale
    if len(regions) >1:
        return 0, False, ObjectCounter

    #Make sure the object is not touching the border of the image
    labeled, nr_objects = ndimage.label(binary_global)
    cleared = clear_border(binary_global)
    labeled, cleaned_objects = ndimage.label(cleared)
    regions2 = regionprops(labeled)

    im = Image.fromarray((labeled*125).astype(np.uint8))
    if len(regions) != cleaned_objects or nr_objects != cleaned_objects:
        return 0, False, ObjectCounter

    #Reject if there is more than the object and the background present
    if nr_objects >2:
        print("Too many objects")
        return 0, False, ObjectCounter


    #Test 6: Check that the object is large enough to be a cell
    
    if regions2[0].area <30:
        print("Object is too small")
        return 0, False, ObjectCounter

    #Test 7: Check that the center object is the cell and not the background
    l = int(s/2+1)
    level = d[l,l]
    if level != 0:
        return targetCrop, True, ObjectCounter + 1
    
    #Else return that it failed the tests
    return 0, False, ObjectCounter


def CreateDatabase(pathlist, labels, dataset_name):

    #Variable initialization
    training_data = []
    number_of_tiff_files = len(pathlist)
    tiff_file_counter = 1
    ObjectCounter = 0

    #For each of the images in the list of paths
    for path, image_label in zip(pathlist, labels):
        #Variable initialization
        training_data = []
        offset = 0
        #Display the current progress
        print("Tiff File Number: ", tiff_file_counter, "out of ", number_of_tiff_files, " files")
        #Print the current number of accepted cells
        print("Objects: ", ObjectCounter)
        #Load the current tiff image
        img = Image.open(path)
        #Check for number of images in tiff, should be a multiple of 4 (BF, N, A, C)
        pages = (int)(img.n_frames/4)

        #Read the tiff image stack in groups of 4 (BF, N, A, C)
        for tiffs in range(pages):
            #Get the current tiff image from the stack
            imagestack = ReadTiff(path,offset,4)
            #index the offset for the next iteration
            offset = offset + 4
            #Get the shape of the image
            shape = imagestack[0].shape
            
            #Initialize an image holder for analysis, seperate from the image we will save from
            stains = np.zeros((shape[0],shape[1],3),'uint8')
            #Channels are mapped to corresponding colours
            #Actin channel
            stains[...,0] = imagestack[2]*255.0/(imagestack[2].max())
            #Cytoplasm channel
            stains[...,1] = imagestack[3]*255.0/(imagestack[3].max())
            #Nuclear channel
            stains[...,2] = imagestack[1]*255.0/(imagestack[1].max())
            stains.astype(np.uint8)

            #Format the image data again, this one for saving from
            target_image = np.zeros((shape[0],shape[1],4),'float')
            #Brightfield
            target_image[...,0] = imagestack[0]*255.0/(imagestack[0].max()) 
            #Image.fromarray(target_image[...,0]).show()
            #Nuclear Channel
            target_image[...,1] = imagestack[1]*255.0/(imagestack[1].max())
            #Actin Channel
            target_image[...,2] = imagestack[2]*255.0/(imagestack[2].max())
            #Cytoplasm Channel
            target_image[...,3] = imagestack[3]*255.0/(imagestack[3].max())
            image = Image.fromarray(stains)

            #Get the otsu threshold value for the nuclear channel
            ch1_Otsu = threshold_otsu(target_image[...,1])
            #Get the otsu threshold value for the cytoplasm channel
            ch2_Otsu = threshold_otsu(target_image[...,2])
            #Get the otsu threshold value for the actin channel
            ch3_Otsu = threshold_otsu(target_image[...,3])

            #Threshold the analysis image, set all values < Otsu threshold (background noise) to zero
            low_values = stains[...,0] < ch2_Otsu
            stains[low_values,0] = 0
            low_values = stains[...,1] < ch3_Otsu
            stains[low_values,1] = 0
            low_values = stains[...,2] < ch1_Otsu
            stains[low_values,2] = 0

            #Remap the nuclear channel
            target = imagestack[1]*255.0/(imagestack[1].max())
            #Convert the nuclear channel to binary
            binary = ChanneltoBinary(target)
            #Get rid of any objects that touch the image boundry
            binary = clear_border(binary)
            #Scale binary image from 0-1 up to 0-255
            binary = np.uint8(binary*255)
            #Convert back to an image
            binary = Image.fromarray(binary)
            #Identify all of the seperate objects
            labeled, nr_objects = ndimage.label(binary)
            #Create a list of numerical sample labels
            b = list(range(1,nr_objects))
            #Get a list of all of the candidate nucelei positions
            centers = ndimage.measurements.center_of_mass(labeled, labeled, b)
            #Iterate through each of the individual samples
            for i in range(1,nr_objects -1):
                #initilize false flag, flag = true when sample is good
                flag = False
                #Analyze the image and return a cropped image, True and found object count if the image passes
                good_image, flag, ObjectCounter = cropSave(image,target_image,labeled,centers[i][1],centers[i][0],IMG_SIZE-1,ObjectCounter, ch1_Otsu, ch2_Otsu, ch3_Otsu)

                #If the image passed
                if(flag == True):
                    #Append the cropped image containing a single cell
                    training_data.append([good_image])

                    #Additionaly can save the fluorescent channels for troubleshooting
                    #saveimg = np.zeros((75,75,3),'uint8')
                    #saveimg[...,0]=good_image[...,3]
                    #saveimg[...,1]=good_image[...,2]
                    #saveimg[...,2]=good_image[...,1]
                    #saveimg = Image.fromarray(saveimg)
                    #saveimg.save(DATADIR + '//' + dataset_name + "//Images//" + str(ObjectCounter+1) + '.png')

        #Save the list of images if there are any    
        if ObjectCounter >1:
            #Randomly shuffle the list of data
            random.shuffle(training_data)
            X = []
            #Get data into numpy format for saving
            X = np.array(training_data)
            #Normalize the samples between 0.0 and 1.0
            X = tf.keras.utils.normalize(X)
            #Location to save the pickle dataset file
            location = save_folder + "//" + dataset_name + '//' + dataset_name + "_" + str(tiff_file_counter) + ".pickle"
            #Open the file location
            pickle_out = open(location,"wb")
            #Write the data into the file
            pickle.dump(X, pickle_out)
            #Close the file
            pickle_out.close()
        #Incriment the number of tiff_files analyzed
        tiff_file_counter = tiff_file_counter + 1

    #Return the numer of found cells
    return ObjectCounter




##-----------------MAIN---------------------##

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



objects = []

#Create a list of all of the images/labels in the training folder
for cell_line in classes:
    print('Working on: ' , cell_line)

    #Get a list of image locations for the current cell line
    pathlist, labels = ListofImages(DATADIR + '//' + cell_line)

    #Pass the current  set of images and label for segmentation
    cells = CreateDatabase(pathlist, labels, cell_line)

    #Append to the list the number of found objects
    objects.append(cell_line + ': ' + str(cells))
    
print('FINISHED')
print(objects)



























