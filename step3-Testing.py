# Step 3: CNN Testing
# Samuel Berryman
# Multi-scale Design Lab
# University of British Columbia
# April, 2019

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import pickle
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import TensorBoard
from vis.visualization import visualize_saliency
from vis.utils import utils
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
import os
import time
import gc

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from scipy import interp

from itertools import cycle
import pandas as pd
from PIL import Image

# plot_confusion_matrix:
# y_true = numpy array of Actual labels
# y_pred = numpy array of predicted labels
# classes = numpy array of class names
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    #Adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #plt.savefig('TestingConfusionFinal.png')
    plt.show()
    return ax


def plot_tSNE(model,X,y,Real_labels ):
    #adapted from https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    #Compute a new model which is truncated at the output we desire (initialized with same weights)
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[27].output) #16 17 27
    #predice each sample in X, outputed at second to last layer
    intermediate_output = intermediate_layer_model.predict(X)

    #Convert output to np array
    result = np.array(intermediate_output)

    print('size of output: ', result.shape)
    #Formatting the result data shape
    resultdata = np.zeros((result.shape[0], result.shape[1]+1))
    resultdata[0:result.shape[0],0:result.shape[1]] = result
    resultdata[:,resultdata.shape[1]-1] = y
    print('result size', resultdata.shape)
    #Save the results
    #np.savetxt('resultdata.csv',resultdata,delimiter=",")
    #Setup t-SNE for iterations
    tsne = TSNE(n_components=2, verbose=1, perplexity=64,n_iter=2000)
    #Start the iterating process
    tsne_results = tsne.fit_transform(result)
    #Format data for exporting
    data = pd.DataFrame(index=range(y.shape[0]), columns=['x','y','label'])
    data['x'] = tsne_results[:,0]
    data['y'] = tsne_results[:,1]
    data['label'] = y
    #Save the data
    data.to_csv('D://SamImaging//Cell-Line Project//Results//U2OS_Removed_Class_tSNE.csv')

    #Grab x, y and label values for plotting
    xr = tsne_results[:,0]
    yr = tsne_results[:,1]
    label = y
    #Plot the figure
    fig = plt.figure()
    #Assign a colour to each class
    plt.scatter(xr, yr, c=label)
    #Add a colour bar as a legend
    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/(Real_labels.shape[0]))
    cb.set_ticks(loc)
    cb.set_ticklabels(Real_labels)
    plt.title('T-SNE')
    #Save the figure
    #plt.savefig('tsne3.png')
    plt.show()
    return

#Adopted from https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras
def display_activation(activations, col_size, row_size, act_index):
    print('display')
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='Greys')
            activation_index += 1
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()




def Get_Features(model,X,y):
    #adapted from https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    #Compute a new model which is truncated at the output we desire (initialized with same weights) 
    # Swap softmax with linear
    layer_idx = 2
    model.layers[layer_idx].activation = tf.keras.activations.linear
    model = utils.apply_modifications(model)
    #grads = visualize_cam(model, X[0], filter_indices=None)
    #print('it worked...')
    #input()
    #filter_idx = 0
    #grads = visualize_saliency(model, layer_idx, filter_indices=filter_idx, seed_input=X[1])
    # Plot with 'jet' colormap to visualize as a heatmap.
    #plt.imshow(grads, cmap='jet')   
    #layer_outputs = [layer.output for layer in model.layers]
    for i in [1,5,9,12]:
        print('Layer: ', i)
        activation_model = tf.keras.Model(inputs=model.input, outputs=model.layers[i].output)
        #predice each sample in X, outputed at second to last layer
        #for a in X:
        activations = [activation_model.predict(X[0].reshape(1,75,75,4))]
        #intermediate_output = intermediate_layer_model.predict(a.reshape(1,75,75,4))
        display_activation(activations,8,8,0)
    


# plot_ROC
# Compute ROC curve and ROC area for each class
# true_labels = np array of label truth for each sample
# confusionPredictions = np array of predicted values
# Real_labels = np array of class names
def plot_ROC(true_labels, confusionPredictions, Real_labels):
    #Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    #Setup data formats for storing
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Binarize the output
    y_binary = label_binarize(true_labels, classes=np.array(range(0,Real_labels.shape[0])))
    #Get the number of classes
    n_classes = y_binary.shape[1]
    print('Number of classes: ', n_classes)
    #Iterating over each class
    for i in range(n_classes):
        #Caluclate the roc for each class
        fpr[i], tpr[i], _ = roc_curve(y_binary[:,i], confusionPredictions[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    #Find the total across all classes
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    #Plotting the results
    plt.figure()
    lw = 2
    colors = cycle(['yellow', 'darkorange', 'darkgreen','darkred','magenta','olive','maroon'])
    
    classes = []
    for c in Real_labels:
        classes.append(c)
        y_axis = c + ' y-axis'
        classes.append(y_axis)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)    

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #plt.savefig('ROC.png')

    #Get axis handle for plot
    axis = plt.gca()
    longest = 0
    #Find how much data we need to save (excel formating)
    for i in range(1):
        series = axis.lines[i]
        new_longest = np.array(series.get_xdata()).shape[0]
        if new_longest > longest:
            longest = new_longest
        
    plot_data = np.zeros([longest,Real_labels.shape[0]*2])
    #Grab the x,y data from the plot
    for i in range(1):
        series = axis.lines[i]
        print('shape: ', np.array(series.get_xdata()).shape)
        x_data = np.array(series.get_xdata())
        y_data = np.array(series.get_ydata())
        plot_data[0:x_data.shape[0],i*2] = x_data
        plot_data[0:y_data.shape[0],i*2+1] = y_data
    plt.show()
    return

def load_data(path, datashape):
    #Load data
    pathlist = []
    labels = []
    #For every folder in the path find the contained pickle files
    print('0')
    for root, dirs, files in os.walk(path):
            for name in files:
                print(name)
                if name.endswith('.pickle'):
                    p = os.path.join(root, name)
                    pathlist.append(p)
                    #Save the containing folder as the class label
                    labels.append(os.path.basename(os.path.dirname(p)))
                    #Print out the label
                    print("label: " + os.path.basename(os.path.dirname(p)))
    data = []
    print('1')
    #Pack the data into a single array so it can be randomized together
    for path, label in zip(pathlist, labels):
        
        x = pickle.load(open(path,"rb"))
        size = len(x)
        #print(size)
        for image in range(size):
                im = [x[image].astype(float)]
                im = np.array(im)
                im = im.squeeze()
                if im.shape == datashape:
                        data.append([im, label])
    #Randomly shuffle the matched samples and labels
    random.shuffle(data)
    #Convert to np format
    data = np.array(data)
    #Return the data
    return data





#----------------------------main------------------------------#

#Here are the variables that may need adjusting
#--------------------------------------------------------------#
#Data path to pickle files
#path = 'D://SamImaging//10x//Clustering Database//New Cell Line'
path = 'D://SamImaging//Cell-Line Project//Testing//Database'
#model Path
model_path = "D://SamImaging//Cell-Line Project//Results//4-channel_model_8-classes.h5"
#Size of the images (will exclude any that are not of this size)
datashape = (75,75,4)
#--------------------------------------------------------------#
#Load the data
data = load_data(path, datashape)
print(data.shape)
#Load the model
model = tf.keras.models.load_model(model_path)
#Declare array to store data in
X = np.zeros((data.shape[0],datashape[0],datashape[1],datashape[2]))
#Prepare array for label storage
y = []
total_samples = 1
#Iterate over each sample to find label from stored data
for index in range(data.shape[0]):
        print('Current number of samples: ',total_samples)
        #grab the image from the data
        feature = np.array(data[index][0])
        #grab the label from the data
        label = data[index][1]
        #Append the image
        X[index,...] = feature
        #Append the label
        y.append(label)
        total_samples = total_samples + 1

print('Converting data type, this may take a while')
#Convert the data to a numpy array format
#This will be made more efficient in the future
X = np.array(X)
y = np.array(y)

#Encode the labels with numerical equivelents
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

#Grab the dictionary matching classes to numbers
labels = list(le.classes_)

#Read labels
pickle_out = pickle.load(open("D://SamImaging//Cell-Line Project//Results//labels.pickle","rb"))
print(pickle_out)


#Convert the labels to np format
Real_labels = np.array(labels)
print('Found labels: ', Real_labels)
#Convert the transformed labels to np format
y = np.array(y)

#Predict on witheld data samples
allPredictions = model.predict(X[...,0:4])
allPredictions = np.array(allPredictions)


predictions = np.argmax(allPredictions, axis=1)
true_labels = y

np.savetxt("D:\\SamImaging//Cell-Line Project/Results//ValClassPredictionsSaved.csv",allPredictions, delimiter =",")
np.savetxt("D:\\SamImaging//Cell-Line Project//Results//ValClassPredictionClasses.csv",true_labels, delimiter =",")

#Plot confusion matrix
plot_confusion_matrix(true_labels, predictions, classes=Real_labels,normalize=True)

#Plot ROC
#plot_ROC(true_labels, allPredictions, Real_labels)

#Plot t-sne
#plot_tSNE(model,X,y, Real_labels)

#Get Activations
#Get_Features(model,X,y)




