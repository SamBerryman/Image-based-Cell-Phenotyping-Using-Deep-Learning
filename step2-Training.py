# Step 2: CNN Training
# Samuel Berryman
# Multi-scale Design Lab
# University of British Columbia
# March, 2019

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Softmax, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import os
import time
import gc
import keras
import numpy as np
import matplotlib.pyplot as plt

from vis.visualization import visualize_saliency
from vis.utils import utils

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from scipy import interp

from itertools import cycle
import pandas as pd
from PIL import Image
from scipy import ndimage, misc

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
    #plt.savefig('confusionFinal.png')
    plt.show()
    return ax


def plot_tSNE(model,X,y,Real_labels ):
    #adapted from https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    #Compute a new model which is truncated at the output we desire (initialized with same weights)
    layer_name = 'activation_5'
    #Grab the second to last layer called activation_5, the name was found using Tensorboard graph
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
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
    np.savetxt('resultdata.csv',resultdata,delimiter=",")
    #Setup t-SNE for iterations
    tsne = TSNE(n_components=2, verbose=1, perplexity=64,n_iter=1000)
    #Start the iterating process
    tsne_results = tsne.fit_transform(result)
    #Format data for exporting
    data = pd.DataFrame(index=range(y.shape[0]), columns=['x','y','label'])
    data['x'] = tsne_results[:,0]
    data['y'] = tsne_results[:,1]
    data['label'] = y
    #Save the data
    #data.to_csv('TsnePlotData.csv')

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
    #plt.savefig('tsne3Final.png')
    plt.show()
    return



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
    plt.savefig('ROC.png')

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
    #Save the results
    #np.savetxt('3-ChannelROCDataFinal.csv', plot_data, delimiter =",")
    #plt.show()
    return plot_data

def training(X,y,channels,run,testing_samples, valX, valy):
    ##-----Normal Training with single validation -----------------
    #Name the model, use time to avoid overwriting previous run
    #Name = "7x7_Cell-Classification-CNN-{}".format(int(time.time()))
    Name = "Cell-Classification-CNN-{}".format(int(time.time()))
    Name = run + '-' + Name
    #Log directory to save the log data to
    tensorboard = TensorBoard(log_dir='Validationlogs/{}'.format(Name))

    #Input
    image_input = tf.keras.Input(shape=(75, 75, channels))

    #Conv layer 1
    conv_1 = tf.keras.layers.Conv2D(256, (7, 7))(image_input)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Activation('relu')(conv_1)

    maxpool_1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_1)
    
    #Conv layer 2
    conv_2 = tf.keras.layers.Conv2D(128, (5, 5))(maxpool_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Activation('relu')(conv_2)

    maxpool_2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_2)
    
    #Conv layer 3
    conv_3 = tf.keras.layers.Conv2D(64, (3, 3))(maxpool_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Activation('relu')(conv_3)
    
    #Conv layer 4
    conv_4 = tf.keras.layers.Conv2D(64, (3, 3))(conv_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Activation('relu')(conv_4)

    maxpool = tf.keras.layers.MaxPooling2D((2, 2))(conv_4)
    
    #Flatten layer
    flatten = tf.keras.layers.Flatten()(maxpool)

    #dense layer 1
    d_1 = tf.keras.layers.Dense(128)(flatten)
    d_1 = tf.keras.layers.BatchNormalization()(d_1)
    d_1 = tf.keras.layers.Activation('relu')(d_1)
    d_1 = tf.keras.layers.Dropout(rate=0.2)(d_1)
    
    #dense layer 2
    d_2 = tf.keras.layers.Dense(128)(d_1)
    d_2 = tf.keras.layers.BatchNormalization()(d_2)
    d_2 = tf.keras.layers.Activation('relu')(d_2)
    d_2 = tf.keras.layers.Dropout(rate=0.2)(d_2)
    
    #dense layer 3
    d_3 = tf.keras.layers.Dense(128)(d_2)
    d_3 = tf.keras.layers.BatchNormalization()(d_3)
    d_3 = tf.keras.layers.Activation('relu')(d_3)
    d_3 = tf.keras.layers.Dropout(rate=0.2)(d_3)

    out = tf.keras.layers.Dense(7, activation='softmax')(d_3)
    model = tf.keras.Model(image_input, out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer ="adam", metrics=['accuracy'])
    
    #Initialize all variables
    initial = tf.global_variables_initializer()
    
    #Train the model on the training data (minus the first X samples, which are witheld for validation)
    model.fit(X[testing_samples:],y[testing_samples:],batch_size=32, epochs = 25, validation_data = (valX,valy), callbacks = [tensorboard])
    return model

def cross_validation(X,y,channels,run):
        ##-----Training with cross validation -----------------
    #Name the model
    length = X.shape[0]
    #Create 5 folds of data for validation
    #Images
    X_1 = X[0::5]
    X_2 = X[1::5]
    X_3 = X[2::5]
    X_4 = X[3::5]
    X_5 = X[4::5]
    #Labels
    y_1 = y[0::5]
    y_2 = y[1::5]
    y_3 = y[2::5]
    y_4 = y[3::5]
    y_5 = y[4::5]
    validation_accuracy = np.zeros(5)
    testing_accuarcy = np.zeros(5)
    for i in range(5):
            #Compute a new name for log file
            Name = "Cross-Validated-Cell-Classification-CNN-{}".format(int(time.time()))
            Name = run + '-' + Name
            #Log directory to save log files
            tensorboard = TensorBoard(log_dir='crossvalidation/{}'.format(Name))
            #Itereate through different combinations for training
            if i == 0:
                    x_training = np.concatenate((X_1,X_2,X_3,X_4),axis=0)
                    y_training = np.concatenate((y_1,y_2,y_3,y_4),axis=0)
                    z = [X_5,y_5]
            elif i==1:
                    x_training = np.concatenate((X_2,X_3,X_4,X_5),axis=0)
                    y_training = np.concatenate((y_2,y_3,y_4,y_5),axis=0)
                    z = [X_1,y_1]
            elif i==2:
                    x_training = np.concatenate((X_1,X_3,X_4,X_5),axis=0)
                    y_training = np.concatenate((y_1,y_3,y_4,y_5),axis=0)
                    z = [X_2,y_2]
            elif i==3:
                    x_training = np.concatenate((X_2,X_1,X_4,X_5),axis=0)
                    y_training = np.concatenate((y_2,y_1,y_4,y_5),axis=0)
                    z = [X_3,y_3]
            else:
                    x_training = np.concatenate((X_2,X_3,X_1,X_5),axis=0)
                    y_training = np.concatenate((y_2,y_3,y_1,y_5),axis=0)
                    z = [X_4,y_4]

            ##-----Normal Training with single validation -----------------
                #Initiate a sequential model
            #Input
            image_input = tf.keras.Input(shape=(75, 75, channels))

            #Conv layer 1
            conv_1 = tf.keras.layers.Conv2D(256, (7, 7))(image_input)
            conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
            conv_1 = tf.keras.layers.Activation('relu')(conv_1)

            maxpool_1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_1)
            
            #Conv layer 2
            conv_2 = tf.keras.layers.Conv2D(128, (5, 5))(maxpool_1)
            conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
            conv_2 = tf.keras.layers.Activation('relu')(conv_2)

            maxpool_2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_2)
            
            #Conv layer 3
            conv_3 = tf.keras.layers.Conv2D(64, (3, 3))(maxpool_2)
            conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
            conv_3 = tf.keras.layers.Activation('relu')(conv_3)

            #maxpool_3 = tf.keras.layers.MaxPooling2D((1, 1))(conv_3)
            
            #Conv layer 4
            conv_4 = tf.keras.layers.Conv2D(64, (3, 3))(conv_3)
            conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
            conv_4 = tf.keras.layers.Activation('relu')(conv_4)

            maxpool = tf.keras.layers.MaxPooling2D((2, 2))(conv_4)
            
            #Flatten layer
            flatten = tf.keras.layers.Flatten()(maxpool)

            #dense layer 1
            d_1 = tf.keras.layers.Dense(128)(flatten)
            d_1 = tf.keras.layers.BatchNormalization()(d_1)
            d_1 = tf.keras.layers.Activation('relu')(d_1)
            d_1 = tf.keras.layers.Dropout(rate=0.2)(d_1)
            
            #dense layer 2
            d_2 = tf.keras.layers.Dense(128)(d_1)
            d_2 = tf.keras.layers.BatchNormalization()(d_2)
            d_2 = tf.keras.layers.Activation('relu')(d_2)
            d_2 = tf.keras.layers.Dropout(rate=0.2)(d_2)
            
            #dense layer 3
            d_3 = tf.keras.layers.Dense(128)(d_2)
            d_3 = tf.keras.layers.BatchNormalization()(d_3)
            d_3 = tf.keras.layers.Activation('relu')(d_3)
            d_3 = tf.keras.layers.Dropout(rate=0.2)(d_3)

            out = tf.keras.layers.Dense(10, activation='softmax')(d_3)
            model = tf.keras.Model(image_input, out)
            model.compile(loss='sparse_categorical_crossentropy', optimizer ="adam", metrics=['accuracy'])
            #Initialize all variables
            initial = tf.global_variables_initializer()
            
            #Train the model
            history = model.fit(x_training,y_training,batch_size=32, epochs = 25, validation_data= z, callbacks = [tensorboard])
            #Get the final validation acccuracy from the log file
            #Can also be exported from Tensorboard
            validation_accuracy[i] = int(history.history['val_acc'][-1])
            print(history.history['val_acc'])
            
            #Compute the testing accuracy
            #test_predictions = model.predict(valX)
            #print(test_predictions)
            #maxPrediction = np.argmax(test_predictions, axis=1)
            #print(maxPrediction)
            #count = 0
            #for p in range(len(maxPrediction)):
            #    if valy[p] == maxPrediction[p]:
            #        count = count + 1
            #testing_accuarcy[i] = count/len(maxPrediction)
            #print('Testing accuracy: ', testing_accuarcy[i])
            
            
    average_accuracy = 0
            
            #Compute the average validation accuracy
    for i in range(5):
        average_accuracy = average_accuracy + validation_accuracy [i]
    average_accuracy = average_accuracy/5
    #Report the validation accuracu
    print('Average Validation Accuracy of final epoch: ', average_accuracy)
    print('Testing Accuracies: ', testing_accuarcy)
        
    return model

def load_data(path, datashape):
    #Load data
    pathlist = []
    labels = []
    #For every folder in the path find the contained pickle files
    for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith('.pickle'):
                    p = os.path.join(root, name)
                    pathlist.append(p)
                    #Save the containing folder as the class label
                    labels.append(os.path.basename(os.path.dirname(p)))
                    #Print out the label
                    print("label: " + os.path.basename(os.path.dirname(p)))
    data = []
    #Pack the data into a single array so it can be randomized together
    for path, label in zip(pathlist, labels):
        
        x = pickle.load(open(path,"rb"))
        size = x.shape
        for image in range(size[0]):
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

##Rotational data augmentation
def data_augmentation(data, samples):
    new_data = []
    
    for i in range(samples-len(data)):
        new_image = data[random.randint(1,len(data)-1)]
        for r in range(random.randint(1,3)):
            new_image = np.rot90(new_image)
        new_data.append(new_image)
    return new_data

#Balance the classes so that they are of equal lengths
#dataset is a list of [image, label]
def format_data(dataset, augment, TotalSamples):
    classes = dict([])
    class_index = []
    data = []
    X = []
    y = []
    for x in dataset:
        # check if exists in unique_list or not 
        if x[1] not in list(classes.keys()):
            classes[x[1]] = 1
        else:
            classes[x[1]] = classes[x[1]] + 1
        class_index.append(x[1])
        data.append(x[0])
    print(classes.items())

    if augment == True:
        for item in list(classes.keys()):
            indicies = [i for i, x in enumerate(class_index) if x == item] 
            if len(indicies) >= TotalSamples:
                indicies = random.sample(indicies, k = TotalSamples)
                for i in indicies:
                    X.append(data[i])
                    y.append(class_index[i])
            else:
                aug = []
                for i in indicies:
                    X.append(data[i])
                    y.append(class_index[i])
                    aug.append(data[i])
                new_data = data_augmentation(aug,TotalSamples)
                for i in range(len(new_data)):
                    X.append(new_data[i])
                    y.append(class_index[indicies[0]])
    else:
         for item in list(classes.keys()):
             indicies = [i for i, x in enumerate(class_index) if x == item]
             for i in indicies:
                 X.append(data[i])
                 y.append(class_index[i])
    return X, y

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
    # Swap softmax with linear
    layer_idx = 2
    model.layers[layer_idx].activation = keras.activations.linear
    model = utils.apply_modifications(model)
    #grads = visualize_cam(model, X[0], filter_indices=None)
    #print('it worked...')
    #input()
    #filter_idx = 0
    grads = visualize_saliency(model, layer_idx, filter_indices=filter_idx, seed_input=X[1])
    # Plot with 'jet' colormap to visualize as a heatmap.
    plt.imshow(grads, cmap='jet')
    input()
    #layer_outputs = [layer.output for layer in model.layers]
    
    
## ----------------------------- main -----------------------------------

#Here are the variables that may need adjusting
#--------------------------------------------------------------#
#Data path to pickle files
#path = 'D://SamImaging//10x//ImageDatabaseNormalized'

path = 'D://SamImaging//Cell-Line Project//Training'
test_Path = 'D://SamImaging//Cell-Line Project//Testing'
#path = 'D://SamImaging//10x//TestingDatabase'
#Size of the images (will exclude any that are not of this size)
datashape = (75,75,4)
#Number of samples to withold for training
#Intentionaly not relative as I wanted a fixed number
testing_samples = 0
#--------------------------------------------------------------#

print('TensorFlow Version: tf.VERSION')
#Load the data
data = load_data(path, datashape)
#Load the testing data
testData = load_data(test_Path, datashape)
#Upsample the dataset so all classes are even
[X,y] = format_data(data, True, 10000)
#Downsample the testing data so all classes are even
[valX,valy] = format_data(testData, False, 500)
#Convert the data to a numpy array format
#This will be made more efficient in the future
X = np.array(X)
y = np.array(y)
valX = np.array(valX)
valy = np.array(valy)

#Encode the labels with numerical equivelents
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
valy = le.transform(valy)
#Grab the dictionary matching classes to numbers
labels = list(le.classes_)
#Save the dictionary
pickle_out = open("D://SamImaging//Cell-Line Project//Results//labels.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()
#Convert the labels to np format
Real_labels = np.array(labels)
print('Found labels: ', Real_labels)
#Convert the transformed labels to np format
y = np.array(y)
valy = np.array(valy)
class_totals = np.bincount(y)
for i in range(Real_labels.shape[0]):
    print(Real_labels[i],' Samples: ',class_totals[i])

#---------------------------------------------Training-----------------------------------------------#
#Standard training
model = training(X,y,4,'4-channel',testing_samples,valX,valy)
save_path = tf.keras.models.save_model(model, "D://SamImaging//Cell-Line Project//Results//4-channel_model_8-classes.h5", overwrite=True, include_optimizer=True)

