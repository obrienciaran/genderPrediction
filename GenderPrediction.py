import pandas as pd
import os
import cv2
import nibabel as nib
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

def openCSV(filename):
    df = pd.read_csv(filename + ".csv", sep=',', skipinitialspace=True)
    return (df)

# collect the patient .nii.gz file paths into a list
def imageFilePaths(path):
    filePaths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".gz")]:
            filePaths.append(os.path.join(dirpath, filename))
    return (filePaths)

# Creates a dictionary with key:value as 'patient file path':'patients images'
def patientImagesDict(patientImagePaths):
    patientImages = []
    for path in patientImagePaths:  # This is the path to each patients brain images
        if ".gz" in path.lower():
            img = nib.load(path)
            data = img.get_fdata()
            newData = cv2.resize(np.array(data), (128, 128))
            patientImages.append(newData)
    z = zip(patientImagePaths, patientImages)
    patients = dict(z)
    return (patients, patientImages)

# A function to take the ID from the patient file path 
def getPatientID(patientImagePaths):
    patientIDList = []
    for patientImagePath in patientImagePaths:        
        patientID = patientImagePath.rsplit("\\",1)         # split on the last backslash 
        patientID = patientID[1]                            # take only what is after backslash 
        patientID = patientID.replace(".","-").split("-")   # replace all . with - for the sake of consistency and split
        patientID = patientID[0]                            # we just want the first info, e.g. IXI001
        patientID = patientID.replace("IXI","")             # No need for the IXI 
        patientIDList.append(patientID)
    return(patientIDList)

# A function to remove the leading 0's from the ID numbers taken from the file path 
def removeLeadingZeros(patientID):
    # remove leading zeros from patient ID numbers
    patientIDNoLeadingZeros = []
    for i in patientID: 
        i = i.lstrip("0")
        patientIDNoLeadingZeros.append(int(i))
    return(patientIDNoLeadingZeros)

# A function to return how many slices is in the patients scan    
def numSlices(patientImages):
    numSlices = [] 
    for i in patientImages: 
        numSlices.append(i.shape[2])
    return(numSlices)

# normalise slices and add them to an array
def XVariable(patients):
    X = []
    # subj is the patient paths. subj_slice is the MR slices 
    for subj, subj_slice in patients.items():
        subj_slice = subj_slice.transpose(2, 0, 1)
        subj_slice_norm = [((imageArray - np.min(imageArray)) / np.ptp(imageArray)) for imageArray in subj_slice]  
        X.extend([s[ :, :, np.newaxis] for s in subj_slice_norm]) # As we are dealing with 1 slice per iteration, it is a 2d numpy array. We need to make a z axis
    
    X = np.stack(X, axis=0)   
    return(X)

'''
multiplication of two lists of numbers. List 1 is the gender (1 or 2), 
and list two is the number of slices for that patients scan.
We have a list of gender ID's in order. and also a list of 3d patient arreys in order. Iterate through them to get y
based on the number of slices in each patients 3d array  
'''

# generates 1 for male or 2 for female for each slice for each patient. 
# list of y variable will be as long as the total number of slices for all patients
def YVariable(genderList, numberOfSlices):    
    y = []
    for f, b in zip(genderList, numberOfSlices): 
        y.extend(str(f) * b)
    y = [int(i) for i in y] # y is currently in string format, make it into an int 
    return(y)

# At the moment, y is 1 for male and 2 for female. Change that to 0 and 1
def yBinary(yTemp):
    y = []
    for i in yTemp:
        i  = i - 1
        y.append(i)
    return(y)
    
# Create Keras CNN model
def createModel():
    '''
    Stack up Convolutional Layers, followed by Max Pooling layers.
    Also include Dropout to avoid overfitting.
    Finally, add a fully connected (Dense) layer followed by a softmax layer.
    '''

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    #for two label classification
    #model.add(Dense(2, activation='softmax'))
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #single class
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy'])
    return (model)
 
# Confusion Matrix
def confusionMatrix(model, X_val, y_val):
    # generate predictions
    predictions = model.predict(X_val)
    # round them and make into Pandas series
    rounded = [round(x[0]) for x in predictions]
    rounded = pd.Series(v for v in rounded)

    y_val = pd.Series(y_val)
    print('Confusion Matrix')
    print(pd.crosstab(y_val, rounded, rownames=['True'], colnames=['Predicted'], margins=True)) 

# Draw ROC and genrate AUC
def rocAuc(model, X_train, y_train, val_X, val_y):

    # get FPR and TPR for the predictions
    pred_y = model.predict(val_X).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(val_y, pred_y)

    # Get AUC
    auc_keras = auc(fpr_keras, tpr_keras)

    # Plot ROC
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    print("The area under the curve is ", auc_keras)
    return(pred_y)

# AUC Confidence Interval - Bootstrap 1000 AUC curve results. 
def bootstrapAUC(Y_pred,Y_true):   
    n_bootstraps = 1000
    #seed = 42  # control reproducibility
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(Y_pred) - 1, len(Y_pred))
        if len(np.unique(Y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(Y_true[indices], Y_pred[indices])
        bootstrapped_scores.append(score)
        print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))   
    return(bootstrapped_scores)

# Confidence Interval 
def confidenceInterval(bootstrapAUCScores):
    sorted_scores = np.array(bootstrapAUCScores)
    sorted_scores.sort()
    
    # Computing the lower and upper bound of the 95% confidence interval.
    # You can change the bounds percentiles to 0.05 and 0.95 to get
    # a 90% confidence interval instead.
    confidence_lower = sorted_scores[int(0.25 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))     

# Draw the loss curve
def loss(history):
    plt.plot(history.history['loss'], linewidth=2.0)
    plt.plot(history.history['val_acc'], linewidth=2.0)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'],
               loc='lower left',
               fontsize=14)
    plt.show()

# Draw the accuracy curve
def accCurve(history):
    plt.plot(history.history['acc'], linewidth=2.0)
    plt.plot(history.history['val_acc'], linewidth=2.0)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'],
               loc='lower left',
               fontsize=14)
    plt.show()

def main(path,pathMetadata):
    
    # =============================================================================
    # Open and process the CSV    
    # =============================================================================
        
    # Labels for each other the patients gender status. 0 = male, 1 = female
    metaData = openCSV(pathMetadata)
    labels_df = metaData.iloc[:,  [0] + [1]]
        
    # =============================================================================
    # Open and process the patient images     
    # =============================================================================
    
    # Collect the patients .nii.gz file paths into a list
    patientImagePaths = imageFilePaths(path)
    
    # Get a dictionary of patients and their images
    patients, patientImages = patientImagesDict(patientImagePaths)
    
    # Get patient ID numbers. Take from file path name
    patientID = getPatientID(patientImagePaths) 
 
    # Remove leading 0s in the digits to match the id in the .csv metadata
    patientIDNoLeadingZeros = removeLeadingZeros(patientID)

    # Filter the patient demographic information. We have 578 patients but demographic
    # information on 619. We need to keep the info on 578 and remove the rest. 

    metaData = metaData[metaData['IXI_ID'].isin(patientIDNoLeadingZeros)]
    
    # this will give a list of numbers. Each number is how many slices were taken for each patient 
    numberOfSlices = numSlices(patientImages)
        
    # this gives a list of genders we are working with (1 or 2). This list is in the same patient order as the above 
    genderList = labels_df['SEX_ID (1=m, 2=f)'].tolist()
        
    # Get the X which is patient slices in numpy array form, and y which is gender labels 
    X = XVariable(patients)
    # YVariable is a function to return 1 and 2 for male and female slices. yBinary is a function to make this 0 and 1
    # 0 = male, 1 = female 
    y = yBinary(YVariable(genderList, numberOfSlices))   
    

    # ==============================================================================
    #  Split the image data, label data, and set batch size and epochs for modeling
    # ==============================================================================

    # randomly split into 80/20 train and test 
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)
  
    batch_size = 60
    epochs = 50
    
    model = createModel()

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                          validation_data=(X_val, y_val))

    # this gives the loss and accuracy on the validation
    model.metrics_names
    model.evaluate(X_val, y_val)

    # summary of model
    model.summary()

    model.weights[1]
    
    '''
    #Maybe useful later:
    from sklearn.model_selection import StratifiedKFold
    seed = 7
    # define 10-fold cross validation test 
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    X = np.array(X)
    y = np.array(y)
    for train, test in kfold.split(X, y):
        model1 = createModel()
        # Compile model
        model1.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Fit the model
        history = model1.fit(X_train, y_train, batch_size = batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))        
        scores = model1.evaluate(X[test], y[test], verbose=1)
        print("%s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        '''

    # ==============================================================================
    # Confusion Matrix
    # ==============================================================================

    cmModel = confusionMatrix(model, X_val, y_val)
 
    # ==============================================================================
    # ROC and AUC
    # ==============================================================================

    rocAucModel = rocAuc(model, X_train, y_train, X_val, y_val)
    
    # ==============================================================================
    #  Confidence intervals
    # ==============================================================================
 
    # https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals 
    # In this case, rocAucModel1 is actually 'Y_pred'
    # We bootstrap by sampling with replacement on the prediction indices. 
    # Make y_val a pd.Series to match the datatype that rocAucModel is
    bootstrapAUCScores = bootstrapAUC(rocAucModel,pd.Series(y_val))
    confidenceIntervalModel = confidenceInterval(bootstrapAUCScores)
        

    '''
    classification error = incorrect predictions / total predictions
    e.g. 0.02 
    or 
    (0.02 * 100.0) = 2% classification error.

    Calculate confidence interval:
        error +/- const * sqrt( (error * (1 - error)) / n)

    z = 1.96 for 95% confidence

    eg. calculation
    error +/- const * sqrt( (error * (1 - error)) / n)
    0.02 +/- 1.96 * sqrt( (0.02 * (1 - 0.02)) / 50)
    0.02 +/- 1.96 * sqrt(0.0196 / 50)
    0.02 +/- 1.96 * 0.0197
    0.02 +/- 0.0388
    '''
    # classification error = incorrect predictions / total predictions. This value is 0
    # classificationError = 0 + 1.96 * sqrt( (0 * (1 - 0)) / 50)
    
    # ==============================================================================
    #  Calibration - Loss Curve
    # ==============================================================================
    lossCurveModel = loss(history)

    # ==============================================================================
    #  Calibration - Accuracy Curves
    # ==============================================================================
    accCurveModel = accCurve(history)


    #Save model
    model = model.save(r"C:\Users\CiaranO'Brien\Desktop\OncoR\Code\Ciaran\Brain\BrainImaging\Models\gender_classification_MRI_CNNADAM.h5\gender_classification_MRI_CNNRMSProp.h5")    
    #model = load_model(r"C:\Users\CiaranO'Brien\Desktop\OncoR\Code\Ciaran\Brain\BrainImaging\Models\gender_classification_MRI_CNNADAM.h5\gender_classification_MRI_CNNADAM.h5")    

if __name__ == "__main__":
    
    path = r"\\Syno-Onco\PUBLIC_DATA\Non-Referenced Data (challenge samples...)\Brain\Healthy Brain\IXI Dataset"
    pathMetadata = r"C:\Users\CiaranO'Brien\Desktop\OncoR\Data\Healthy Brain\IXI Dataset\IXI"

    main(path,pathMetadata)
    