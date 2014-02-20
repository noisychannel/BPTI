#!/usr/bin/env python

import os
import numpy as np
import sys
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib

dataLocation = '/damsl/projects/mddb/bpti_db/features'

# Each row needs to represent one observation
# Columns represent features
# Apply mean and variance normalization to once column at a time 
# Remember the scaling factor so that these can be applied to the test data

# Will store the data as an n x d matrix where n is the number of observations
# (rows) and d is the number of features (columns)

# This might get big pretty quickly, is there any benefit is storing 
# segmented versions of this dataset? What is the advantage if any?
# What's a good way to store these?

def createDataMatrix(featureFiles):
  print "Creating the feature matrix"
  dataMatrix = None
  labels = []
  for fileName in featureFiles:
    featureFile = open(fileName)
    featureFileData = featureFile.read()
    listOfObservations = eval(featureFileData)
    featureFile.close()

    for index, item in enumerate(listOfObservations):
      # Create a temporary list which will contain the details of the feature
      # values for this observation
      tmpList = [] 
      # This is dictionary, enumerate by key, exclude RMSD
      for key, value in item.iteritems():
        if key == "rmsd":
          # Get the minimum value of the RMSD and treat its index as the label
          labels.append(value.index(min(value)))
        else:
          tmpList += value

      if dataMatrix is None:
        dataMatrix = np.array(tmpList)
      else:
        dataMatrix = np.vstack((dataMatrix, tmpList))

  print "Done creating the feature matrix"
  return dataMatrix, labels

def cmvn(featureMatrix):
  '''
  Does mean and variance normalization for features
  '''
  print "Performing CMVN"
  # Remember the transformation so that this can be applied to the 
  # testing set later
  transformations = []
  for column_no in range(len(featureMatrix[0])):
    scaler = preprocessing.StandardScaler().fit(featureMatrix[:,column_no])
    normalizedFeature = scaler.transform(featureMatrix[:,column_no])
    # Now replace data
    featureMatrix[:,column_no] = normalizedFeature
    # Store the scalers for later
    transformations.append(scaler)
  print "Done performing CMVN"

  return transformations

def cmvnSanityCheck(featureMatrix):
  '''
  Checks to see if all columns (features) in the feature matrix have
  0 mean and 1 STD
  '''
  for column_no in range(len(featureMatrix[0])):
    mean = featureMatrix[:,column_no].mean()
    if not np.allclose([mean],[0.0]):
      raise StandardError("Error : CMVN not done properly (MEAN) | " + str(column_no) + " | " + str(mean))
    std = featureMatrix[:,column_no].std()
    # The second condition gets activated only when the feature vector is all zeros
    if not np.allclose([std],[1.0]) and not np.allclose([std],[0.0]):
      raise StandardError("Error : CMVN not done properly (STD) | " + str(column_no) + " | " + str(std))
  print "CMVN verified : OK"


def trainSVM(featureMatrix, labels, model):
  '''
  Trains the SVM on the training set provided
  '''
  # SVM regularization parameter : Vary this and see what happens
  # Also vary this with rbf gamma and poly kernel degree and see what happens
  C = 1.0 
  trainedModel = None
  if model == "linear_kernel":
    trainedModel = svm.SVC(kernel='linear', C=C).fit(featureMatrix, labels)
  elif model == "rbf_kernel":
    trainedModel = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(featureMatrix, labels)
  elif model == "poly_kernel":
    trainedModel = svm.SVC(kernel='poly', degree=3, C=C).fit(featureMatrix, labels)
  elif model == "linear":
    trainedModel = svm.LinearSVC(C=C).fit(featureMatrix, labels)
  return trainedModel

def transformTestData(transformations, featureMatrix):
  for column_no in range(len(featureMatrix[0])):
    normalizedFeature = transformations[column_no].transform(featureMatrix[:,column_no])
    # Now replace data
    featureMatrix[:,column_no] = normalizedFeature
  print "Done applying transformations to the test data"

def predict(trainedModel, featureMatrix, labels):
  models = [trainedModel]
  correctCounts = np.zeros(len(models))
  for row_no in range(len(featureMatrix)):
    observation = featureMatrix[row_no,:]
    result = labels[row_no]
    for index, model in enumerate(models):
      prediction = model.predict(observation)[0]
      print str(prediction) + "\t" + str(result) + "\n"
      if prediction == result:
        correctCounts[index] = correctCounts[index] + 1
  for index, model in enumerate(models):
    print "------------------------------------"
    print model
    print "------------------------------------"
    print float((correctCounts[index] * 100.0 ) / len(featureMatrix) )
    print "------------------------------------"

def parseArgs(argv):
  model = None
  runMode = "train"
  if len(argv) > 1:
    model = argv[1]
    if model not in ['linear_kernel', 'poly_kernel', 'rbf_kernel', 'linear_svc']:
      print "Valid SVM model choices are linear_kernel , poly_kernel , rbf_kernel or linear_svc"
      sys.exit(2)
  if len(argv) > 2:
    runMode = argv[2]
    if runMode not in ['train', 'test']:
      print "The run mode can only be train or test"
      sys.exit(2)
  return model, runMode

def persistData(trainedModel, model, transformations):
  joblib.dump(trainedModel, 'models/' + model + '_400.pkl')
  if not os.path.isfile("transformations/train_transform_400.pkl"):
    joblib.dump(transformations, 'transformations/train_transform_400.pkl')

def loadPersistedData(model):
  trainedModel = joblib.load('models/' + model + '_400.pkl')
  transformations = joblib.load('transformations/train_transform_400.pkl')
  return trainedModel, transformations

if __name__ == '__main__':  
  # Check for arguments to this script specifying which model to run and to train or decode
  model, runMode = parseArgs(sys.argv)
  print "model is: ", model
  print "runMode is: " , runMode

  #  sys.exit(1)
  # Run cross-folding
  dataFiles = [ dataLocation + '/' + f for f in os.listdir(dataLocation) if os.path.isfile(os.path.join(dataLocation,f)) ]
#  print "datafiles", dataFiles
  if len(dataFiles) > 4:
    dataFiles = dataFiles[:4]

#  print "datafiles-trimmed", dataFiles
  trainIndex = int(0.8*len(dataFiles))
  featureFileNames = dataFiles[:trainIndex]
  testFileNames = dataFiles[trainIndex:]
  print "testFileNames", testFileNames

  if runMode == "train":
    # Create the training set first 
    print "entering the training mode"
    training_featureMatrix, training_labels = createDataMatrix(featureFileNames)
    print "The training data has " + str(len(training_featureMatrix)) + " observations and " + str(len(training_featureMatrix[0])) + " features"
    transformations = cmvn(training_featureMatrix)
    cmvnSanityCheck(training_featureMatrix)
    trainedModel = trainSVM(training_featureMatrix, training_labels, model)
    persistData(trainedModel, model, transformations)

  if runMode == "test":
    test_featureMatrix, test_labels = createDataMatrix(testFileNames)
    print "The test data has " + str(len(test_featureMatrix)) + " observations and " + str(len(test_featureMatrix[0])) + " features"
    # Start testing now
    # First apply the transformations to the test data
    trainedModel, transformations = loadPersistedData(model)
#    print "after loadPersist-trainedModel", trainedModel
#    print "after loadPersist-transformations", transformations
    transformTestData(transformations, test_featureMatrix)
    predict(trainedModel, test_featureMatrix, test_labels)
