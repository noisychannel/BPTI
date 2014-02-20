#!/usr/bin/env python

import os 

dataLocation = '/damsl/projects/mddb/bpti_db/features'

def getLabels(featureFiles, outFile):
  for fileName in featureFiles:
    featureFile = open(fileName)
    featureFileData = featureFile.read()
    listOfObservations = eval(featureFileData)
    featureFile.close()

    for index, item in enumerate(listOfObservations):
      for key, value in item.iteritems():
        if key == "rmsd":
          # Get the minimum value of the RMSD and treat its index as the label
          outFile.write(str(value.index(min(value))) + "\n")

  return labels


if __name__ == '__main__':  

  dataFiles = [ dataLocation + '/' + f for f in os.listdir(dataLocation) if os.path.isfile(os.path.join(dataLocation,f)) ]
  
  outFile = open('state_labels_large','w+')
  ylabels = getLabels(dataFiles, outFile)
  outFile.close()
