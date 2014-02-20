#!/usr/bin/env bash

if [ ! -f general_svm.py ]
then
  echo "The script general_svm.py needs to be in the the same directory as this script"
  exit 1
fi

mkdir -p transformations
mkdir -p models
mkdir -p logs

#nohup ./general_svm.py linear_kernel train > logs/linear_kernel_train.log &
#nohup ./general_svm.py poly_kernel train > logs/poly_kernel_train.log &
#nohup ./general_svm.py rbf_kernel train > logs/rbf_kernel_train.log &
#nohup ./general_svm.py linear_svc train > logs/linear_svc_train.log &
nohup ./general_svm2.py linear_kernel test > logs/linear_kernel_test2.log &
nohup ./general_svm2.py poly_kernel test > logs/poly_kernel_test2.log &
nohup ./general_svm2.py rbf_kernel test > logs/rbf_kernel_test2.log &
nohup ./general_svm2.py linear_svc test > logs/linear_svc_test2.log &
