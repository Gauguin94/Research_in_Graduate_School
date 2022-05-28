import torch
import torch.nn as nn
import numpy as np
import copy
import adabound
import time
import sys
import os
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TTL = 21892 # Total Test data length

def evaluate(model, test_batch):
    predict_sheet = np.array([])
    iter = 0
    running_corrects = 0
    with torch.no_grad():
        for X_test, Y_test in test_batch:
            iter += 1
            X_test = X_test.view(X_test.size(0), 1, X_test.size(1))
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)

            predict = model(X_test)
            predict = predict.squeeze(1)
            Y_test = Y_test.squeeze(1)
            _, prediction = torch.max(predict, 1)
            predict_sheet = np.append(predict_sheet, prediction.cpu())
            running_corrects += torch.sum(prediction == Y_test.data)
        #sys.stdout = open('result.txt', 'w')
        print('Model Acc : {}'.format(running_corrects.double()/TTL))
        #sys.stdout.close()
        #print('Model Acc : {}'.format(running_corrects.double()/TTL))
    return predict_sheet

def f1_score(whole_test_batch, predict_sheet):
    for _, answer in whole_test_batch:
        answer = answer.to(device)
        answer_sheet = answer.squeeze(1).to(device)
    y_true = np.array(answer_sheet.cpu())
    y_pred = np.array(predict_sheet)

    recall = metrics.recall_score(y_true, y_pred, average = None)
    precision = metrics.precision_score(y_true, y_pred, average = None)
    f1 = metrics.f1_score(y_true, y_pred, average = None)
    print('Sensitivity(Recall) of model')
    print('N: {:.4f} S: {:.4f} V: {:.4f} F: {:.4f} Q: {:.4f}'.format(recall[0], recall[1], recall[2], recall[3], recall[4]))
    print('Precision of model')
    print('N: {:.4f} S: {:.4f} V: {:.4f} F: {:.4f} Q: {:.4f}'.format(precision[0], precision[1], precision[2], precision[3], precision[4]))
    print('F1-Score of model')
    print('N: {:.4f} S: {:.4f} V: {:.4f} F: {:.4f} Q: {:.4f}'.format(f1[0], f1[1], f1[2], f1[3], f1[4]))
    mat = confusion_matrix(y_pred, y_true)
    print(mat)