import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import adabound
import time
import sklearn.metrics as metrics
from .test import*
from .timeCheck import*
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FocalLoss(nn.Module):
    '''
    gamma = [0,  0.5,  1,  2,  5]
    If gamma is 0, 
    Focal loss == Cross Entropy Loss.
    In that paper, they recommend alpha = 0.25 and gamma 2.0.
    (Default is 0.)
    '''
    def __init__(self, gamma = 0, alpha = None, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, float)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt*Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum() 

def model_train(model, data, data_size, training_epochs, lr, test_batch, whole_test_batch):
    train_arr = []
    val_arr = []
    val_loss_arr = []
    # 72471, 2223, 5788, 641, 6431
    optimizer = adabound.AdaBound(model.parameters(), lr = lr, final_lr = 0.1)
    criterion = FocalLoss(alpha = [0.15, 0.75, 0.65, 0.75, 0.25], gamma = 2) ## Insert gamma and alpha value.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()

    for epoch in range(training_epochs):
        start = time.time()
        print('Epoch: %d'%(epoch+1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data[phase]:
                inputs = inputs.view(inputs.size(0), 1, inputs.size(1)).to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.squeeze(1).to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).cuda()
                    outputs = outputs.squeeze(1).to(device)
                    _, val_preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels.long()).cuda()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        val_preds = val_preds.cuda()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(val_preds == labels.data).to(device)

            epoch_loss = running_loss / data_size[phase]
            epoch_acc = running_corrects.double() / data_size[phase]
            print('{} Loss: {:.8f} Acc: {:.6f} '.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                epoch_time = time_check(start)
                print('Time: {:.4f}'.format(epoch_time))
                val_arr.append(epoch_acc.cpu().numpy().max())
                val_loss_arr.append(epoch_loss)
            else:
                train_arr.append(epoch_acc.cpu().numpy().max())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                with torch.no_grad():
                    predict_sheet = evaluate(model, test_batch)
                    f1_score(whole_test_batch, predict_sheet)
                testing_time = time_check(start)
                print('Testing time: {:.4f}'.format(testing_time))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed/3600, (time_elapsed%3600)/60, time_elapsed%60))
    print('Best val Acc: {:.5f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return best_acc, train_arr, val_arr, val_loss_arr