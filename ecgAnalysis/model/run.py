import torch
import torch.nn.init
from torchinfo import summary
import warnings
import time
warnings.filterwarnings("ignore")

from data import*
from models import*
from utils import*

training_epochs = 200
lr = 0.001

def init(train_batch, val_batch, trainset):
    data_ = {'train': train_batch, 'val': val_batch}
    data_size_ = {'train': len(trainset), 'val': len(trainset)}
    return data_, data_size_

if __name__ == "__main__":
    load_start = time.time()
    train_batch, val_batch, test_batch, whole_test_batch, trainset, testset = setData().save_data()
    print("Loading process is finished! %fs for loading data." % (time_check(load_start)))

    data, data_size = init(train_batch, val_batch, trainset)
    model = kModel()
    summary(model, (128, 1, 187))
    best_acc, train_arr, val_arr, val_loss_arr = model_train(model, data, data_size, training_epochs, lr, test_batch, whole_test_batch)
    np.save('val_loss',val_loss_arr)
    plt_show(train_arr, val_arr, training_epochs)