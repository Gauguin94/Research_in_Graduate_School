import numpy as np
import matplotlib.pyplot as plt

def plt_show(train_arr, val_arr, epoch):
    train_range = np.arange(epoch)
    val_range = np.arange(epoch)
    plt.plot(train_range, train_arr, 'y-')
    plt.plot(val_range, val_arr, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Model accuracy')
    plt.legend()
    plt.show()