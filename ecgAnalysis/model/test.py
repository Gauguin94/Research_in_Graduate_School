import numpy as np
import matplotlib.pyplot as plt

x = np.load('/home/ee515/Analysis/21_11_16/val_loss.npy')
Y = np.load('/home/ee515/Analysis/21_11_16/val_loss_resnext.npy')
X = x[:55]
t = np.arange(len(Y))
plt.plot(t, X, 'r-', label = 'Proposed model')
plt.plot(t, Y, 'b--', label = 'ResNeXt')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()