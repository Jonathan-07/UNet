from matplotlib import pyplot as plt
import numpy as np

data = np.load('./output/750 epochs/acc38_lr0.1_bs1_mom0.99_epoch750.npy')
print(data)
plt.imshow(data, cmap='gray')
plt.show()
