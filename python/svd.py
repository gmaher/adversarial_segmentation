import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import train, FCLayerTF
import argparse


plot_dir='./plots/svd/'
dataset = tf.contrib.learn.datasets.load_dataset('mnist')
X_train = dataset.train.images

N,V = X_train.shape
d = int(np.sqrt(V))

u,s,v = np.linalg.svd(X_train.T,full_matrices=0)

for i in range(50):

    plt.figure()
    plt.imshow(u[:,i].reshape((d,d)),cmap='gray')
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(plot_dir+'u{}.png'.format(i),dpi=300)
    plt.close()

    plt.figure()
    plt.plot(v[:,i],linewidth=2)
    plt.tight_layout()
    plt.savefig(plot_dir+'v{}.png'.format(i),dpi=300)
    plt.close()

plt.figure()
a = range(len(s))
plt.plot(s[a][a],label='singular values',linewidth=2)
plt.legend()
plt.savefig(plot_dir+'sings.png',dpi=300)
plt.close('all')
