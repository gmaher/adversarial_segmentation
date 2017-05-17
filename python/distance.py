import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import train, FCLayerTF
import argparse
np.random.seed(1)
tf.set_random_seed(1)
#Get MNIST data from tensorflow
parser = argparse.ArgumentParser()
parser.add_argument('--reg', default=False)
parser.add_argument('--grad_reg',default=False)
args = parser.parse_args()
reg = args.reg
grad_reg = args.grad_reg
plot_dir='./plots/'
dataset = tf.contrib.learn.datasets.load_dataset('mnist')
X_train = dataset.train.images
Y_train = dataset.train.labels
Y_train_1hot = np.eye(Y_train.max()+1)[Y_train]

X_test = dataset.test.images
Y_test = dataset.test.labels
Y_test_1hot = np.eye(Y_test.max()+1)[Y_test]

#Script parameters
EPS = 5.0/255
Nlayers = 10
Nunits = 100
activation = tf.nn.relu
Nnoise = 5
lamb = 75.0
lamb_grad = 1000.0
#Set up graph
x_tf = tf.placeholder(dtype=tf.float32, shape=[None, X_train.shape[1]])
y_tf = tf.placeholder(dtype=tf.float32, shape=[None, Y_train.max()+1])

fcs = []
outs = []
l = FCLayerTF.FCLayer(shape=(X_train.shape[1],Nunits), activation='relu')
fcs.append(l)
o = l.forward(x_tf)
outs.append(o)

for i in range(1,Nlayers):
    l = FCLayerTF.FCLayer(shape=(Nunits,Nunits), activation='relu')
    fcs.append(l)
    o = l.forward(o)
    outs.append(o)

l = FCLayerTF.FCLayer(shape=(Nunits,Y_train.max()+1), activation=None)
fcs.append(l)
o = l.forward(o)
outs.append(o)

yhat = o
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=yhat,labels=y_tf))

if reg:
    plot_dir += 'reg/'
    for l in fcs:
        w = l.weights[0]
        I = np.eye(l.shape[0],l.shape[0])
        loss += lamb*tf.reduce_mean(tf.abs(tf.matmul(w,w,transpose_b=True)-I))
elif grad_reg:
    plot_dir += 'gradreg/'
    g = tf.gradients(yhat,x_tf)[0]
    loss += lamb_grad*tf.reduce_mean(tf.square(g))

else:
    plot_dir += 'noreg/'
sess = tf.Session()
output = train.train_tf(sess,loss,x_tf,y_tf,
    X_train,Y_train_1hot,X_test,Y_test_1hot, num_iter=10000, learning_rate=5e-3, opt='sgd')

plt.figure()
plt.plot(output['train_loss'],color='r',linewidth=2,label='train loss')
plt.plot(output['test_loss'],color='g',linewidth=2,label='test loss')
plt.legend()
plt.xlabel('steps (50 iterations/step)')
plt.ylabel('loss')
lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1)
plt.grid('on')
plt.savefig(plot_dir+'mnist_fc_tf.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

#Noise analysis
weight_tensors = [l.weights[0] for l in fcs]
reg_outs = sess.run(outs,{x_tf:X_test,y_tf:Y_test_1hot})
weights = sess.run(weight_tensors,{x_tf:X_test,y_tf:Y_test_1hot})
wnorms = [np.linalg.norm(w, ord=2) for w in weights]
wbounds = []
wbounds.append(wnorms[0])
for i in range(1,len(wnorms)):
    wbounds.append(wbounds[i-1]*wnorms[i])

noise_outs = []
diffs = []
for i in range(Nnoise):
    noise = np.random.rand(X_test.shape[0],X_test.shape[1])*EPS
    noise_outs.append(sess.run(outs,{x_tf:X_test+noise,y_tf:Y_test_1hot}))
    d = [np.mean(np.linalg.norm(x-y, axis=1)) for x,y in zip(reg_outs,noise_outs[i])]
    diffs.append(d)

plt.figure()
for i in range(Nnoise):
    plt.plot(diffs[i], label='run {}'.format(i))
plt.xlabel('layer')
plt.ylabel('l2 norm of difference')
lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1)
plt.grid('on')
plt.savefig(plot_dir+'distance.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure()
plt.plot(wbounds, label='upper bound')
plt.xlabel('layer')
plt.ylabel('upper bound')
lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1)
plt.grid('on')
plt.savefig(plot_dir+'bound.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

u,s,v = np.linalg.svd(weights[0],full_matrices=0)
d = int(np.sqrt(784))
for i in range(u.shape[1]):
    plt.figure()
    im = u.T[i,:]
    plt.imshow(im.reshape((d,d)),cmap='gray')
    plt.colorbar()
    plt.savefig(plot_dir+'singvec{}'.format(i))
    plt.close()

###################################
# Adversarial test
###################################
UT = u.T
coeffs = np.arange(0,6.0,0.1)
accs = []
accs_renorm = []
accs_noise = []
noise_scale = np.linalg.norm(np.random.rand(X_test.shape[0],X_test.shape[1]),ord=2,axis=1)
noise_scale = np.mean(noise_scale)
for c in coeffs:
    X_test_mod = X_test + c*UT[0]
    X_test_mod_renorm = (X_test_mod-np.amin(X_test_mod))/(np.amax(X_test_mod)-np.amin(X_test_mod))
    X_test_mod_noise = X_test + c*np.random.rand(X_test.shape[0],X_test.shape[1])/noise_scale

    ypred = sess.run(yhat,{x_tf:X_test})
    yadv = sess.run(yhat,{x_tf:X_test_mod})
    yadv_renorm = sess.run(yhat,{x_tf:X_test_mod_renorm})
    yadv_noise = sess.run(yhat,{x_tf:X_test_mod_noise})

    ypred_labels = np.argmax(ypred,axis=1)
    yadv_labels = np.argmax(yadv,axis=1)
    yadv_renorm_labels = np.argmax(yadv_renorm,axis=1)
    yadv_noise_labels = np.argmax(yadv_noise,axis=1)

    acc = float(np.sum(np.argmax(ypred,axis=1) == Y_test))/len(X_test)
    acc_adv = float(np.sum(np.argmax(yadv,axis=1) == Y_test))/len(X_test)
    acc_renorm_adv = float(np.sum(np.argmax(yadv_renorm,axis=1) == Y_test))/len(X_test)
    acc_noise_adv = float(np.sum(np.argmax(yadv_noise,axis=1) == Y_test))/len(X_test)

    accs.append(acc_adv)
    accs_renorm.append(acc_renorm_adv)
    accs_noise.append(acc_noise_adv)

plt.figure()
plt.plot(coeffs,accs,linewidth=2,label='add')
plt.plot(coeffs,accs_renorm,linewidth=2,color='r',label='renorm')
plt.plot(coeffs,accs_noise,linewidth=2,color='g',label='noise')
plt.legend(loc='center right')
plt.xlabel('coefficient')
plt.ylabel('accuracy')
plt.savefig(plot_dir+'acc.png',dpi=300)

plt.close()

X_test_mod = X_test + 2.0*UT[0]
yadv = sess.run(yhat,{x_tf:X_test_mod})
ypred_labels = np.argmax(ypred,axis=1)
yadv_labels = np.argmax(yadv,axis=1)

def implot(mp,ax):
    im = ax.imshow(mp.astype(np.float32), cmap='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

for i in range(30):

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    implot(X_test[i].reshape((d,d)),ax1)
    implot(X_test_mod[i].reshape((d,d)),ax2)
    ax1.set_title('(Left): Original = {}'.format(ypred_labels[i]))
    ax2.set_title('(Right): Adversary = {}'.format(yadv_labels[i]))
    plt.tight_layout
    plt.savefig(plot_dir+'adversary{}.png'.format(i),dpi=300)

plt.close('all')

a = range(len(s))
sings = s[a][a]

plt.figure()
plt.plot(a,sings,label='singular values')
plt.savefig(plot_dir+'sings.png',dpi=300)
plt.close()
