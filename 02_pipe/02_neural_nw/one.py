# -*- coding: utf-8 -*-
"""one.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I0vAEuRo776JmDY1K8f839e6bpBg1dPG
"""

#%%-------------------------------------------------
import os
os.chdir('/content/drive/My Drive/gofrendly')
#os.getcwd()

!pip install dgl
import torch

device = torch.device('cuda'); #print(device) #cuda
#print(torch.cuda.current_device()) #0
#print(torch.cuda.device(0)) #<torch.cuda.device at 0x7fc1f955b198>
#print(torch.cuda.device_count()) #1
print(torch.cuda.get_device_name(0)) #'Tesla P100'
#print(torch.cuda.is_available()) #True
#print(os.cpu_count()) #2
#!nvidia-smi

#%%-------------------------------------------------
# Load the files
import pickle
import dgl

choice = 1

if choice == 1:
  # 15,000 | 48 
  with open('data/colab.pkl', 'rb') as f: [_, X, _, _] = pickle.load(f) #[G, X, trainpos, trainneg]
  with open('data/nw.pkl', 'rb') as f: [pos, neg] = pickle.load(f)
  with open('data/valpos.pkl', 'rb') as f: valpos = pickle.load(f)
elif choice == 2:
  # 15,000 | 816
  with open('data/X.pkl', 'rb') as f: X = pickle.load(f)
  with open('data/nw.pkl', 'rb') as f: [pos, neg] = pickle.load(f)
  with open('data/valpos.pkl', 'rb') as f: valpos = pickle.load(f)
elif choice == 3:
  # 120,000 | 48
  with open('data/all_X.pkl', 'rb') as f: X = pickle.load(f)
  with open('data/all_nw.pkl', 'rb') as f: [pos, neg] = pickle.load(f)
  with open('data/all_valpos.pkl', 'rb') as f: valpos = pickle.load(f)
  with open('data/nw.pkl', 'rb') as f: [pos2, neg2] = pickle.load(f)
  with open('data/valpos.pkl', 'rb') as f: valpos2 = pickle.load(f)

#with open('one.pkl', 'rb') as f: [emb] = pickle.load(f)

"""Move all the tensors to GPU """
import torch
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)
X = X.to(device); print(X.is_cuda)

# pos, neg, valpos = torch.tensor(pos), torch.tensor(neg), torch.tensor(valpos)
# pos, neg, valpos = pos.to(device), neg.to(device), valpos.to(device)
# print(pos.is_cuda, neg.is_cuda, valpos.is_cuda)

print(X.shape)

#%%-------------------------------------------------
""" 01. Embedding similarity distribution """
import time
import random
import torch
from scipy.stats import kurtosis 
import torch.nn.functional as F

def embplot(emb, N=2000):
  t0 = time.time()
  ind = random.sample(range(emb.shape[0]), N)
  iemb = emb[ind]
  s=[]; limit = len(iemb)-1
  
  for i,a in enumerate(iemb):
      if i == limit: break
      val = F.cosine_similarity(a[None,:], iemb[i+1:])
      s.extend(val)
  print('Time taken:', time.time()-t0)
  s = torch.stack(s)
  return s.tolist()

#%%-----------------------------------
""" 03. Encoder Model """
import pipe
import nn
import torch
import time
import importlib; importlib.reload(nn); importlib.reload(pipe)

fdim = X.shape[1]
#embed = torch.nn.Embedding(*X.shape).to(device)
onemodel = nn.net(  inputs = X, #embed.weight,
                    output_size = fdim,
                    layers = [],
                    dropout = 0.00,
                    lr = 1e-4, #1e-2, 2e-3
                    opt = 'RMSprop', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                    cosine_lossmargin = 0.25, #-1, -0.5
                    pos = pos, #72382: (16063,56319)
                    neg = neg, #402761: (2134,400627)
                    val_pos = valpos)
onemodel.to(device)

print(X.shape)

#%%-------------------------------------------------
""" 03. a. Training """
from scipy.stats import kurtosis 

epochs = 500
lr = 1e-3 #2e-3
loss_interval, eval_interval, emb_interval = 10, 50, 50
intervals = [loss_interval, eval_interval, emb_interval]

onemodel.optimizer  = getattr(torch.optim, 'RMSprop')(onemodel.net.parameters(), lr)
[newemb, train_eval, val_eval, loss_values, embs] = onemodel.train(epochs, intervals)

#with open('data/one_training.pkl', 'wb') as f: pickle.dump([newemb, train_eval, val_eval, loss_values, embs], f)
# Hitrate = 26.3 MRR = 1.8; Hitrate = 28.2 MRR = 0.7

#%%-------------------------------------------------
import pickle
#with open('data/one_training.pkl', 'wb') as f: pickle.dump([newemb, train_eval, val_eval, loss_values, embs], f)
with open('data/one_training.pkl', 'rb') as f: [newemb, train_eval, val_eval, loss_values, embs] = pickle.load(f)

#with open('data/one.pkl', 'wb') as f: pickle.dump(embs[3], f)

#%%--------------------
""" Plot the graphs """
import matplotlib.pyplot as plt 
import seaborn as sns

epochs = 600
X_loss, X_hr, X_mrr = 0.0891, 26.4, 1.3
loss_interval, eval_interval, emb_interval = 10, 50, 100 #5, 30, 100
# nn01_loss, nn01_hr, nn01_mrr = 0.069, 27.1, 0.8

#%%-------------------------------------------------
# Plot the loss values

fig1 = plt.figure(1); plt.grid()
plt.plot(range(1,epochs+1), loss_values)
plt.xticks([1] + list(range(100, epochs+1, 100)))
fig1.suptitle('Loss value Vs. Epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss value')

plt.axhline(y=X_loss, color='green', linestyle='--')
plt.text(x=350, y=X_loss+0.02, s='Non-learned = '+str(X_loss), fontsize=12, color='green')
plt.show()

fig1.savefig('diagrams/one_lossvalues.jpg')

#%%-------------------------------------------------
# Plot the hit-rate and MRR
train_hr = [i for i,j in train_eval]; train_mrr = [j for i,j in train_eval] 
val_hr = [i for i,j in val_eval]; val_mrr = [j for i,j in val_eval]  

fig2 = plt.figure(2); plt.grid()
plt.xticks( list(range(len(train_hr))), [eval_interval] + list(range(eval_interval*2, eval_interval * (1+len(train_hr)), eval_interval)))
plt.plot(train_hr, label = 'train'); plt.plot(val_hr, label = 'valid'); plt.legend(loc='center right')
fig2.suptitle('Hit-rate Vs. Epoch'); plt.xlabel('Epoch'); plt.ylabel('Hitrate')

plt.axhline(y=X_hr, color='green', linestyle='--')
plt.text(x=0, y=X_hr+0.5, s='Non-learned = '+str(X_hr), fontsize=12, color='green')

fig3 = plt.figure(3); plt.grid()
plt.xticks( list(range(len(train_mrr))), [eval_interval] + list(range(eval_interval*2, eval_interval * (1+len(train_mrr)), eval_interval)))
plt.plot(train_mrr, label = 'train'); plt.plot(val_mrr, label = 'valid');  plt.legend(loc='center right')
fig3.suptitle('MRR Vs. Epoch'); plt.xlabel('Epoch'); plt.ylabel('Hitrate')

plt.axhline(y=X_mrr, color='green', linestyle='--')
plt.text(x=1, y=X_mrr+0.05, s='Non-learned = '+str(X_mrr), fontsize=12, color='green')

fig2.savefig('diagrams/one_hr.png')
fig3.savefig('diagrams/one_mrr.png')

#%%-------------------------------------------------
# Plot the embedding similarity distribution curves
from scipy.stats import kurtosis 

fig4 = plt.figure(4); plt.grid()
k = [] #kurtosis_values
for i,emb in enumerate(embs[:4]):
  e = embplot(emb, 2000); k.append(kurtosis(e)); print('Kurtosis number', i, '=',  k[i])
  sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'Epoch %d' %(emb_interval*(i)))

plt.xlim(-1.0,1.0); plt.legend(loc='upper center')
fig4.suptitle('Embedding similarity distribution'); plt.xlabel('Embedding cosine similarity'); plt.ylabel('Probability density of pairwise distances')

# fig4.savefig('diagrams/one_embdist.jpg')
# Kurtosis = 14.14 [-1.51, -1.2, -0.63, -0.197]

#%%-------------------------------------------------
#e = embplot(X, 2000); print('Kurtosis =', kurtosis(e)) #14.14
sns.distplot(e, hist=False, kde = True, norm_hist=True)
plt.grid(); 
plt.xlim(min(e), 1.0)
#plt.xlim(-1.0, 1.0) 
fig4.suptitle('Embedding similarity distribution'); plt.xlabel('Embedding cosine similarity'); plt.ylabel('Probability density of pairwise distances')

#fig4.savefig('diagrams/X_embdist.jpg')
#fig4.savefig('diagrams/X_zoomed_embdist.jpg')

#%%-------------------------------------------------

""" Recommendations on raw vectors """
import time
import pipe
import importlib; importlib.reload(pipe)

t0 = time.time()
onepipe = pipe.pipeflow(X, K=500)
res_train = onepipe.dfmanip(pos)
res_val = onepipe.dfmanip(valpos)
print('Time taken:', time.time()-t0)

#%%-------------------------------------------------
""" 01. Embedding similarity distribution """
import time
import random
import torch
from scipy.stats import kurtosis 
import torch.nn.functional as F

def embplot(emb):
  t0 = time.time()
  ind = random.sample(range(emb.shape[0]), 2000)
  iemb = emb[ind]
  s=[]; limit = len(iemb)-1
  
  for i,a in enumerate(iemb):
      if i == limit: break
      val = F.cosine_similarity(a[None,:], iemb[i+1:])
      s.extend(val)
  print('Time taken:', time.time()-t0)
  s = torch.stack(s)
  return s.tolist()

import matplotlib.pyplot as plt
import seaborn as sns

fig1 = plt.figure(1); sns.set_style('whitegrid')

e = embplot(X[:,:3]); k = kurtosis(e); print('01 =', k)
sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'Numerical')

e = embplot(X[:,3:48]); k = kurtosis(e); print('02 =', k)
sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'Categorical')

e = embplot(X[:,48:]); k = kurtosis(e); print('03 =', k)
sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'S-BERT')

e = embplot(X); k = kurtosis(e); print('04 =', k)
sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'All')

plt.legend(loc='upper left'); plt.xlim(-0.3,1.0)
fig1.suptitle('Embedding similarity distribution'); 
plt.xlabel('Embedding cosine similarity'); plt.ylabel('Probability density of pairwise distances')

#fig1.savefig('diagrams/emb_dist.jpg')

#%%-------------------------------------------------
""" Curriculum learning: Encoder Model """

#%%-----------------------------------
""" 03. Encoder Model """
import pipe
import nn
import torch
import matplotlib.pyplot as plt 
import importlib; importlib.reload(nn); importlib.reload(pipe)
import seaborn as sns
import time

testneg = neg 
testpos = pos
# embed = nn.Embedding(*X.shape)
onemodel = nn.net(  inputs = X, #embed.weight
                    output_size = 816,
                    layers = [],
                    dropout = 0,
                    lr = 1e-3, #1e-2, 2e-3
                    opt = 'RMSprop', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                    cosine_lossmargin = 0.25, #-1, -0.5
                    pos = testpos, #72382: (16063,56319)
                    neg = testneg) #402761: (2134,400627)

run = 1; train=[]; val=[]; exp=[]
if run==0:
  t0 = time.time()
  emb = onemodel.train(epochs=1, lossth=0.01)
  onepipe = pipe.pipeflow(emb, K=500)
  train.append(onepipe.dfmanip(testpos)[0])
  val.append(onepipe.dfmanip(valpos)[0])

  e = embplot([emb]); plt.figure(4); sns.kdeplot(e)
  print('Time taken:', time.time()-t0)


#%%--------------------
""" 03. a. Training """
for i in range(4):
  t1 = time.time()
  onemodel.optimizer = getattr(torch.optim, 'RMSprop')(onemodel.net.parameters(), 2e-3/(2*i+1))
  emb = onemodel.train(epochs = 50, lossth=0.01)
  onepipe = pipe.pipeflow(emb, K=500, nntype='cosine')
  train.append(onepipe.dfmanip(testpos)[0])
  #exp.append(onepipe.dfmanip(trainpos)[0])
  val.append(onepipe.dfmanip(valpos)[0])
  e = embplot([emb]); plt.figure(4); sns.kdeplot(e)
  print('Time taken by train round', i, '=', time.time()-t1)

fig2 = plt.figure(2); plt.plot(train); plt.plot(val); plt.legend()
fig2.suptitle('Hitrate Vs. Epoch*50')
plt.xlabel('Epoch'); plt.ylabel('Hitrate');
#plt.figure(3); plt.plot(exp)
print(emb.mean(0)); print(emb.std(0))

#%%-----------------------------------
""" 04. Indiv. encoder Model """
import pipe
import nn
import torch
import matplotlib.pyplot as plt 
import importlib; importlib.reload(nn); importlib.reload(pipe)
import seaborn as sns
import time

for i,node in enumerate(nodes):
  t0 = time.time()

  [npos, nsoftneg, nhardneg] = func_getsamples(node)
  ind = torch.cat([npos, nsoftneg, nhardneg],0)
  model[i] = nn.encoder(  model = pretrained_encoder,
                          inputs = X[ind],
                          output_size = 48,
                          layers = [],
                          dropout = 0.1,
                          lr = 1e-3,
                          opt = 'RMSprop',
                          cosine_lossmargin = 0,
                          pos = npos, #72382: (16063,56319)
                          neg = [nsoftneg, nhardneg]) #402761: (2134,400627)
  # Training
  emb[i] = model[i].train(epochs = 1, lossth=0.01)
  eval_train[i] = pipe.pipeflow(emb[i], K=100, nntype='cosine').dfmanip(npos)[0]
  
  # eval_val[i] = pipe.pipeflow(emb[i], K=100, nntype='cosine').dfmanip(val_npos)[0]
  print('Time taken by node', i, '=',time.time()-t0)

  #e = embplot([emb]); plt.figure(4); sns.kdeplot(e)

""" 03. a. Training """
for i in range(4):
  t1 = time.time()
  onemodel.optimizer = getattr(torch.optim, 'RMSprop')(onemodel.net.parameters(), 2e-3/(2*i+1))
  emb = onemodel.train(epochs = 50, lossth=0.01)
  onepipe = pipe.pipeflow(emb, K=100, nntype='cosine')
  train.append(onepipe.dfmanip(testpos)[0])
  #val.append(onepipe.dfmanip(valpos)[0])
  #e = embplot([emb]); plt.figure(4); sns.kdeplot(e)
  print('Time taken by train round', i, '=', time.time()-t1)

plt.figure(2); plt.plot(train)
plt.figure(3); plt.plot(val)
print(emb.mean(0))
print(emb.std(0))

#%%-------------------------------------------------
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Hit-rate')#, color=color)
ax1.plot( train_hr, color='tab:blue')
ax1.plot( val_hr, color='tab:cyan')
ax1.tick_params(axis='y')#, labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('MRR')#, color=color)  # we already handled the x-label with ax1
ax2.plot( train_mrr, color='tab:green')
ax2.plot( val_mrr, color='tab:olive')
ax2.tick_params(axis='y')#, labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()