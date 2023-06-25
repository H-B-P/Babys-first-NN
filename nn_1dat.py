import numpy as np
import pandas as pd

import math
import random

import metrics


MOMENTUM=0.1
LEAK=0.05

np.random.seed(0)
np.set_printoptions(threshold=np.inf)

#Calculus!

def Logit_link(x):
 return 1/(1+np.exp(-x))

def Logit_link_grad(x):
 return np.exp(-x)/((1+np.exp(-x))**2)

def Logistic_grad(pred,act):
 return (pred-act)/(pred*(1-pred))

#

def make_censor_array(rows,cols):
 arr = np.zeros([rows, cols])
 arr[:rows//2, :cols//2] = 1
 arr[rows//2:, cols//2:] = 1
 return arr

def make_censor_arrays(model, stopat = 2):
 censor = []
 for l in range(len(model)):
  if l<stopat:
   censor.append(make_censor_array(len(model[l]), len(model[l][0])))
  else:
   censor.append(np.ones([len(model[l]),len(model[l][0])]))
 return censor

def apply_censor_arrays(model, censor):
 op = []
 for l in range(len(model)):
  op.append(model[l]*censor[l])
 return op

def lrelu(x):
 if x>0:
  return x
 else:
  return LEAK*x

def dlrelu(x): #Helpfully this always gives the same result for input or output
 if x>0:
  return 1
 else:
  return LEAK

vec_lrelu = np.vectorize(lrelu)
vec_dlrelu = np.vectorize(dlrelu)

def prepare_model(layerList = [4,3,3,3,1], mult=1):
 model = []
 for i in range(len(layerList)-1):
  model.append((np.random.random([layerList[i+1],layerList[i]])*2-1)*mult) #Randomness, really?
 return model

def predict(inputData, model, archive=False, censorstop=0):
 iD = np.array(inputData)
 pred = iD
 vecs = []
 vecs.append(pred)
 for L in range(len(model)-1):
  mat = model[L]
  pred = mat @ pred
  if L!=(censorstop-1):
   pred = vec_lrelu(pred)
  vecs.append(pred)
 
 mat = model[-1]
 pred = mat @ pred
 vecs.append(pred)
 pred = Logit_link(pred)
 
 if len(pred)==1:
  pred = pred[0]
 
 if archive:
  return pred, vecs
 return pred

def replace_near_zero(arr, distance):
    arr[np.abs(arr) < distance] = 0
    return arr

def train_model(inputData, model, act, lr, nrounds = 100, censor=[], censorstop=0):
 
 prevDelta = [0]*len(model)
 
 for r in range(nrounds):
  #print(lr, r)
  
  prevModel = model
  
  pred, vecs = predict(inputData, model, True, censorstop)
  dlosses = Logistic_grad(pred,act)*Logit_link_grad(vecs[-1]) #dLoss/d[final node] = dLoss/dpred * dpred/d[final node]
  
  for L in range(len(model)):
   ins = vecs[-(L+2)]
   outs = vecs[-(L+1)]
   mat = model[-(L+1)]
   
   if ((L==0) or (L==(len(model)-censorstop))):
    dldp = dlosses #No relu-ing on final node or directly after merge!
   else:
    dldp = dlosses*vec_dlrelu(outs) #dloss/dpre = dloss/dpost * dpost/dpre
   
   
   #propagate losses
   
   dlosses = np.transpose(mat)@dldp #dloss/dinput = dpre/dinput * dloss/dpre
   
   #update connections
   
   deltaC = dldp@np.transpose(ins) #dloss/dlayer = dloss/dpre * dpre/dlayer
   
   deltaC = deltaC * censor[-(L+1)]
   #print(deltaC)
   
   model[-(L+1)]-=lr/len(inputData[0])*deltaC
  
  
  
  for L in range(len(model)):
   model[L] = model[L] + prevDelta[L]*MOMENTUM
   prevDelta[L] = model[L] - prevModel[L]
  
 
 model = [replace_near_zero(l, 0.00001) for l in model]
 
 return model

###################

def expand_into_noise(arr, noise1, noise2):
 op = np.zeros([len(arr)+noise1, len(arr[0])+noise2])
 op[:len(arr),:len(arr[0])]=arr
 op[len(arr):, :] = np.random.random([noise1, len(op[0])])*2-1
 return op

def expand_into_noise_duo(arr, noise1, noise2):
 op= np.zeros([len(arr)+2*noise1, len(arr[0])+2*noise2])
 op[:(len(op)//2), :(len(op[0])//2)] = expand_into_noise(arr[:(len(arr)//2),:(len(arr[0])//2)], noise1, noise2)
 op[(len(op)//2):, (len(op[0])//2):] = expand_into_noise(arr[(len(arr)//2):,(len(arr[0])//2):], noise1, noise2)
 return op

def expand_into_noise_switchup(arr, noise1, noise2):
 op = np.zeros([len(arr)+noise1, len(arr[0])+noise2])
 op[:len(arr),:(len(arr[0])//2)] = arr[:,:(len(arr[0])//2)]
 op[:len(arr),(len(arr[0])//2):] = arr[:,(len(arr[0])//2):]
 op[len(arr):, :] = np.random.random([noise1, len(op[0])])*2-1
 return op
 

def round_two_ify_censorlvl(arr, noise1, noise2):
 op = np.zeros([(len(arr)+noise1), len(arr[0])+noise2])
 op[(len(op)-noise1):len(op), :] = 1
 return op

def round_two_ify_censorlvl_duo(arr, noise1, noise2):
 op= np.zeros([len(arr)+2*noise1, len(arr[0])+2*noise2])
 op[:len(arr)//2+noise1, :len(arr[0])//2+noise2] = round_two_ify_censorlvl(arr[:len(arr)//2,:len(arr[0])//2], noise1, noise2)
 op[len(arr)//2+noise1:, len(arr[0])//2+noise2:] = round_two_ify_censorlvl(arr[len(arr)//2:,len(arr[0])//2:], noise1, noise2)
 return op

###################

df = pd.read_csv('../cleanedData.csv')

trainDf = df[:10000].reset_index()
#testDf = df[10000:10002].reset_index()
testDf = df[100000:].reset_index()


heroes = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'J', 'L', 'M', 'P', 'R', 'S', 'T', 'W']

Xes = ['Blue_A', 'Blue_B', 'Blue_C', 'Blue_D', 'Blue_F', 'Blue_G', 'Blue_H', 'Blue_J', 'Blue_L', 'Blue_M', 'Blue_P', 'Blue_R', 'Blue_S', 'Blue_T', 'Blue_W'] + ['Green_A', 'Green_B', 'Green_C', 'Green_D', 'Green_F', 'Green_G', 'Green_H', 'Green_J', 'Green_L', 'Green_M', 'Green_P', 'Green_R', 'Green_S', 'Green_T', 'Green_W']

train_X = np.array(trainDf[Xes])
train_y = np.array(trainDf["GreenWin?"])
test_X = np.array(testDf[Xes])
test_y = np.array(testDf["GreenWin?"])

hlayers = [24,2,12]
print(hlayers)

SEP=2
NOISE=12
model=prepare_model([len(Xes)]+hlayers+[1], 0.5)
censor = make_censor_arrays(model, SEP)

model = apply_censor_arrays(model, censor) 

if True:
 
 #preds = predict(np.transpose(train_X), model)
 for lr in [1]:
  print(lr)
  model = train_model(np.transpose(train_X), model, train_y, lr, 50 , censor, SEP)

 for n in range(10):
  print('-')
  print(n)
  print('-')
  model = train_model(np.transpose(train_X), model, train_y, 2, 100 , censor, SEP)
  
  preds = predict(np.transpose(test_X), model,False ,SEP)
  
  testDf["PREDICTED"]=preds
  testDf["ACTUAL"]=testDf["GreenWin?"]
  
  print(metrics.get_gini(testDf, "PREDICTED","ACTUAL"))
  print(metrics.get_Xiles(testDf, "PREDICTED","ACTUAL"))
 
 print("-")
 
 for m in model:
  print(len(m), len(m[0]))
 for c in censor:
  print(len(c), len(c[0]))
 
 print("-")
 
 model[0] = expand_into_noise_duo(model[0], NOISE,0)
 model[1] = expand_into_noise_duo(model[1], 1,NOISE)
 model[2] = expand_into_noise_switchup(model[2], NOISE,2)
 model[3] = expand_into_noise(model[3], 0,NOISE)
 
 censor[0] = round_two_ify_censorlvl_duo(censor[0], NOISE,0)
 censor[1] = round_two_ify_censorlvl_duo(censor[1], 1,NOISE)
 censor[2] = round_two_ify_censorlvl(censor[2], NOISE,2)
 censor[3] = round_two_ify_censorlvl(censor[3], 0,NOISE)
 
 for m in model:
  print(len(m), len(m[0]))
 for c in censor:
  print(len(c), len(c[0]))
 
 print("-")
 
 for lr in [1]:
  print(lr)
  model = train_model(np.transpose(train_X), model, train_y, lr, 50 , censor, SEP)

 for n in range(10):
  print('-')
  print(n)
  print('-')
  model = train_model(np.transpose(train_X), model, train_y, 2, 100 , censor, SEP)
  
  preds = predict(np.transpose(test_X), model,False ,SEP)
  
  #preds, v = predict(np.transpose(test_X), model,True ,SEP)
  #print(v)
  
  testDf["PREDICTED"]=preds
  testDf["ACTUAL"]=testDf["GreenWin?"]
  
  print(metrics.get_gini(testDf, "PREDICTED","ACTUAL"))
  print(metrics.get_Xiles(testDf, "PREDICTED","ACTUAL"))

 #print(preds)
 print(sum(preds)/len(preds))

###################

print(model)

###################

#Okay, here's the fun stuff.

if True:
 
 preds, vecs = predict(np.transpose(train_X), model, True, SEP)

 #print(vecs, len(vecs))
 #print(vecs[2], len(vecs[2]))
 #print(vecs[2][1], len(vecs[2][1]))

 trainDf["PREDICTED"] = preds
 for n in range(len(vecs[2])):
  trainDf["neck_"+str(n)] = vecs[2][n]
 trainDf.to_csv("nodeyTrainDf.csv")
 print(trainDf.corr())

###################

