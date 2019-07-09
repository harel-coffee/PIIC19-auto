from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix
from sklearn.preprocessing import normalize
import time
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, target_names, title='Confusion matrix (f1-score)',cmap=None, normalize=True):
    
    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
def calculate_metrics(y_true,y_pred,plot=True, title=""):
    dic_return = {}
    dic_return["Precision raw"] = precision_score(y_true,y_pred,average=None,labels=[0,1])
    dic_return["Recall raw"] = recall_score(y_true,y_pred,average=None,labels=[0,1])
    #dic_return["Precision"] = precision_score(y_true,y_pred,average=None,labels=[0,1])
    #dic_return["Recall"] = recall_score(y_true,y_pred,average=None,labels=[0,1])
    dic_return["F1 raw"] = f1_score(y_true,y_pred,average=None,labels=[0,1])
    dic_return["F1 weighted"] = f1_score(y_true,y_pred,average="weighted",labels=[0,1])
    dic_return["F1 macro"] = f1_score(y_true,y_pred,average="macro",labels=[0,1])
    dic_return["F1 micro"] = f1_score(y_true,y_pred,average="micro",labels=[0,1])
    matriz=confusion_matrix(y_true,y_pred)
    if plot:
        df = pd.DataFrame(dic_return)
        df.index = ["False Positive","Confirmed"]
        print(df)
        plot_confusion_matrix(normalize(matriz,axis=1,norm='l1'),["False Positive","Confirmed"],title)
    dic_return["Confusion Matrix"] = matriz
    return dic_return


"""
import matplotlib.pyplot as plt
learners = len(p_test)
M = np.arange(learners)
LABELS=["K-NN","SVM","RANDOM FOREST"]

fig = plt.figure(figsize=(12,8))

#PRECISION SCORES
aux = list(map(list, zip(*p_test))) #transpose
plt.bar(M-0.15, aux[0], width=0.3,facecolor='#ff0000', edgecolor='white',label="False Positive")
plt.bar(M+0.15, aux[1], width=0.3,facecolor='#1C9900', edgecolor='white',label="Confirmed")

#RECALL SCORES
aux = list(map(list, zip(*r_test))) #transpose
plt.bar(M-0.15, np.array(aux[0])*-1, width=0.3,facecolor='#FF6666', edgecolor='white')
plt.bar(M+0.15, np.array(aux[1])*-1, width=0.3,facecolor='#76C166', edgecolor='white')

#ANOTATIONS OF SCORES
for x, (a,b) in zip(M, p_test):
    plt.text(x + 0.02-0.15, a + 0.01, '%.3f' % a, ha='center', va='bottom')#fp
    plt.text(x + 0.02+0.15, b + 0.01, '%.3f' % b, ha='center', va='bottom')#conf
for x, (a,b) in zip(M, r_test):
    plt.text(x + 0.02-0.15, -a - 0.01, '%.3f' % a, ha='center', va='top')#fp
    plt.text(x + 0.02+0.15, -b - 0.01, '%.3f' % b, ha='center', va='top')#conf
        
plt.xticks(M, LABELS)
plt.title("PRECISION & RECALL on TEST")   
plt.xlabel("Learners")  
plt.ylabel("RECALL - Score - PRECISION") 
plt.legend()
plt.ylim(-1.2,1.2)
plt.show()
fig = plt.figure(figsize=(12,4))
plt.bar(M, f1_score_test, facecolor='#9999ff', edgecolor='white')

for x, y in zip(M, f1_score_test):
    plt.text(x + 0.02, y + 0.01, '%.3f' % y, ha='center', va='bottom')
    
plt.xticks(np.arange(learners), LABELS)
plt.title("F1-SCORE on TEST")   
plt.xlabel("Learners")  
plt.ylabel("Score") 
plt.ylim(0,1.2)
plt.show()
"""