from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

def calculate_metrics(y_true,y_pred,plot=True):
    dic_return = {}
    dic_return["Precision raw"] = precision_score(y_true,y_pred,average=None,labels=[0,1])
    dic_return["Recall raw"] = recall_score(y_true,y_pred,average=None,labels=[0,1])
    #dic_return["Precision"] = precision_score(y_true,y_pred,average=None,labels=[0,1])
    #dic_return["Recall"] = recall_score(y_true,y_pred,average=None,labels=[0,1])
    dic_return["F1 raw"] = f1_score(y_true,y_pred,average=None,labels=[0,1])
    dic_return["F1 weighted"] = f1_score(y_true,y_pred,average="weighted",labels=[0,1])
    dic_return["F1 macro"] = f1_score(y_true,y_pred,average="macro",labels=[0,1])
    dic_return["F1 micro"] = f1_score(y_true,y_pred,average="macro",labels=[0,1])
    if plot:
        df = pd.DataFrame(dic_return)
        df.index = ["False Positive","Confirmed"]
        print(df)
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