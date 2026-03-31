from sklearn import svm
import pandas as pd
import numpy as np
import time
from sklearn.svm import OneClassSVM
from collections import defaultdict   
from sklearn.model_selection import GridSearchCV 
class MISVM:
    def mct(self,x,sv):
        x=np.array(x)
        sv=np.array(sv)
        plus=np.count_nonzero(sv[:, -1] == 1)
        neg=np.count_nonzero(sv[:, -1] == -1)
        tau_n=plus/neg
        tau_p=neg/plus
        tx=[0 for col in range(x.shape[0])]
        j=0
        for i in range(x.shape[0]):
            if(i in sv[:, 0]): 
                if(sv[:, -1][j]==-1):
                    tx[i]=tau_n
                    j=j+1
                else: 
                    tx[i]=tau_p
                    j=j+1
            else:
                tx[i]=1 
        tx=np.array(tx)
        return tx                     
    def euclidean_dist_matrix(data_1, data_2):
        norms_1 = (data_1 ** 2).sum(axis=1)
        norms_2 = (data_2 ** 2).sum(axis=1)
        return np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))
    def rbpk(x,y):
        sigma=float(x.shape[1])*x.var()
        dists_sq= np.exp((np.dot(x,y.T)+1)**3/sigma**2)
        dists_sq = MISVM.euclidean_dist_matrix(x, y)
        return np.exp(-(dists_sq / sigma)) 
    def __init__(self):
        self.train_x=None
        self.train_y=None
        self.test_x=None
    def KTI(self,x,y,sv,dc,svi):
        sv=np.column_stack((svi,sv, np.sign(dc)[0]))
        tn=pd.DataFrame(np.sign(dc)[0]).value_counts().to_numpy()
        t1=(tn[0]/tn[1]*2)-1
        if(t1>tn[0]):
            t1=tn[0]/2
        t2=pd.DataFrame(np.sign(dc)[0]).value_counts().index.tolist()[0]
        ind=np.random.choice(np.where(sv[:, -1] == t2)[0], t1.astype(int), replace=False)
        sv=np.delete(sv, ind, axis=0)
        a=self.mct(x,sv)
        b = a if np.array_equal(x, y) else self.mct(y, sv)
        a.shape = (a.shape[0], 1)
        b.shape=(1,b.shape[0])
        a=a*b
        return np.multiply(a,MISVM.rbpk(x,y))
    def train_multiclass_svdd(X, y, n, g):
        classes = np.unique(y)
        models = {}
        boundary_points = defaultdict(list)
        for cls in classes:
            X_cls = X[y == cls]
            model = OneClassSVM(nu=n, kernel="rbf", gamma=g)
            model.fit(X_cls)
            support_vectors = model.support_vectors_
            models[cls] = model
            boundary_points[cls] = support_vectors
        return boundary_points 
    def training(self,temp,train_x,train_y,Xborders):
        if(len(temp)>1): 
            train_x1=train_x[(train_y==temp[0]) | (train_y==temp[1])]
            train_y1=train_y[(train_y==temp[0]) | (train_y==temp[1])]
            if(str(temp[0])+str(temp[1]) not in self.cpair):
                svm=svm.SVC(kernel="precomputed") 
                grid = GridSearchCV(svm, self.param_grid, scoring='recall', cv=5)
                gram_train =MISVM.rbpk(train_x1, train_x1)
                svm1=grid.fit(gram_train, train_y1)
                support_vectors_ = train_x1[svm1.support_]
                gram_train = self.KTI(train_x1, train_x1, support_vectors_,svm1.dual_coef_,svm1.support_)
                m=svm.SVC(kernel="precomputed")
                m=m.fit(gram_train, train_y1)
                self.svmtemp[temp[0]][temp[1]]=m 
                self.cpair.append(str(temp[0])+str(temp[1]))
            else:
                m=self.svmtemp[temp[0]][temp[1]]
            g1=[temp[0]] 
            g2=[temp[1]]
            support_vectors_new = train_x1[m.support_]
            for i in temp[2:]: 
                gram_test = self.KTI(Xborders[i], train_x1,support_vectors_new,m.dual_coef_,m.support_)
                pred=pd.DataFrame(m.predict(gram_test)).value_counts().index.tolist()
                if(len(pred)==1):
                    if(pred[0][0]==temp[0]):
                        g1.append(i)
                    else:
                        g2.append(i)
                else:
                    g1.append(i)
                    g2.append(i)
            self.training(g1,train_x,train_y,Xborders)
            self.training(g2,train_x,train_y,Xborders)           
    def testing(self,test_x,i,t):
        m=self.svmtemp[int(self.cpair[t][0])][int(self.cpair[t][1])]
        test_x=test_x.reshape(1, -1)
        train_x1=self.train_x[(self.train_y==int(self.cpair[t][0])) | (self.train_y==int(self.cpair[t][1]))]  
        support_vectors_new = train_x1[m.support_]                       
        gram_test = self.KTI(test_x, train_x1,support_vectors_new,m.dual_coef_,m.support_)                                              
        self.preClass[i]=m.predict(gram_test)[0]            
        for j in range(t+1,len(self.cpair)):
            if(self.preClass[i]==int(self.cpair[j][0])): 
                t=j
                self.testing(test_x,i,t)
            else:
                return          
    def fit(self,train_x, train_y):
        self.train_x=train_x
        self.train_y=train_y
        self.param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10] }
        Xborders = MISVM.train_multiclass_svdd(self.train_x, self.train_y, 0.3, 0.1)
        temp=pd.DataFrame(self.train_y).value_counts().index.tolist()   
        temp=np.array(temp).flatten()
        self.svmtemp=[[0 for col in range(len(temp))] for row in range(len(temp))] 
        self.cpair=[] 
        self.train_x=train_x.to_numpy()
        self.training(temp,self.train_x,self.train_y,Xborders)
        return self
    def predict(self,test_x):
        self.test_x=test_x
        self.preClass = np.zeros(test_x.shape[0])
        self.test_x=self.test_x.to_numpy()
        elapsed_time_test=0
        for i in range(len(test_x)):
            start_time1 = time.time()
            self.testing(self.test_x[i],i,0)
            elapsed_time_test=elapsed_time_test+(time.time() - start_time1)
        elapsed_time_test=elapsed_time_test/len(test_x)
        return self.preClass,elapsed_time_test
misvm=MISVM()