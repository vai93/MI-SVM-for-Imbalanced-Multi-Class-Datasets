from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import time
from dataPreProcessing import *     
from MISVM import *
for dataset in range(1,12): 
    X= eval("data.X"+str(dataset))
    y= eval("data.y"+str(dataset))
    for c in range(10): 
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33,stratify=y, random_state=c)
        start_time = time.time()
        model=misvm.fit(train_x, train_y)
        elapsed_time = time.time() - start_time
        preClass,elapsed_time_test=model.predict(test_x)
        print(round(f1_score(test_y, preClass,average='weighted'),4))
        print(round(precision_score(test_y, preClass,average='weighted'),4))
        print(round(accuracy_score(test_y, preClass),4))
        print(round(recall_score(test_y, preClass,average='weighted'),4))
        print(round(elapsed_time,3))
        print(round(elapsed_time_test,3) )    