import sys
path = 'C://User//youfu//Desktop//machine_learning//SVM//libsvm-3.23//python'
sys.path.append(path)
from svmutil import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import scale

class_1_number = 59
class_2_number = 71
y,x = svm_read_problem("wine_scale.txt")
acc = []
for i in range(20):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = svm_train(y_train,x_train,"-t 1")
    p_label, p_acc, p_val = svm_predict(y_test,x_test,model)
    acc.append(max(p_acc))
print(acc)
acc_sum = sum(acc)
acc_average = acc_sum / len(acc)
print(acc_average)

