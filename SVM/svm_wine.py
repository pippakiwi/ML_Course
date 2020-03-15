import sys
path = 'C://User//youfu//Desktop//machine_learning//SVM//libsvm-3.23//python'
sys.path.append(path)
from svmutil import *
y,x = svm_read_problem("wine.txt")
model = svm_train(y,x)
p_label, p_acc, p_val = svm_predict(y,x,model)
# print(p_label)
# print(p_acc)
# print(p_val)

