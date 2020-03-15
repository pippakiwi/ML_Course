# import sys
# path='C://User//youfu//Desktop//machine_learning//SVM//libsvm-3.24//python'
# sys.path.append(path)
from svmutil import *
train_label,train_pixel = svm_read_problem('../heart_scale')
model = svm_train(train_label[:200],train_pixel[:200],'-c 4')
print("result:")
p_label, p_acc, p_val = svm_predict(train_label[200:], train_pixel[200:], model);
print(p_acc)
