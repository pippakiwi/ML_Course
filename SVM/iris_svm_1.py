import sys
path = 'C://User//youfu//Desktop//machine_learning//SVM//libsvm-3.23//python'
sys.path.append(path)
from svmutil import *
from sklearn.model_selection import train_test_split

y,x = svm_read_problem("preprocessed_iris.txt")
acc = []

for i in range(20):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = svm_train(y_train,x_train,'-t 2 -c 32 -g 0.0078')
    p_label, p_acc, p_val = svm_predict(y_test,x_test,model)
    acc.append(max(p_acc))

acc_sum = sum(acc)
acc_average = acc_sum / len(acc)
print(acc_average)

