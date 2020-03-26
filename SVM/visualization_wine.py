import sys
path = 'C://User//youfu//Desktop//machine_learning//SVM//libsvm-3.23//python'
sys.path.append(path)
from svmutil import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import *

class_1_number = 59
class_2_number = 71


data = pd.read_csv("wine.csv")
y = data.iloc[:,0]
x = data.iloc[:,1:]


x = maxabs_scale(x)


tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(x)
x_tsne_class1 = x_tsne[ : class_1_number]
x_tsne_class2 = x_tsne[class_1_number : class_1_number + class_2_number]
x_tsne_class3 = x_tsne[class_1_number + class_2_number :]
plt.plot(x_tsne_class1[:,0], x_tsne_class1[:,1], 'r.')
plt.plot(x_tsne_class2[:,0], x_tsne_class2[:,1], 'go')
plt.plot(x_tsne_class3[:,0], x_tsne_class3[:,1], 'b*')
plt.show()
# y,x = svm_read_problem("wine_scale.txt")

# # plt.plot(x,y)
# acc = []
# for i in range(20):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#     model = svm_train(y_train,x_train)
#     p_label, p_acc, p_val = svm_predict(y_test,x_test,model)
#     acc.append(max(p_acc))
# print(acc)
# acount = sum(acc)
# average = acount / len(acc)
# print(average)
# print(p_label)
# print(p_acc)
# print(p_val)

