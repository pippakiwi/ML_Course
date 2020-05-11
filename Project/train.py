import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import lr_schedule
import networks
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--batchsize', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--GPU', default=True, help='train with GPU')
parser.add_argument('--epoch', default=304, type=int, help='epoch')
parser.add_argument('--data_path', default='data')
parser.add_argument('--test_iterval',default=30,type=int)
parser.add_argument('--gpu_id',default="1")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
best_test_acc = 0
test_acc = 0
transform_train = transforms.Compose(
    [
        # transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                            std=[0.229, 0.224, 0.225])
    ]
)

transform_test = transforms.Compose(
    [
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ]
)

train_data = torchvision.datasets.ImageFolder(
    os.path.join(args.data_path, 'train'), transform=transform_train
)

test_data = torchvision.datasets.ImageFolder(
    os.path.join(args.data_path, 'test'), transform=transform_test
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batchsize, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=1, shuffle=False
)


# net = networks.ResNetFc(resnet_name='ResNet18', class_num=7)
net = networks.VGGFc(vgg_name='VGG16', class_num=7)

if use_cuda:
    net.cuda()
params = net.get_parameters()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

loss_z = []
loss_ori = []
test_error = []
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
def train(Epoch):
    for i, (inputs, target) in enumerate(train_loader):
        global loss_z
        global  loss_ori
        net.train()

        inputs, target = Variable(inputs), Variable(target)
        if use_cuda:
            inputs = inputs.to(device)
            target = target.to(device)
        outputs = net(inputs)

        # lr_scheduler = lr_schedule.schedule_dict["inv"]
        #         # schedule_param = {"lr": args.lr, "gamma": 0.001, "power": 0.75}
        #         # optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        loss = criterion(outputs, target)
        softmax_out = nn.Softmax(dim=1)(outputs)
        loss_entropy = torch.mean(Entropy(softmax_out))
        total_loss = loss + 0.1 * loss_entropy
        total_loss.backward()
        optimizer.step()
        loss_ori.append(loss.item())
        if i == 0 and Epoch:
            print("iter:{}, loss_classi:{:4f}".format(Epoch-1, np.mean(loss_ori) ))
            loss_z.append(np.mean(loss_ori))
            loss_ori = []
            print(loss_z)
def test(Epoch):
    global test_acc
    global best_test_acc
    net.eval()
    total = 0
    correct = 0

    # matric_pred = []
    # matric_corr = []


    for i, (inputs, target) in enumerate(test_loader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, target = Variable(inputs), Variable(target)
        if use_cuda:
            inputs = inputs.to(device)
            target = target.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

        # if i % 46 == 0:
        #     print(i)
        #     print(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).sum()

        # pred_1 = list(predicted.view_as(target.data).cpu().detach().numpy())
        # corr_1  = list(target.data.cpu().detach().numpy())
        # matric_pred += pred_1
        # matric_corr += corr_1


    test_acc = 100.0 * int(correct.data) / total

    # matric_pred = np.array(matric_pred)
    # matric_corr = np.array(matric_corr)
    # confusion_matrix1 = confusion_matrix(matric_pred, matric_corr)
    # print (confusion_matrix1)
    test_error.append(100.0-test_acc)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
    print('Epoch:{}, Testing acc is {:4f}%, best acc is {:4f}%'.format(Epoch, test_acc, best_test_acc))
    print(test_error)



if __name__ == '__main__':
    for epoch in range(args.epoch):
        train(epoch)
        if epoch % args.test_iterval == 0 and epoch:
            test(epoch)
