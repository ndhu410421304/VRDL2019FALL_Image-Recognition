# MODIFY pep8
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import copy as copy
import time
import os

start_time = time.time()  # calculate time

TRAIN_DATA_SIZE = 2819
NUM_CLASSES = 13
NUM_EPOCHS = 50000  # set a big enough number for running unstop
BATCH_SIZE = 13
LEARNING_RATE = 0.00001
TEST_DATA_SIZE = 1040
ori_acc = 0  # for comparison

torch.cuda.empty_cache()  # for getting more memory for computation
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

data_transforms = {
    'train': transforms.Compose([
        # Doing data augmentation here
        # transforms.RandomRotation(5) can also apply if want
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print('train')

data_dir = 'dataset'
traindata = datasets.ImageFolder(
    os.path.join(data_dir, 'train'), data_transforms['train'])
testdata = datasets.ImageFolder(
    os.path.join(data_dir, 'test'), data_transforms['test'])
trainloader = torch.utils.data.DataLoader(
    traindata, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(
    testdata, batch_size=TEST_DATA_SIZE, shuffle=False, num_workers=0)
class_names = traindata.classes

model2 = models.densenet201(pretrained=True).cuda()
features = model2.classifier.in_features
model2.classifier = nn.Sequential(
    nn.Linear(1920, 960), nn.Linear(960, 480),
    nn.Linear(480, 13)).type(dtype).cuda()
print(model2)

criterion = nn.CrossEntropyLoss(
    ).type(dtype).cuda()  # use crossentropy for loss function
optimizer = torch.optim.Adam(
    model2.parameters(), lr=LEARNING_RATE)  # use adam as optimizer

print("Read:    --- %s seconds ---" % (
    time.time() - start_time))  # print time for reading data
mid_time = time.time()

haccs = []  # hid accs, for observation
hloss = []  # hid loss, for observation
for epoch in range(NUM_EPOCHS):
    print(epoch)
    trainloss = []
    batchcorrect = []
    for i, (data, label) in enumerate(
            trainloader):  # batch input data from dataloader
        if(i < (TRAIN_DATA_SIZE/BATCH_SIZE)-1):
            # use long to avoid RuntimeError: Expected object of
            # scalar type Long but got scalar type Float for
            # argument #2 'target'
            labels = Variable(
                label.long(), requires_grad=False).cuda()
            optimizer.zero_grad()  # clear optimizer's grad
            # get prediction from model
            outputs = model2(data.type(dtype)).type(dtype).cuda()
            loss = criterion(outputs, labels)  # get this epoch's loss
            loss.backward()  # back propagation
            optimizer.step()  # optimizer optimize
            # duplicate label
            labelc = Variable(
                label.float(), requires_grad=False).type(dtype).cuda()
            trainloss.append(loss.item())  # for calculate loss
            model2.eval()  # evaluation mode
            correct = (outputs == labelc).sum()  # calculate correct 'items'
            # calculate mean, must use float instead of long
            batchcorrect.append((torch.sum(
                torch.argmax(outputs, dim=1) == labels) * 100).float())

            if i == 214:  # 2800 / 300 = 9.xx > 9
                haccs.append(np.mean(trainloss))
                hloss.append((correct * 100 / outputs.shape[0]))
                # count the correct classification
                correct = torch.sum(torch.argmax(outputs, dim=1) == labels)
                # for our observation
                print(
                    'Epoch: [%d/%d], Loss: %.4f, Accuracy: %.2f' % (
                        epoch+1, NUM_EPOCHS,
                        np.mean(trainloss), (torch.mean(torch.stack(
                            batchcorrect), dim=0) / outputs.shape[0])))
                if (epoch+1) % 10 == 0 and (epoch) != 0:  # last epoch in loop
                    with torch.no_grad():  # disable auto-grad
                        # copy model from current model and
                        # set it on cuda, for prevent to much thing in gpu
                        model3 = copy.deepcopy(model2)
                        model3 = model3.cpu()  # set on cpu
                        # load test data #drop label
                        for i, (data, label) in enumerate(testloader):
                            # transfer format to Variable for network
                            data = Variable(data, requires_grad=False).cpu()
                            # get prediction of test data
                            outputs = model3(data).cpu()
                            outputs = outputs.detach()  # flat
                            # get predict label
                            out, index = torch.max(outputs, 1)
                            print(out)
                            print(index)
                            indexc = index.cpu()  # put in cuda
                            # change format prepare for observation and output
                            indexc = indexc.numpy()
                            testy = indexc  # copy
                        # transfer label back into string
                        testy = testy.astype(np.str_)
                        testy = np.char.replace(testy, '10', 'street')
                        testy = np.char.replace(testy, '11', 'suburb')
                        testy = np.char.replace(testy, '12', 'tallbuilding')
                        testy = np.char.replace(testy, '0', 'bedroom')
                        testy = np.char.replace(testy, '1', 'coast')
                        testy = np.char.replace(testy, '2', 'forest')
                        testy = np.char.replace(testy, '3', 'highway')
                        testy = np.char.replace(testy, '4', 'insidecity')
                        testy = np.char.replace(testy, '5', 'kitchen')
                        testy = np.char.replace(testy, '6', 'livingroom')
                        testy = np.char.replace(testy, '7', 'mountain')
                        testy = np.char.replace(testy, '8', 'office')
                        testy = np.char.replace(testy, '9', 'opencountry')
                        # save per 100
                        testbuilding_name = [
                            'image_%04d' % n for n in range(0, 1039+1)]
                        output = pd.DataFrame({
                            "id": testbuilding_name, "label": testy})
                        # output result to csv
                        output.to_csv(
                            "CS_IOC5008_0856619_HW1(%d).csv" % (epoch+1),
                            columns=["id", "label"], index=False)
                        # save data also
                        haccsarray = np.asarray(haccs).ravel()  # save accuracy
                        hlossarray = np.asarray(hloss).ravel()  # save loss
                        output2 = pd.DataFrame({
                            "samp_id": range(1, len(haccsarray)+1),
                            "loss": hlossarray, "acc": haccsarray})
                        # output result to csv for observatioN
                        output2.to_csv(
                            'hidobs(%d).csv' % (epoch+1),
                            columns=["samp_id", "loss", "acc"], index=False)

                        # save model
                        torch.save(model2, 'hid_net(%d).pt' % ((epoch+1)))
                        # print total use time, originally use for
                        # observe performance
                        print("--- %s seconds ---" % (
                            time.time() - start_time))

# print time use for training model
print("Train:   --- %s seconds ---" % (
    time.time() - mid_time))
mid_time2 = time.time()

# these part were originally use for output, but
# when we do infinite we dont need this
#
# put model on cuda when we are going to
# ouptuT result to save moemory space
model2 = model2.type(dtype).cuda()

# basically same as output part in the infinite loop
with torch.no_grad():  # disable auto-grad
    for i, (data) in enumerate(testloader):
        # change data format
        data = Variable(
            data, requires_grad=False).type(dtype).cuda()
        # get prediction
        outputs = model2(data).type(dtype).cuda()
        outputs = outputs.detach()  # flatten
        out, index = torch.max(outputs, 1)
        indexc = index.type(dtype).cuda()  # put in cuda
        indexc = indexc.numpy()  # change format for output
        testy = indexc

# print time model use for doing prediction
print("Predict:--- %s seconds ---" % (time.time() - mid_time2))

# change back to building name
testy = testy.astype(np.str_)
testy = np.char.replace(testy, '10', 'street')
testy = np.char.replace(testy, '11', 'suburb')
testy = np.char.replace(testy, '12', 'tallbuilding')
testy = np.char.replace(testy, '0', 'bedroom')
testy = np.char.replace(testy, '1', 'coast')
testy = np.char.replace(testy, '2', 'forest')
testy = np.char.replace(testy, '3', 'highway')
testy = np.char.replace(testy, '4', 'insidecity')
testy = np.char.replace(testy, '5', 'kitchen')
testy = np.char.replace(testy, '6', 'livingroom')
testy = np.char.replace(testy, '7', 'mountain')
testy = np.char.replace(testy, '8', 'office')
testy = np.char.replace(testy, '9', 'opencountry')

testbuilding_name = [
    'image_%04d' % n for n in range(0, 1039+1)]
output = pd.DataFrame({
    "id": testbuilding_name, "label": testy})
output.to_csv(
    "CS_IOC5008_0856619_HW1.csv",
    columns=["id", "label"], index=False)  # output result to csv

torch.save(model2, 'net.pkl')  # save model
# print total use time, originally use for observe performance
print("--- %s seconds ---" % (time.time() - start_time))
