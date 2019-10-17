# Clearing and formatting code under PEP8
#
# CSV of train and test data were save by
# more older version of code.
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

START_TIME = time.time()  # Calculate time

TRAIN_DATA_SIZE = 2819
NUM_CLASSES = 13
NUM_EPOCHS = 50000  # Set a big enough number for running unstop
BATCH_SIZE = 13
LEARNING_RATE = 0.00001
TEST_DATA_SIZE = 1040

input_size = 1  # only one data-one label
hid_size1 = 64  # hidden layer size
hid_size2 = 192
hid_size3 = 384
hid_size4 = 256
hid_size5 = 256  # adjust depend on convolution result
hid_size6 = 256  # mlp hidden size
hid_size7 = 128

torch.cuda.empty_cache()  # For getting more memory for computation
torch.backends.cudnn.benchmark = True
gpu_accept_type = torch.cuda.FloatTensor


class train_dataset(data.Dataset):  # read train data
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values  # read csv
        self.data = np.asarray(pd_data[:, 1:])  # data
        # reshape to 4 dimension,
        # (numitem) * 1 *256*256(picture original shape)
        self.data = np.reshape(
            self.data, (-1, 1, 256, 256)).astype(np.float)
        print(self.data.shape)  # checking if correct
        print(self.data.dtype)
        self.label = np.asarray(
            pd_data[:, 0:1]).astype(np.float)  # label
        print(self.label.shape)  # checking if correct
        self.length = self.data.shape[0]

    def __len__(self):
        return self.length  # dataset class require

    def __getitem__(self, index):
        # dataset class require
        return torch.Tensor(
            self.data[index]), torch.Tensor(self.label[index])


class test_dataset(data.Dataset):  # read train data
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values  # read csv
        self.data = np.asarray(pd_data[:, 0:])
        # reshape to 4 dimension,
        # (numitem) * 1 *256*256(picture original shape)
        self.data = np.reshape(
            self.data, (-1, 1, 256, 256)).astype(np.float)
        self.length = self.data.shape[0]

    def __len__(self):
        return self.length  # dataset class require

    def __getitem__(self, index):
        return torch.Tensor(self.data[index])  # dataset class require


print('train')
# create a item of class(which would read csv when initial)
traindata = train_dataset('train.csv')
print('test')
testdata = test_dataset('test.csv')
trainloader = data.DataLoader(
    traindata, batch_size=batch_size,
    num_workers=0, shuffle=True)  # initial data loader by class item
testloader = data.DataLoader(
    testdata, batch_size=test_size, num_workers=0)


class NetWork(nn.Module):  # network structure
    # read size when initial calss member
    # for more flexibility
    def __init__(
            self, input_size,
            hid_size1, hid_size2,
            hid_size3, hid_size4,
            num_classes):
        super(NetWork, self).__init__()
        self.conv1 = nn.Conv2d(
            input_size, hid_size1,
            kernel_size=11, stride=4, padding=2)  # convlution layer 1
        self.conv2 = nn.Conv2d(
            hid_size1, hid_size2,
            kernel_size=5, padding=2)  # convlution layer 2
        self.conv3 = nn.Conv2d(
            hid_size2, hid_size3,
            kernel_size=3, padding=1)  # convlution layer 3
        self.conv4 = nn.Conv2d(
            hid_size3, hid_size4,
            kernel_size=3, padding=1)  # convlution layer 4
        self.conv5 = nn.Conv2d(
            hid_size4, hid_size5,
            kernel_size=3, padding=1)  # convlution layer 5
        self.linear1 = nn.Linear(
            hid_size6, hid_size7)  # mlp layer 1
        self.linear2 = nn.Linear(
            hid_size7, hid_size7)  # mlp layer 2
        self.linear3 = nn.Linear(
            hid_size7, num_classes)  # mlp layer 3
        self.pool = nn.MaxPool2d(3, 2)  # maxpool
        self.norm = nn.BatchNorm2d(32)  # batch nomarlization
        self.d = nn.Dropout(0.5)  # rop 50 %

    def forward(self, x):
        hid_out1 = self.pool((
            F.relu(self.conv1(x))))  # cnn hidden output 1
        hid_out2 = self.pool((
            F.relu(self.conv2(hid_out1))))  # cnn hidden output 2
        hid_out3 = self.pool(
            F.relu(self.conv3(hid_out2)))  # cnn hidden output 3
        hid_out4 = F.relu(self.pool(
            self.conv4(hid_out3)))  # cnn hidden output 5
        hid_out5 = self.pool((
            F.relu(self.conv5(hid_out4))))  # cnn hidden output 6
        fcinput1 = hid_out5.view(
            x.size(0), -1)  # flatten cnn output for mlp layers
        lhid_out1 = self.linear1(F.relu(
            self.d(fcinput1)))  # mlp hidden output1
        lhid_out2 = self.linear2(F.relu(
            self.d(lhid_out1)))  # mlp hidden output2
        out = self.linear3(lhid_out2)  # hidden output3
        prob = F.softmax(out, dim=1)  # do softmax, then output
        return out


''' initial network class member, using cuda to accelerate '''
Model = NetWork(
    input_size, hid_size1,
    hid_size2, hid_size3,
    hid_size4, num_classes).cuda()

criterion = nn.CrossEntropyLoss(
    ).type(gpu_accept_type).cuda()  # Use crossentropy for loss function
optimizer = torch.optim.Adam(
    Model.parameters(), lr=LEARNING_RATE)  # Use adam as optimizer

print("Read:    --- %s seconds ---" % (
    time.time() - START_TIME))  # Print time for reading data
MID_TIME = time.time()

model_accuracy = []  # Model accuracy recorded for observation


for epoch in range(NUM_EPOCHS):
    print(epoch)
    train_loss = []
    batch_correct = []
    for train_batch_num, (data, label) in enumerate(
            trainloader):  # Batch input data from dataloader
        if(train_batch_num < (TRAIN_DATA_SIZE/BATCH_SIZE)-1):
            # Use long to avoid RuntimeError: Expected object of
            # scalar type Long but got scalar type Float for
            # argument #2 'target'
            label_long = Variable(
                label.long(), requires_grad=False).cuda()
            optimizer.zero_grad()  # Clear optimizer's grad

            ''' Get prediction from model '''
            model_output = Model(
                data.type(gpu_accept_type)).type(gpu_accept_type).cuda()
            loss = criterion(model_output, label_long)  # Get this epoch's loss
            loss.backward()  # Back propagation
            optimizer.step()  # Optimizer optimize
            # Duplicate label
            label_float = Variable(
                label.float(), requires_grad=False).type(
                    gpu_accept_type).cuda()
            train_loss.append(loss.item())  # For calculate loss
            Model.eval()  # Evaluation mode
            correct_item = (
                model_output == label_float).sum()  # Calculate correct 'items'
            # calculate mean, must use float instead of long
            batch_correct.append((torch.sum(
                torch.argmax(
                    model_output, dim=1) == label_long) * 100).float())

            ''' Output current model prediction in a period '''
            if train_batch_num == 214:  # Near the last epoch
                model_accuracy.append(np.mean(train_loss))
                # Count the correct classification
                correct_item = torch.sum(
                    torch.argmax(model_output, dim=1) == label_long)
                # For our observation
                print(
                    'Epoch: [%d/%d], Loss: %.4f, Accuracy: %.2f' % (
                        epoch+1, NUM_EPOCHS,
                        np.mean(train_loss), (torch.mean(torch.stack(
                            batch_correct), dim=0) / model_output.shape[0])))

                ''' Save current model prediction in a period '''
                if (epoch+1) % 5 == 0 and (epoch) != 0:  # Every five epoch
                    with torch.no_grad():  # Disable auto-grad
                        # Copy model from current model and
                        # set it on cuda, for prevent to much thing in gpu
                        Model_CPU = copy.deepcopy(Model)
                        Model_CPU = Model_CPU.cpu()  # Set model on cpu

                        ''' Load test data to predict '''
                        for test_batch_num, (
                                data, label) in enumerate(testloader):
                            # Transfer format to Variable for network
                            data = Variable(data, requires_grad=False).cpu()
                            # Get prediction of test data
                            model_output = Model_CPU(data).cpu()
                            model_output = model_output.detach()  # Flat
                            # Get predict label
                            out, index = torch.max(model_output, 1)
                            print(out)
                            print(index)
                            index_cpu = index.cpu()  # Put in cuda
                            # Change format prepare for observation and output
                            index_cpu = index_cpu.numpy()
                            predict_label = index_cpu  # Copy

                        # Transform label back into string
                        predict_label = predict_label.astype(np.str_)
                        predict_label = np.char.replace(
                            predict_label, '10', 'street')
                        predict_label = np.char.replace(
                            predict_label, '11', 'suburb')
                        predict_label = np.char.replace(
                            predict_label, '12', 'tallbuilding')
                        predict_label = np.char.replace(
                            predict_label, '0', 'bedroom')
                        predict_label = np.char.replace(
                            predict_label, '1', 'coast')
                        predict_label = np.char.replace(
                            predict_label, '2', 'forest')
                        predict_label = np.char.replace(
                            predict_label, '3', 'highway')
                        predict_label = np.char.replace(
                            predict_label, '4', 'insidecity')
                        predict_label = np.char.replace(
                            predict_label, '5', 'kitchen')
                        predict_label = np.char.replace(
                            predict_label, '6', 'livingroom')
                        predict_label = np.char.replace(
                            predict_label, '7', 'mountain')
                        predict_label = np.char.replace(
                            predict_label, '8', 'office')
                        predict_label = np.char.replace(
                            predict_label, '9', 'opencountry')

                        ''' Output prediction '''
                        testbuilding_name = [
                            'image_%04d' % n for n in range(0, 1039+1)]
                        predict_label_dataframe = pd.DataFrame({
                            "id": testbuilding_name, "label": predict_label})
                        # Output result to csv
                        predict_label_dataframe.to_csv(
                            "CS_IOC5008_0856619_HW1(%d).csv" % (epoch+1),
                            columns=["id", "label"], index=False)

                        ''' Output recorded accuracy '''
                        accuracy_array = np.asarray(
                            model_accuracy).ravel()  # Save accuracy
                        accuracy_dataframe = pd.DataFrame({
                            "samp_id": range(1, len(accuracy_array)+1),
                            "acc": accuracy_array})
                        # Output result to csv for observatioN
                        accuracy_dataframe.to_csv(
                            'hidobs(%d).csv' % (epoch+1),
                            columns=["samp_id", "loss", "acc"], index=False)

                        ''' Save current Model '''
                        torch.save(Model, 'hid_net(%d).pt' % ((epoch+1)))

                        # Print total use time, originally use for
                        # observe performance
                        print("--- %s seconds ---" % (
                            time.time() - START_TIME))

# Print time use for training model
print("Train:   --- %s seconds ---" % (
    time.time() - MID_TIME))
MID_TIME2 = time.time()

# These part were used for output
# when training process is done, which is
# reach the last epoch.
# But we can stop training in early stage
# if we think it's alreay converge.
#
# Put model on cpu when we are going to
# ouptut result to save moemory space
Model = Model.type(gpu_accept_type).cpu()

''' Output final prediction '''
with torch.no_grad():  # Disable auto-grad
    for test_batch_num, (data) in enumerate(testloader):
        # Change data format
        data = Variable(
            data, requires_grad=False).type(gpu_accept_type).cpu()
        # Get prediction
        model_output = Model(data).type(gpu_accept_type).cpu()
        model_output = model_output.detach()  # Flatten
        out, index = torch.max(model_output, 1)
        index_cpu = index.type(gpu_accept_type).cpu()  # Put in cuda
        index_cpu = index_cpu.numpy()  # Change format for output
        predict_label = index_cpu

# Print time model use for doing prediction
print("Predict:--- %s seconds ---" % (time.time() - MID_TIME2))

# change back to building name
predict_label = predict_label.astype(np.str_)
predict_label = np.char.replace(predict_label, '10', 'street')
predict_label = np.char.replace(predict_label, '11', 'suburb')
predict_label = np.char.replace(predict_label, '12', 'tallbuilding')
predict_label = np.char.replace(predict_label, '0', 'bedroom')
predict_label = np.char.replace(predict_label, '1', 'coast')
predict_label = np.char.replace(predict_label, '2', 'forest')
predict_label = np.char.replace(predict_label, '3', 'highway')
predict_label = np.char.replace(predict_label, '4', 'insidecity')
predict_label = np.char.replace(predict_label, '5', 'kitchen')
predict_label = np.char.replace(predict_label, '6', 'livingroom')
predict_label = np.char.replace(predict_label, '7', 'mountain')
predict_label = np.char.replace(predict_label, '8', 'office')
predict_label = np.char.replace(predict_label, '9', 'opencountry')

testbuilding_name = [
    'image_%04d' % n for n in range(0, 1039+1)]
predict_label_dataframe = pd.DataFrame({
    "id": testbuilding_name, "label": predict_label})
predict_label_dataframe.to_csv(
    "CS_IOC5008_0856619_HW1.csv",
    columns=["id", "label"], index=False)  # Output result to csv

torch.save(Model, 'net.pkl')  # Save model
# Print total use time, originally use for observe performance
print("--- %s seconds ---" % (time.time() - START_TIME))
