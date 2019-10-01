#ResNet Version
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import copy as copy
import time
start_time = time.time() #calculate time

#train data size
num = 2819

#setup
input_size = 1 #only one data-one label
hid_size1 = 64 #hidden layer size
hid_size2 = 128
hid_size3 = 256
hid_size4 = 512
hid_size5 = 512
num_classes = 13 #total building type
num_epochs = 50000 #10 epoch as 1 unit
batch_size = 75
learning_rate = 0.00001
verbose = False 
test_size = 1040 #test data counts

torch.cuda.empty_cache() # for getting more memory for computation
torch.backends.cudnn.benchmark = True

ori_acc = 0 #for comparison

class train_dataset(data.Dataset): #read train data
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values #read csv
        self.data = np.asarray(pd_data[:,1:]) #data
        self.data = np.reshape(self.data, (-1, 1, 256, 256)).astype(np.float) #reshape to 4 dimension, (numitem) * 1 *256*256(picture original shape)
        print(self.data.shape) #checking if correct
        #self.data = self.data.convert_objects(convert_numeric=True)
        print(self.data.dtype)
        self.label = np.asarray(pd_data[:,0:1]).astype(np.float) #label
        print(self.label.shape) #checking if correct
        self.length = self.data.shape[0]
    
    def __len__(self):
        return self.length # dataset class require
    
    def __getitem__(self, index):
        return torch.Tensor(self.data[index]), torch.Tensor(self.label[index]) # dataset class require

class test_dataset(data.Dataset): #read train data
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values #read csv
        self.data = np.asarray(pd_data[:,0:])
        self.data = np.reshape(self.data, (-1, 1, 256, 256)).astype(np.float)  #reshape to 4 dimension, (numitem) * 1 *256*256(picture original shape)  
        self.length = self.data.shape[0]
    
    def __len__(self):
        return self.length # dataset class require
    
    def __getitem__(self, index):
        return torch.Tensor(self.data[index]) # dataset class require

print('train')
traindata = train_dataset('train.csv') # create a item of class(which would read csv when initial)
print('test')
testdata = test_dataset('test.csv')
trainloader = data.DataLoader(traindata, batch_size=batch_size,num_workers=0,shuffle=True) #initial data loader by class item
testloader = data.DataLoader(testdata,batch_size=test_size,num_workers=0)

def conv3x3(input_channels, output_channels, stride=1):
    return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class res_block(nn.Module):
    def __init__(self, input_channels, output_channels, same_shape=True):
        super(res_block, self).__init__()
        self.same_shape = same_shape
        self.stride=1 if self.same_shape else 2
        self.conv1 = conv3x3(input_channels, output_channels, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = conv3x3(output_channels, output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=self.stride)
        
    def forward(self, x):
        hid_out1 = F.relu(self.bn1(self.conv1(x)), True)
        hid_out2 = F.relu(self.bn2(self.conv2(hid_out1)), True)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+hid_out2, True)

class ResNet(nn.Module):
    def __init__(self, input_channels, hid_size1, hid_size2, hid_size3, hid_size4, hid_size5, num_classes, verbose=False):
        super(ResNet, self).__init__()
        self.verbose = verbose
        self.pool = nn.MaxPool2d(3, 2)
        self.res_block1 = nn.Conv2d(input_channels, hid_size1, 7, 2)
        self.res_block2 = nn.Sequential(
            self.pool,
            res_block(hid_size1, hid_size1),
            res_block(hid_size1, hid_size1)
        )
        self.res_block3 = nn.Sequential(
            self.pool,
            res_block(hid_size1, hid_size2, False),
            res_block(hid_size2, hid_size2)
        )
        self.res_block4 = nn.Sequential(
            res_block(hid_size2, hid_size3, False),
            res_block(hid_size3, hid_size3),
        )
        self.res_block5 = nn.Sequential(
            res_block(hid_size3, hid_size4, False),
            res_block(hid_size4, hid_size4),
            nn.AvgPool2d(3)
        )
        self.classifier = nn.Linear(hid_size5, num_classes)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

model2 = ResNet(input_size, hid_size1, hid_size2, hid_size3, hid_size4, hid_size5, num_classes, verbose).cuda() #initial network class member, using cuda to accelerate
criterion = nn.CrossEntropyLoss()  # use crossentropy for loss function
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate) #se adam as optimizer

print("Read:    --- %s seconds ---" % (time.time() - start_time)) #print time use fir reading data(include pca trasform)
mid_time = time.time()

ep10 = 0
haccs = [] #hid accs, for observation
hloss = [] #hid loss, for observation
for epoch in range(num_epochs):
    print(epoch)
    for i, (data, label) in enumerate(trainloader): #batch input data from dataloader
        trainloss = []
        
        data = Variable(data.view(-1,1,256,256),requires_grad=False).cuda() #reshape data we get from data loader for cnn
        labels = Variable(label.long(),requires_grad=False).cuda() #use long to avoid RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target'
        optimizer.zero_grad() #clear optimizer's grad
        outputs = model2(data) #get prediction from model
        #print(outputs.shape[0])
        
        labels = labels.view(outputs.shape[0]).cuda() #for batch size beside set ones

        loss = criterion(outputs, labels) #get this epoch's loss
        loss.backward() #back propagation
        optimizer.step() #optimizer optimize
        
        labelc = label.cuda() #duplicate label
        
        trainloss.append(loss.item()) #for calculate loss
        
        model2.eval() #evaluation mode
        correct = (outputs == labelc).sum() #calculate correct items

        if i == 9: #2800 / 300 = 9.xx > 9
            correct = torch.sum(torch.argmax(outputs,dim=1)==labels) # count the correct classification
            print ('Epoch: [%d/%d], Loss: %.4f, Accuracy: %.2f'
                    % (epoch + 1 + 10*ep10, num_epochs + 10*ep10, np.mean(trainloss), (correct * 100 / outputs.shape[0]))) #for our observation
            haccs.append(np.mean(trainloss))
            hloss.append((correct*100 / outputs.shape[0]))
            if (epoch)%100 == 0 and (epoch) != 0: #last epoch in one loop
                with torch.no_grad(): # disable auto-grad
                    model3 = copy.deepcopy(model2) #copy model from current model and set it on CPU, for prevent to much thing in gpu
                    model3 = model3.cpu() #set on cpu
                    for i, (data) in enumerate(testloader): #load test data
                        data = Variable(data,requires_grad=False).cpu() #transfer format to Variable for network
                        
                        outputs = model3(data).cpu() #get prediction of test data
                        outputs = outputs.detach() #flat 
                        out, index = torch.max(outputs,1) #get predict label
                        indexc = index.cpu() #put in cpu
                        indexc = indexc.numpy() #change format prepare for observation and output
                        testy = indexc #copy
                    testy = testy.astype(np.str_) #trransfer label back into string
                    testy = np.char.replace(testy,'10','street')
                    testy = np.char.replace(testy,'11','suburb')
                    testy = np.char.replace(testy,'12','tallbuilding')
                    testy = np.char.replace(testy,'0','bedroom')
                    testy = np.char.replace(testy,'1','coast')
                    testy = np.char.replace(testy,'2','forest')
                    testy = np.char.replace(testy,'3','highway')
                    testy = np.char.replace(testy,'4','insidecity')
                    testy = np.char.replace(testy,'5','kitchen')
                    testy = np.char.replace(testy,'6','livingroom')
                    testy = np.char.replace(testy,'7','mountain')
                    testy = np.char.replace(testy,'8','office')
                    testy = np.char.replace(testy,'9','opencountry')
                    #save per 100
                    testbuilding_name = ['image_%04d'%n for n in range(0, 1039+1)]
                    output = pd.DataFrame({"id": testbuilding_name, "label": testy})
                    output.to_csv("CS_IOC5008_0856619_HW1(%d).csv"%(ep10 * 10 + epoch), columns=["id", "label"], index=False) #output result to csv
                    #save data also
                    haccsarray = np.asarray(haccs).ravel() #save accuracy
                    hlossarray = np.asarray(hloss).ravel() #save loss
                    output2 = pd.DataFrame({"samp_id": range(1,len(haccsarray)+1), "loss": hlossarray, "acc": haccsarray})
                    output2.to_csv('hidobs(%d).csv'%(ep10*10 + epoch + 1), columns=["samp_id", "loss", "acc"], index=False) #output result to csv for observatio

                    torch.save(model2, 'hid_net(%d).pt'%((ep10*10 + epoch + 1))) #save model
                    print("--- %s seconds ---" % (time.time() - start_time)) #print total use time, originally use for observe performance
ep10 += 1

print("Train:   --- %s seconds ---" % (time.time() - mid_time)) #print time use for training model
mid_time2 = time.time()

#these part were originally use for output, but when we do infinite we dont need this   
model2 = model2.cpu() #put model on cpu when we are going to ouptu result to save moemory space

#basically same as output part in the infinite loop
with torch.no_grad(): # disable auto-grad
    for i, (data) in enumerate(testloader):
        data = Variable(data,requires_grad=False).cpu() #change data format
        
        outputs = model2(data).cpu() #get prediction
        outputs = outputs.detach() #flatten 
        out, index = torch.max(outputs,1)
        indexc = index.cpu() #put in cpu
        indexc = indexc.numpy() #change format for output
        testy = indexc

print("Predict:--- %s seconds ---" % (time.time() - mid_time2)) #print time model use for doing prediction

testy = testy.astype(np.str_) #change back to building name
testy = np.char.replace(testy,'10','street')
testy = np.char.replace(testy,'11','suburb')
testy = np.char.replace(testy,'12','tallbuilding')
testy = np.char.replace(testy,'0','bedroom')
testy = np.char.replace(testy,'1','coast')
testy = np.char.replace(testy,'2','forest')
testy = np.char.replace(testy,'3','highway')
testy = np.char.replace(testy,'4','insidecity')
testy = np.char.replace(testy,'5','kitchen')
testy = np.char.replace(testy,'6','livingroom')
testy = np.char.replace(testy,'7','mountain')
testy = np.char.replace(testy,'8','office')
testy = np.char.replace(testy,'9','opencountry')
   
testbuilding_name = ['image_%04d'%n for n in range(0, 1039+1)]
output = pd.DataFrame({"id": testbuilding_name, "label": testy})
output.to_csv("CS_IOC5008_0856619_HW1.csv", columns=["id", "label"], index=False) #output result to csv

torch.save(model2, 'net.pkl') #save model
print("--- %s seconds ---" % (time.time() - start_time)) #print total use time, originally use for observe performance
