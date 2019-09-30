import csv
import copy as copy
import numpy as np
import imageio #for reading image
import pandas as pd

bedroom_data = ['dataset/train/bedroom/image_%04d.jpg'%n for n in range(0, 135+1)]
coast_data = ['dataset/train/coast/image_%04d.jpg'%n for n in range(0, 279+1)]
forest_data = ['dataset/train/forest/image_%04d.jpg'%n for n in range(0, 247+1)]
highway_data = ['dataset/train/highway/image_%04d.jpg'%n for n in range(0, 179+1)]
insidecity_data = ['dataset/train/insidecity/image_%04d.jpg'%n for n in range(0, 227+1)]
kitchen_data = ['dataset/train/kitchen/image_%04d.jpg'%n for n in range(0, 129+1)]
livingroom_data = ['dataset/train/livingroom/image_%04d.jpg'%n for n in range(0, 208+1)]
mountain_data = ['dataset/train/mountain/image_%04d.jpg'%n for n in range(0, 293+1)]
office_data = ['dataset/train/office/image_%04d.jpg'%n for n in range(0, 134+1)]
opencountry_data = ['dataset/train/opencountry/image_%04d.jpg'%n for n in range(0, 329+1)]
street_data = ['dataset/train/street/image_%04d.jpg'%n for n in range(0, 211+1)]
suburb_data = ['dataset/train/suburb/image_%04d.jpg'%n for n in range(0, 160+1)]
tallbuilding_data = ['dataset/train/tallbuilding/image_%04d.jpg'%n for n in range(0, 275+1)]
train_data = bedroom_data+coast_data+forest_data+highway_data+insidecity_data+kitchen_data+livingroom_data+mountain_data+office_data+opencountry_data+street_data+suburb_data+tallbuilding_data
test_data = ['dataset/test/image_%04d.jpg'%n for n in range(0, 1039+1)]

train_labels = []
test_labels = []

def readdata(data, labels, typeofinput):
    siz = 72600
    datas = np.empty((0,siz)) #temporary container for input images
    if(typeofinput == 'train'):
        print('train')
        datas = np.empty((0,siz+1))
    num_count = 0
    pix = []
    for i in range(siz):
        if(typeofinput == 'train' and i == 0):
            pix.append('label')
        pix.append('pixel' + str(i+1))
    #print(pix)
    pix = np.asarray(pix)
    datas = np.vstack([datas, pix])
    #print(pix)
    #print(datas)
    for d in data:
        num_count = num_count + 1
        print(num_count)
        label = ''
        if(num_count <= 136):
            #label = 'bedroom'
            label = 0
        elif(num_count <= 416):
            #label = 'coast'
            label = 1
        elif(num_count <= 664):
            #label = 'forest'
            label = 2
        elif(num_count <= 844):
            #label = 'highway'
            label = 3
        elif(num_count <= 1072):
            #label = 'insidecity'
            label = 4
        elif(num_count <= 1202):
            #label = 'kitchen'
            label = 5
        elif(num_count <= 1411):
            #label = 'livingroom'
            label = 6
        elif(num_count <= 1705):
            #label = 'mountain'
            label = 7
        elif(num_count <= 1840):
            #label = 'office'
            label = 8
        elif(num_count <= 2170):
            #label = 'opencountry'
            label = 9
        elif(num_count <= 2382):
            #label = 'street'
            label = 10
        elif(num_count <= 2543):
            #label = 'suburb'
            label = 11
        elif(num_count <= 2819):
            #label = 'tallbuilding'
            label = 12
        data = imageio.imread(d, as_gray=True) #different decompress mjpeg method, use imageio could get faster image reading speed
        pic = np.asarray(data) #add this line for io
        pic = np.resize(pic, (330, 220))
        pic = np.reshape(pic, (-1, siz)) #add this line for io
        #print(pic.shape)
        #print(pic[0])
        if(typeofinput == 'train'):
            p = pic.tolist()
            for row in p:
                row.insert(0, label)
            #print(p)
            pic = np.asarray(p)
            #print(pic.shape)
        #print(pic)
        datas = np.vstack([datas, pic]) #put new read image into data
        labels.append(label) #use tag number on pictures' as label
    return datas, labels #return the whole list of image and their tag
    
train_building_data, train_labels = readdata(train_data, train_labels, 'train') #use class we create to read data sequentially
test_building_data, test_labels = readdata(test_data, test_labels, 'test')

output = pd.DataFrame(train_building_data)
output.to_csv("train.csv", columns=None, index=False, header=None) #output result to csv

output = pd.DataFrame(test_building_data)
output.to_csv("test.csv", columns=None, index=False, header=None) #output result to csv
