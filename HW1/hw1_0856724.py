import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import random
import time
import cv2
import csv
import os
import PIL.Image as Image
from IPython.display import display
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_root_dir = 'training_data/training_data/'
test_root_dir = 'testing_data/testing_data/'
test_number = 100

print(device)
print(torch.cuda.get_device_name(device))

class label_library:
    # A library to change car_type label to a number
    def __init__(self):
        self.label2index = {}
        self.label2count = {}
        self.index2label = {}
        self.n_classes = 0

    def addlabel(self, label):
        if label not in self.label2index:
            self.label2index[label] = self.n_classes
            self.label2count[label] = 1
            self.index2label[self.n_classes] = label
            self.n_classes += 1
        else:
            self.label2count[label] += 1

class CarDataSet(Dataset):
    # Cover the training_data to a custom dataset
    def __init__(self, root_dir, labels, transform):
        self.root_dir = root_dir
        self.transform = transform
        all_imgs = os.listdir(root_dir)
        self.total_imgs = natsorted(all_imgs)
        self.lbls = labels
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)
        label = self.lbls[idx]
        return tensor_image, label

         
    
train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                
car_type = label_library()
label_csv = pd.read_csv('training_labels.csv')
label_csv = label_csv.sort_values("id")
car_labels = list(label_csv.iloc[:, 1])

# Store the car_type in dictionary
for i in range(len(car_labels)):
    car_type.addlabel(car_labels[i])
    car_labels[i]=car_type.label2index[car_labels[i]]
    

car_dataset=CarDataSet(root_dir=train_root_dir,
                       labels=car_labels,
                       transform=train_tfms)
            
#Valid some data as test data
test_list = random.sample(range(1, len(car_dataset)), test_number)

trainloader = torch.utils.data.DataLoader(car_dataset, batch_size = 32, shuffle=True, num_workers = 2)

def train_model(model, criterion, optimizer, scheduler, n_epochs = 5):
    # train the model
    losses = []
    accuracies = []
    test_accuracies = []
    # set the model to train mode initially
    model.train()
    for epoch in range(n_epochs):
        torch.save(model_ft,'HW1_resnet101_'+str(epoch)+'.pt')
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs and assign them to cuda
            inputs, labels = data

            #inputs = inputs.to(device).half() # uncomment for half precision model
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_duration = time.time()-since
        epoch_loss = running_loss/len(trainloader)
        epoch_acc = 100/32*running_correct/len(trainloader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))
        
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        # switch the model to eval mode to evaluate on test data
        model.eval()
        test_acc = eval_model(model)
        test_accuracies.append(test_acc)
        
        # re-set the model to train mode after validating
        model.train()
        scheduler.step(test_acc)
        since = time.time()
    print('Finished Training')
    return model, losses, accuracies, test_accuracies

def eval_model(model):
    correct = 0.0
    for i in range(test_number):
        test_image_number = test_list[i]
        image = torch.autograd.Variable(car_dataset[test_image_number][0], requires_grad=True)
        image = image.unsqueeze(0)
        image = image.cuda() 
        output = model(image)
        conf, predicted = torch.max(output.data, 1)
        if (predicted.item() == car_dataset[test_image_number][1]):
            correct += 1
    test_acc = 100.0 * correct / test_number
    print('Accuracy of the network on the test images: %d %%' % (
        test_acc))
    return test_acc

def gen_test_csv(model):
    all_test_imgs = os.listdir(test_root_dir)
    test_imgs_len = len(all_test_imgs)
    all_test_result = []
    for i in range(test_imgs_len):
        if i % 100 ==0:
            print(i/100)
        test_img_path = os.path.join(test_root_dir,all_test_imgs[i])
        test_img = Image.open(test_img_path).convert('RGB')
        test_img = test_tfms(test_img).float()
        test_img = torch.autograd.Variable(test_img, requires_grad=True)
        test_img = test_img.unsqueeze(0)
        test_img = test_img.cuda()
        output = model(test_img)
        conf, predicted = torch.max(output.data, 1)
        test_result = [all_test_imgs[i], car_type.index2label[predicted.item()]]
        all_test_result.append(test_result)
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        
        for i in range(test_imgs_len):
            writer.writerow(all_test_result[i])
            
    display(Image.open(test_img_path))
    print(all_test_imgs[1], ":", car_type.index2label[predicted.item()], "confidence: ", conf.item())

    return 0


model_ft = models.resnet101(pretrained=True)
num_ftrs = model_ft.fc.in_features

# replace the last fc layer with an untrained one (requires grad by default)
model_ft.fc = nn.Linear(num_ftrs, 196)
model_ft = model_ft.to(device)

# uncomment this block for half precision model

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)
lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

model_ft, training_losses, training_accs, test_accs = train_model(model_ft, criterion, optimizer, lrscheduler, n_epochs=20)
traing_state = {'training_losses' : training_losses,
                'training_accs' : training_accs,
                'test_accs' : test_accs}

torch.save(model_ft,'HW1_resnet101.pt')
torch.save(traing_state,'HW1_resnet101_state.a')

HW1_model=torch.load('HW1_resnet101.pt')
HW1_model.eval()
eval_model(HW1_model)
gen_test_csv(HW1_model)

