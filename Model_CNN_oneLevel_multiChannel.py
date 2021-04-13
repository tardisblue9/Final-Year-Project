import matplotlib.pyplot as plt
import numpy  as np
from glob import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

# np.random.seed(100）

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def read_label():
    labels = np.load('/Users/vanellope/Desktop/FYP/textbook_corpus/gt.npy')
    return labels

def adj_acc(cm):
    right = sum(np.diag(cm))+sum(np.diag(cm,1))+sum(np.diag(cm,-1))
    total = sum(sum(cm))
    return right/total

def save_fig(logbook_epoch, os): # output image
    A = logbook_epoch
    A = np.array(A)
    x = np.arange(A.shape[0])
    np.save(os, A)

    fig, ax = plt.subplots(figsize=(12,8))

    plt.ylim(ymin = -0.0,ymax = 1)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    #     plt.title("batch:{};lr:{}".format(batch_size,lr))
    ax.plot(x, A[:,0], 'b--', label='Train exact accuracy')
    ax.plot(x, A[:,1], 'r--', label='Train adjacent accuracy')
    ax.plot(x, A[:,2], 'b:', label='Test exact accuracy')
    ax.plot(x, A[:,3], 'r:', label='Test adjacent accuracy')
    ax.plot(x, A[:,0], 'b.', 
            x, A[:,1], 'r.', 
            x, A[:,2], 'b.', 
            x, A[:,3], 'r.')
    legend = ax.legend(loc='lower center', shadow=True)
    legend.get_frame().set_facecolor('C2')
    plt.show()
    fig.savefig(os)
    return

# define torch.utils.data.Dataset
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        self.train = train
        
        self.labels =  read_label()

    def __getitem__(self, index):
        
        article_bert = torch.load('/Users/vanellope/2020-2021 Final Year Project/FYP_BERT_CORPUS_6CLASS_HK/article{}'.format(index))
        article_bert = torch.cat(article_bert[0], 1)
        article_static = torch.load('/Users/vanellope/2020-2021 Final Year Project/FYP_Static_CORPUS_6CLASS_HK/article{}'.format(index))
        article_static = torch.cat(article_static[0], 1)
        
        label = self.labels[index]
                    
        return (article_bert,article_static),label

    def __len__(self):
        return len(self.labels)
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.channel_num = 150

        self.embed_dimB = 768
        self.embed_dimS = 40

        self.conv11 = nn.Conv2d(in_channels = 1, out_channels = self.channel_num, kernel_size = (2, self.embed_dimB), stride = (1 , 1))
        self.conv12 = nn.Conv2d(in_channels = 1, out_channels = self.channel_num, kernel_size = (5, self.embed_dimB), stride = (3 , 1))
        self.conv13 = nn.Conv2d(in_channels = 1, out_channels = self.channel_num, kernel_size = (8, self.embed_dimB), stride = (5 , 1))
        
        self.conv21 = nn.Conv2d(in_channels = 1, out_channels = self.channel_num, kernel_size = (2, self.embed_dimS), stride = (1 , 1))
        self.conv22 = nn.Conv2d(in_channels = 1, out_channels = self.channel_num, kernel_size = (5, self.embed_dimS), stride = (3 , 1))
        self.conv23 = nn.Conv2d(in_channels = 1, out_channels = self.channel_num, kernel_size = (8, self.embed_dimS), stride = (5 , 1))             
        
        self.fc_class_1 = nn.Linear(self.channel_num*6, 120)
        self.fc_class_2 = nn.Linear(120, 6)
        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x1 = x[0] # bert embed
        
        temp_x_sent1_1 = self.conv11(x1) # convolution
        temp_x_sent1_1,_ = torch.max(temp_x_sent1_1,dim = 2,keepdim=True) # pooling

        temp_x_sent1_2 = self.conv12(x1)
        temp_x_sent1_2,_ = torch.max(temp_x_sent1_2,dim = 2,keepdim=True)

        temp_x_sent1_3 = self.conv13(x1)
        temp_x_sent1_3,_ = torch.max(temp_x_sent1_3,dim = 2,keepdim=True)
        
        x2 = x[1]
        
        temp_x_sent2_1 = self.conv21(x2) # convolution
        temp_x_sent2_1,_ = torch.max(temp_x_sent2_1,dim = 2,keepdim=True) # pooling

        temp_x_sent2_2 = self.conv22(x2)
        temp_x_sent2_2,_ = torch.max(temp_x_sent2_2,dim = 2,keepdim=True) 

        temp_x_sent2_3 = self.conv23(x2)
        temp_x_sent2_3,_ = torch.max(temp_x_sent2_3,dim = 2,keepdim=True)    
        
        x_combine = torch.cat([temp_x_sent1_1,temp_x_sent1_2,temp_x_sent1_3,
                            temp_x_sent2_1,temp_x_sent2_2,temp_x_sent2_3], 1) # stack
        
        x = torch.transpose(x_combine,1,3)
        x = x[0]
        x = self.dp1(x)
        x = F.relu(self.fc_class_1(x))
        x = self.dp2(x)
        x = self.fc_class_2(x)
        
        return x[0]

def creat_train_val_loader(split_ratio):
    # make the instance for dataset object
    dataset = TrainDataset('hello world',train=True)
    ratio =  split_ratio   
    train_set, val_set = torch.utils.data.random_split(dataset, 
                        [int(ratio*len(dataset)), len(dataset)-int(ratio*len(dataset))])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)
    print('the dataset is divided as :',len(train_set),len(val_set))
    return train_loader,val_loader

def train_test_model(net, train_loader,val_loader,epoch_number,learning_rate):
    logbook_epoch = []
    print('epoch_number:',epoch_number,';learning_rate:',learning_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)# Define a Loss function and optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(epoch_number):
        if epoch>0:print("last time record:",logbook_epoch[-1])
        print('progress: %.1f %%' % (100*epoch/epoch_number),'\n') # progress display
        acm_loss = 0
        total = len(train_loader)
        
        for i, batch in enumerate(train_loader):

            artics = batch[0]
    #         print(len(batch[0]))
            labels = batch[1]
    #         print(labels)

            pred = net(artics)
    #         print("pre:",pred,";label:",labels)
            loss = loss_func(pred, labels) 

            acm_loss+=loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)        
            optimizer.step()       

            if i % 100 == 99:
                print("[epoch {}/{},batch {}/{}]:Loss:{}".format(epoch,epoch_number,i+1,total,acm_loss))
                acm_loss = 0
                
        pred = []
        gt = []
        print('train error get.')
        for batch in train_loader:
            artics = batch[0]
            labels = batch[1]
            temp = net(artics)[0]
            gt.append(labels)
            pred.append(torch.argmax(temp))
        cm = confusion_matrix(gt, pred)
        train_acc = accuracy_score(gt, pred)
        train_adj = adj_acc(cm)
          
        pred = []
        gt = []
        print('validation error get.')
        for batch in val_loader:
            artics = batch[0]
            labels = batch[1]
            temp = net(artics)[0]
            gt.append(labels)
            pred.append(torch.argmax(temp))
        cm = confusion_matrix(gt, pred)
        val_acc = accuracy_score(gt, pred)
        val_adj = adj_acc(cm)

        logbook_epoch.append([train_acc,train_adj,val_acc,val_adj])
    print("plotting plot by epoch...")
    save_fig(logbook_epoch, "/Users/vanellope/Desktop/FYP/Plot归档/embedBert&Static40_level1_CNN_2_5_8_150c_lr{}_epoch{}.jpg".format(learning_rate,epoch_number))
    return logbook_epoch

############## MAIN ################

net = Net()
print(net)
epoch_number = 70
learning_rate = 0.0001

# loader class instance created
train_loader,val_loader = creat_train_val_loader(0.7) # spliting ratio, 70% as train, 30% as validate

# start training 
log = train_test_model(net,train_loader,val_loader,epoch_number,learning_rate)

print("Finished!!!Congrats!!!!")




