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

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# np.random.seed(100）

import Embedding_Static


def read_data():
    train_data = glob("/Users/vanellope/2020-2021 Final Year Project/FYP_Static_CORPUS_6CLASS_HK/*")
    return train_data

def read_label():
    labels = np.load('/Users/vanellope/Desktop/FYP/textbook_corpus/gt.npy')
#     labels = np.load('/Users/vanellope/Desktop/FYP/textbook_corpus/gt_CN_12class.npy')
    return labels

def adj_acc(cm):
    right = sum(np.diag(cm))+sum(np.diag(cm,1))+sum(np.diag(cm,-1))
    total = sum(sum(cm))
    return right/total

# define torch.utils.data.Dataset
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        self.train = train
        self.filenames = read_data()
        self.labels =  read_label()

    def __getitem__(self, index):
        
        article = torch.load('/Users/vanellope/2020-2021 Final Year Project/FYP_Static_CORPUS_6CLASS_HK/article{}'.format(index))
        article = torch.cat(article[0], 1) # cat at dimension 1 
        label = self.labels[index]     
        return article,label

    def __len__(self):
        return len(self.filenames)
    

class Net(nn.Module):
    def __init__(self,embed_dim):
        super(Net, self).__init__()

        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 150, kernel_size = (2, self.embed_dim), stride = (1 , 1))
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = 150, kernel_size = (5, self.embed_dim), stride = (3 , 1))
        # self.conv3 = nn.Conv2d(in_channels = 1, out_channels = 150, kernel_size = (8, self.embed_dim), stride = (6 , 1))
#         self.conv_art1 = nn.Conv2d(in_channels = 1, out_channels = 300, kernel_size = (2, 300), stride = (1 , 1))
                
        self.fc_class_1 = nn.Linear(150*2, 120)
        self.fc_class_2 = nn.Linear(120, 6)

        # self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)

    def forward(self, x):
        temp_x_sent1 = self.conv1(x) # convolution
        temp_x_sent1,_ = torch.max(temp_x_sent1,dim = 2,keepdim=True) # pooling

        temp_x_sent2 = self.conv2(x)
        temp_x_sent2,_ = torch.max(temp_x_sent2,dim = 2,keepdim=True)
        
        # temp_x_sent3 = self.conv3(x)
        # temp_x_sent3,_ = torch.max(temp_x_sent3,dim = 2,keepdim=True)
            
        x_combine = torch.cat([temp_x_sent1,temp_x_sent2], 1) # stack
        
        x = torch.transpose(x_combine,1,3)
        x = x[0]
        # x = self.dp1(x)
        x = self.fc_class_1(x)
        x = self.dp2(x)
        x = self.fc_class_2(x)
        
        x = x[0] # squeeze 3d to 2d
#         print(x)
        return x

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

def save_fig(logbook_epoch, os): # output image
    A = logbook_epoch
    A = np.array(A)
    x = np.arange(A.shape[0])

    fig, ax = plt.subplots(figsize=(12,8))

    plt.ylim(ymin = -0.0,ymax = 1)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    #     plt.title("batch:{};lr:{}".format(batch_size,lr))
    ax.plot(x, A[:,0], 'b--', label='Train exact accuracy')
    ax.plot(x, A[:,1], 'r--', label='Train adjacent accuracy')
    ax.plot(x, A[:,2], 'b:', label='Val exact accuracy')
    ax.plot(x, A[:,3], 'r:', label='Val adjacent accuracy')
    ax.plot(x, A[:,0], 'b.', 
            x, A[:,1], 'r.', 
            x, A[:,2], 'b.', 
            x, A[:,3], 'r.')
    legend = ax.legend(loc='lower center', shadow=True)
    legend.get_frame().set_facecolor('C2')
    plt.show()
    fig.savefig(os)
    return


def train_test_model(net,train_loader,val_loader,epoch_number,learning_rate):
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
            labels = batch[1]
            pred = net(artics)
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

    A = logbook_epoch
    A = np.array(A)
    return A


############## MAIN ################

loop_list = [20]#[5,10,20,30,40,50,60,70,80,90,100] #[10, 25, 50, 100, 200, 400, 800] # , 300, 400, 500, 600, 700, 800, 900, 1000]
epoch_number = 70
learning_rate = 0.0001

for loop_number in loop_list:
    # refresh embedding file
    embedding_dimension = loop_number

    corpus_os = '/Users/vanellope/Desktop/FYP/textbook_corpus/textbook_HK_6class.txt'
    texts = Embedding_Static.get_texts(corpus_os)
    Embedding_Static.train_save_static_model(embedding_dimension = embedding_dimension, 
        window_size = 5, texts=texts)
    Embedding_Static.save_embed_texts(OS_embedding_dimension=embedding_dimension, 
        corpus_os=corpus_os)
    print('Embedding of DIM {} refreshing DONE!!!'.format(embedding_dimension))
    # refresh end
    net = Net(embed_dim=embedding_dimension)
    print('Net for dim {} initialized'.format(embedding_dimension))
    # loader class instance created
    train_loader,val_loader = creat_train_val_loader(0.7) # spliting ratio, 70% as train, 30% as validate
    # start training and get logbook A
    A = train_test_model(net,train_loader,val_loader,epoch_number,learning_rate)
    np.save('/Users/vanellope/Desktop/numpy_file_diff_embedding/{}'.format(embedding_dimension), A)


print("Finished!!!Congrats!!!!")





