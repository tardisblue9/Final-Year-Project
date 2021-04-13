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


def read_data():
    train_data = glob("/Users/vanellope/2020-2021 Final Year Project/FYP_BERT_CORPUS_6CLASS_HK/*")
#     train_data = glob("/Users/vanellope/2020-2021 Final Year Project/FYP_BERT_CORPUS_12CLASS_CN/*")
    return train_data

def read_label():
    labels = np.load('/Users/vanellope/Desktop/FYP/textbook_corpus/gt.npy')
#     labels = np.load('/Users/vanellope/Desktop/FYP/textbook_corpus/gt_CN_12class.npy')
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
        
        self.filenames = read_data()
        self.labels =  read_label()


    def __getitem__(self, index):
        
        article = torch.load('/Users/vanellope/2020-2021 Final Year Project/FYP_BERT_CORPUS_6CLASS_HK/article{}'.format(index))

        article = article[0]
        
        label = self.labels[index]
            
        return article,label

    def __len__(self):
        return len(self.filenames)
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc_sent_1 = nn.Linear(768, 192)
        self.fc_sent_2 = nn.Linear(192, 1)
#         self.fc_sent_3 = nn.Linear(20, 1)
        
        self.fc_art_1 = nn.Linear(768, 192)
        self.fc_art_2 = nn.Linear(192, 1)
#         self.fc_art_3 = nn.Linear(20, 1)
        
        self.fc_class_1 = nn.Linear(768, 192)
        # self.fc_class_2 = nn.Linear(192, 32)
        self.fc_class_3 = nn.Linear(192, 6)
#         self.fc_class_3 = nn.Linear(32, 12) # 12 class for mainland textbooks

        
        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)
        self.dp3 = nn.Dropout(p=0.2)


    def forward(self, x):
        temp_x = []
#         print("sentence number:",len(x))
        for x_sent in x:
#             print(x_sent.shape)
#             x_sent = x[0][0][0] # squeeze redundant dimension
#             print("in:",x_sent.shape)
            w_sent = F.relu(self.fc_sent_1(x_sent))
            w_sent = self.dp1(w_sent)
            w_sent = self.fc_sent_2(w_sent)
#             w_sent = self.dp5(w_sent)
#             w_sent = self.fc_sent_3(w_sent)
#             print("w after fc:",w_sent.shape)
            x_sent = torch.mul(x_sent, w_sent)
#             print("x after mul:",x_sent.shape)
            x_sent = torch.mean(x_sent, 2, True) # normalize： sum/length
#             print("out:",x_sent.shape)
            temp_x.append(x_sent)
        temp_x = torch.cat(temp_x, 2) # tensor 4d
        # print('sentence vec:',temp_x.shape)
        # print('for loop over')
        # return
        
        x = temp_x
#         print(x.shape)
        w_art = F.relu(self.fc_art_1(x))
        w_art = self.dp2(w_art)
        w_art = self.fc_art_2(w_art)
#         w_sent = self.dp6(w_sent)
#         w_art = self.fc_art_3(w_art)
        x = torch.mul(x, w_art)
        x = torch.mean(x, 2, True)  # 768x1
#         print(x.shape)
        x = F.relu(self.fc_class_1(x))
        x = self.dp3(x)
        # x = F.relu(self.fc_class_2(x))
#         x = self.dp4(x)
        x = self.fc_class_3(x)
        x = x[0][0] # squeeze 4d to 2d, in order to match loss func
        
#         print('article vec shape ',x.shape,' ;and is:',x)

        return x
#         return torch.sigmoid(x)


net = Net()
print(net)



# initailize class instance
dataset = TrainDataset('hello world',train=True) # make the instance for dataset object
ratio = 0.7    # spliting ratio, 70% as train, 30% as validate
train_set, val_set = torch.utils.data.random_split(dataset, 
                    [int(ratio*len(dataset)), len(dataset)-int(ratio*len(dataset))])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)
print('the dataset is divided as :',len(train_set),len(val_set))


logbook_epoch = []
epoch_number = 30
learning_rate = 0.0001
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
save_fig(logbook_epoch, "/Users/vanellope/Desktop/FYP/Plot归档/embedBert_level2_attention768_192_lr{}_epoch{}_p0.2.jpg".format(learning_rate,epoch_number))
