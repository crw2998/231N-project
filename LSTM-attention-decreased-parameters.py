
# coding: utf-8

# In[1]:


from data import Data
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import itertools
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

CHECKPOINT_PATH = 'checkpoints/'


# In[2]:


d = Data(x_transpose=(0, 3, 1, 2))
X_train,y_train = d.get_train()
X_cross, y_cross = d.get_dev()
X_test,y_test = d.get_test()


# In[3]:


# y_train = y_train
print(X_train.shape)
print(y_train.shape)
print(X_cross.shape)
print(y_cross.shape)
print(X_test.shape)
print(y_test.shape)


# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.models as models

dtype = torch.float

USE_GPU = True
if USE_GPU:
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    device = torch.device('cuda')
else:
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    device = torch.device('cpu')


# In[5]:


def flatten(X):
    N = X.shape[0]
    return X.view(N,-1)


# In[6]:


class Flatten(nn.Module):
    def forward(self, x): 
        return flatten(x)


# In[7]:


class MyCustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        return (img, label)

    def __len__(self):
        return self.X.shape[0]


# In[8]:


batch_size = 20
check_batch_size = 20
NUM_TRAIN = X_train.shape[0]
train_dataset = MyCustomDataset(X_train, y_train)
loader_train = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)), drop_last = True)

train_dataset_check = MyCustomDataset(X_train, y_train)
loader_train_check = DataLoader(train_dataset_check, batch_size=check_batch_size, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)), drop_last = True)

NUM_CROSS = X_cross.shape[0]
cross_dataset = MyCustomDataset(X_cross, y_cross)
loader_cross = DataLoader(cross_dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(NUM_CROSS)), drop_last = True)

cross_dataset_check = MyCustomDataset(X_cross, y_cross)
loader_cross_check = DataLoader(cross_dataset_check, batch_size=check_batch_size, sampler=sampler.SubsetRandomSampler(range(NUM_CROSS)), drop_last = True)


# In[9]:


def check_accuracy(loader, model): 
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            scores = model(Variable(x.float().type(dtypeFloat)))
            _, preds = scores.max(1)
            num_correct += (preds == Variable(y.long().type(dtypeLong))).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


# In[10]:


def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    print(img.shape)
    img = img.transpose(1,2,0)
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    #plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')


# In[11]:


print_every = 10
print_acc_every = 1000
show_transformations = False

def train(m, optimizer, epochs=60):
    loss_arr = []
    for e in range(epochs):
        if e % 1 == 0:
            torch.save(m.state_dict(), CHECKPOINT_PATH + str(e) + '.chkpoint')
        for t, (x, y) in enumerate(loader_train):
            m.train()
            scores = m(Variable(x.float().type(dtypeFloat)))
            loss = F.cross_entropy(scores, Variable(y.long().type(dtypeLong)))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss))
                loss_arr += [loss.item()]
                if (t % print_acc_every == 0):
                    print ("train acc:")
                    check_accuracy(loader_train_check, m)
                    print ("cross acc:")
                    m.eval()
                    check_accuracy(loader_cross_check, m)
                    
                    # print transformations
                    if show_transformations:
                        x_ = x[5][None]
                        stn = next(m.modules())[0]
                        plt.subplot(1, 2, 1)
                        i = 50
                        imshow_noax(x_.data.numpy()[0], normalize=False)
                        plt.subplot(1, 2, 2)
                        stn_out = stn(Variable(x_.float().type(dtypeFloat))).data.numpy()[0]
                        imshow_noax(stn_out, normalize=False)
                        plt.show()
                    
    return loss_arr


# In[12]:


attn_channel_1 = 32
attn_channel_2 = 64
attn_channel_3 = 96
attn_channel_4 = 128
attn_channel_5 = 160
n_class=2

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        
        self.model = nn.Sequential(            
            nn.Conv2d(3, attn_channel_1, 7, stride=1, padding=3), # 224 x 224 x 96
            nn.MaxPool2d(2, stride=2), # 112 x 112 x 32
            nn.ReLU(),
#             nn.Conv2d(attn_channel_1, attn_channel_2, 5, stride=1, padding=2), # 112 x 112 x 128
            nn.Conv2d(attn_channel_1, attn_channel_2, 5, stride=1, padding=2), # 112 x 112 x 128
            nn.MaxPool2d(2, stride=2), # 56 x 56 x 64
            torch.nn.Dropout2d(p=0.3),
            nn.ReLU(),
#             nn.Conv2d(attn_channel_2, attn_channel_3, 3, stride=1, padding=1), # 56 x 56 x 256
            nn.Conv2d(attn_channel_2, attn_channel_3, 3, stride=1, padding=1), # 56 x 56 x 256
            nn.MaxPool2d(2, stride=2), # 28 x 28 x 96
            torch.nn.Dropout2d(p=0.3),
            nn.ReLU(),
#             nn.Conv2d(attn_channel_2, attn_channel_3, 3, stride=1, padding=1), # 56 x 56 x 256
            nn.Conv2d(attn_channel_3, attn_channel_4, 3, stride=1, padding=1), # 56 x 56 x 256
            nn.MaxPool2d(2, stride=2), # 14 x 14 x 128
            nn.Dropout2d(p=0.3), 
            nn.ReLU(),
            nn.Conv2d(attn_channel_4, attn_channel_5, 3, stride=1, padding=1), # 56 x 56 x 256
            nn.MaxPool2d(2, stride=2), # 7 x 7 x 160
            nn.ReLU(),
#             Flatten(),
#             nn.Linear(28*28*256, n_class),
        )       
        
    def forward(self, x):
        return self.model(x)


# In[13]:


first_downsample_layer = 64
second_downsample_layer = 32
C = 160
class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        
        self.model = nn.Sequential(            
            nn.Conv2d(C, first_downsample_layer, 1, stride=1, padding=0),
#             nn.Conv2d(first_downsample_layer, second_downsample_layer, 1, stride=1, padding=0),
        )
        
    def forward(self, x):
        return self.model(x)  


      


# In[14]:


n_class = 2
C, H, W = 160, 7, 7

class MapToClass(nn.Module):
    def __init__(self):
        super(MapToClass, self).__init__()
        
        self.model = nn.Sequential(            
            Flatten(),
            nn.Linear(C * H * W, n_class),
        )
        
    def forward(self, x):
        return self.model(x)  


# In[ ]:


filter_size=5
C, H, W = 160, 7, 7

class AttentiveConvLSTM(nn.Module):
    def __init__(self):
        super(AttentiveConvLSTM, self).__init__()
        
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
        
        self.Conv_Wa = nn.Conv2d(C, C, filter_size, padding=2)
        self.Conv_Ua = nn.Conv2d(C, C, filter_size, padding=2)
        self.Conv_Va = nn.Conv2d(C, 1, filter_size, padding=2)
        self.SoftMax_a = nn.Softmax(dim=2)
        self.ba = Variable(torch.randn(H, device = device, dtype=dtype, requires_grad=True))
        
        self.Conv_Wi = nn.Conv2d(C, C, filter_size, padding=2)
        self.Conv_Ui = nn.Conv2d(C, C, filter_size, padding=2)
        self.bi = Variable(torch.randn(H, device = device, dtype=dtype, requires_grad=True))
    
        self.Conv_Wf = nn.Conv2d(C, C, filter_size, padding=2)
        self.Conv_Uf = nn.Conv2d(C, C, filter_size, padding=2)
        self.bf = Variable(torch.randn(H, device = device, dtype=dtype, requires_grad=True))

        self.Conv_Wo = nn.Conv2d(C, C, filter_size, padding=2)
        self.Conv_Uo = nn.Conv2d(C, C, filter_size, padding=2)
        self.bo = Variable(torch.randn(H, device = device, dtype=dtype, requires_grad=True))
        
        self.Conv_Wc = nn.Conv2d(C, C, filter_size, padding=2)
        self.Conv_Uc = nn.Conv2d(C, C, filter_size, padding=2)
        self.bc = Variable(torch.randn(H, device = device, dtype=dtype, requires_grad=True))
        
        self.H0 = Variable(torch.randn(C, H, W, device = device, dtype=dtype, requires_grad=True))
        self.C0 = Variable(torch.randn(C, H, W, device = device, dtype=dtype, requires_grad=True))
        
    def forward(self, X):
        # put the forward pass here (iterating through the channels of X)
        # iteratively call AttentitiveConvLSTM_step
        # Return X', which is the features modified with the attention map
        Hprev = self.H0
        Cprev = self.C0
        Hprev = Hprev.unsqueeze(0)
        Cprev = Cprev.unsqueeze(0)
        
        #####################################################
        ################### FUCK VINEET #####################
        #####################################################
        
        for i in range (10):
            
            # Attention layer
            tanh_input = self.Conv_Wa(X) + self.Conv_Ua(Hprev) + self.ba
            tanh_output = self.Tanh(tanh_input)
            Z_t = self.Conv_Va(tanh_output)
#             print (i)
#             print (X.size())
#             print (Z_t.size())
            Z_t_reshaped = Z_t.view(Z_t.size()[0], 1, -1)
            A_t_initial = self.SoftMax_a(Z_t_reshaped)
            A_t = A_t_initial.view(-1, 1, H, W)
            A_t_expanded = A_t.expand(X.size())
            X_t = A_t_expanded * X

            # LSTM layer
            I_t = self.Sigmoid(self.Conv_Wi(X_t) + self.Conv_Ui(Hprev) + self.bi)
            F_t = self.Sigmoid(self.Conv_Wf(X_t) + self.Conv_Uf(Hprev) + self.bf)
            O_t = self.Sigmoid(self.Conv_Wo(X_t) + self.Conv_Uo(Hprev) + self.bo)
            G_t = self.Tanh(self.Conv_Wc(X_t) + self.Conv_Uc(Hprev) + self.bc)

            Cprev = Cprev * F_t + I_t * G_t
            Hprev = O_t * self.Tanh(Cprev)
        
        return X_t
        
#         tanh_input = F.conv2d(X, self.Wa, padding=2) + F.conv2d(Hprev, self.Ua, padding=2) + self.ba
#         Z_t = F.conv2d(nn.Tanh(tanh_input), self.Va, stride=1, padding=2)
#         print(Z_t.size())
        
        
        
        
        #         tanh_input1 = F.conv2d(X, (C, C, filter_size, filter_size), stride=1, padding=2)
#         tanh_input2 = F.conv2d(Hprev, (C, C, filter_size, filter_size), stride=1, padding=2)
#         tanh_input = tanh_input1 + tanh_input2 + self.ba


# In[ ]:


learning_rate = 2.2e-4
num_classes = 2

model = nn.Sequential(
    FeatureExtraction(),
    AttentiveConvLSTM(), # outputs attention map
#     Downsample(),
    MapToClass(),
)
if USE_GPU:
    model = model.cuda()
    
# # image generation (to see feature map), and also the way to forward pass
# for t, (x, y) in enumerate(loader_train):
#     img = model(Variable(x.float().type(dtypeFloat)))
#     img = img[0].data.cpu().numpy()
#     print (img.shape)
#     imgplot = plt.imshow(img[100])
#     break
    
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
loss_arr = train(model, optimizer, 40)
print(loss_arr)

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.plot(loss_arr)
plt.savefig("loss.png")


# In[ ]:


for t, (x, y) in enumerate(loader_train):
    featureExtraction, attention = next(model.modules())[:2]
    feature_out = featureExtraction(Variable(x.float().type(dtypeFloat)))
    attention_out = attention(feature_out)
    attention_out_numpy = attention_out.data.cpu().numpy()
    img = np.mean(attention_out_numpy, axis=1)
    real_img = x
    break



# In[ ]:


# imgplot = plt.imshow(img[17])
# plt.show()


# # In[ ]:


# real_img = real_img.data.cpu().numpy()


# # In[ ]:


# imshow_noax(real_img[17])

