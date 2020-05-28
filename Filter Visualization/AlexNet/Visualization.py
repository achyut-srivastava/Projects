#!/usr/bin/env python
# coding: utf-8

# # The concept behind this.
# 
# The project is done in Pytorch version 1.5.0
# 
# The algorithm for the below code is taken from the paper - https://arxiv.org/abs/1311.2901. 
# 
# A great paper for peeps working on or(just started) in CNN. The technique used here is the same that is used in **GAN's** and **Neural Style Transfer**. Only the technique is same, not he method.
# 
# The Project in short is all about what **patterns** is CNN looking for in the image that is passed in the network.

# ## Importing datasets:
# 
# 1. **cv2**: for reading and reszing the image
# 2. **torch libraries**: for tensor applications and models
# 3. **matplotlib**: for plotting
# 4. **numpy**: for conversion
# 5. **partial**: a function that will help in hooking layers in the models
# 6. **torchsummary**: to show the summary of the model (similar to model.summary() in keras)
# 7. **OrderedDict**: for storing the feature maps and pool indices

# In[1]:


import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from torchsummary import summary
from collections import OrderedDict


# In[2]:


torch.__version__


# Moving the model to gpu or cpu depending on the availability.

# In[4]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ### Importing the model from pytorch and moving to device.
# 
# You can change the size of the input image (N, H, W) in summary (**args** ) and can see the size of the at each layer.
# 
# N: Number of channels
# 
# H: Height of image
# 
# W: Width of Image

# In[15]:


model = models.alexnet(pretrained =  True).to(device)
print(summary(model, (3,223,223)))


# # Making a model for alexnet network as mentioned in pytorch.
# # Here - https://pytorch.org/docs/stable/torchvision/models.html?highlight=alexnet#torchvision.models.alexnet
# 
# #### What we are doing in this section is we will create a module for our alexnet having features and classifiers and then assigning the pretrained weights to our created features and classifier.

# In[3]:


class alexConv(nn.Module):
    def __init__(self):
        super(alexConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride = 4, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride = 2, return_indices=True), 
    # return indices has to be true as it act as switches which holds the location of max pixel values from receptive-field
            
            #2nd
            nn.Conv2d(64, 192, kernel_size = 5, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride = 2, return_indices = True),
            
            #3rd
            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride = 2, return_indices = True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.Softmax(dim = 1),
        )
        self.feature_maps = OrderedDict() # For storing all the feature maps of *this* convnet_model {0: [[[[]]]], .....}
        self.pool_loci = OrderedDict() # For storing the switch indices for pool layers {2: [[[[]]]], .....}
        self.conv_layer_indices = [0, 3, 6, 8, 10] #storing the conv2d layer indices
        self.init_weights() # calling function to assign the pretrained alexnet weights to our model
    
    def check(self):
        model = models.alexnet(pretrained = True)
        return model
    
    def init_weights(self):
        alexnet_pre = self.check()
        for idx, layer in enumerate(alexnet_pre.features):
        # 0 , Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)). This how idx, layer will look like
            if isinstance(layer, nn.Conv2d): #checking if layer is nn.Conv2d or not if  yes then proceed
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data
                
        for idx, layer in enumerate(alexnet_pre.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data
    
    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, loc = layer(x) #since we are using return indices = True for MaxPool2d. it gives 2 output, x and location of pool
            else:
                x = layer(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


# # Defining Deconv net for our alexnet model.
# 
# Note here we won't declare the classifier section as mentioned above. We will start from bottom and go to top for our input.
# 
#             Conv-Net:  Input --> Conv1 --> ReLu --> MaxPool2d.
# 
#             Decon-Net: MaxPool2d --> ReLU --> Conv1 --> Input.

# In[4]:


class alexDeconv(nn.Module):
    def __init__(self):
        super(alexDeconv, self).__init__()
        self.features = nn.Sequential(
            #1st
            nn.MaxUnpool2d(stride = 2, kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 384, 3, padding = 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(384, 192, 3, padding = 1),
            
            #2nd
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 64, 5, padding = 2),
            
            #3rd
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 11, stride= 4, padding = 2),
        )
        
        self.conv_deconv_idx = {0:12 , 3:9 , 6:6 , 8:4, 10:2} # Mapping location of conv2d layer in both Convnet and Deconvnet.
        self.unpool_pool_idx = {10:2, 7:5, 0:12} # Mapping location of Un-Pooling layer:Pooling Layer
        self._init_weights() # assigning same weights of alexnet pretrained to our deconvnet model 
    
    def _init_weights(self):
        alexnet_pre = models.alexnet(pretrained = True)
        for ids, layer in enumerate(alexnet_pre.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv_deconv_idx[ids]].weight.data = layer.weight.data
    
    def forward(self, x, layer, pool_locs):
        if layer in self.conv_deconv_idx:
            start_idx = self.conv_deconv_idx[layer]
        else:
            print("It is not a conv layer")
        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d): 
        #If the layer is unpooling we need to pass 2 inputs. First is x and second is switches stored in pool_locs 
        # when we call return indices = true in MaxPool2d
                x = self.features[idx](x, pool_locs[self.unpool_pool_idx[idx]])
            else:
                x = self.features[idx](x)
        return x        


# Here we are reading the Imagent label txt file.

# In[5]:


file = open("label_idx.txt", "r").read()
dicti = eval(file)


# # Funtions:
# 
# 1. **get_image**:  It takes image path as input and returns image in a format accepted by the model.
# 
# 2. **collect**:  It iterate over every layer in the alexnet conv model and hook the function i.e stores the output of every layer in feature_maps and pool_loci defined in alexConv().
# 
#     2.1. **hook**:  provides the hook for each layer.
#     
# 3. **vis_layer**:  It calculates the maximum activated feature map in each Conv2d layer. Zero out all the other feature maps of the layer and pass that feature maps in alexDeconv() to produce the output patterns

# In[6]:


def get_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig = img
    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=[1.1, 1.15], saturation=[0.9, 0.95], contrast=[0.9, 0.95]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = (0.229, 0.224, 0.225))
    ])
    img = tfms(img)
    img.unsqueeze_(0)
    return img, orig


# In[7]:


def collect(mod):
    def hook(module, input, output, key):        
        if isinstance(module, nn.MaxPool2d):
            mod.feature_maps[key] = output[0] #storing the output feature maps in feature_maps defined in alexConv()
            mod.pool_loci[key] = output[1] # storing the switches in pool_loci defined in alexConv()
        else:
            mod.feature_maps[key] = output
    
    for idx, layer in enumerate(mod._modules.get('features')): # enumerate over every layer and hook each of them
        layer.register_forward_hook(partial(hook, key=idx))
        
def vis_layer(layer, alex_conv, alex_deconv, top = 5): #Conv2d layer number, alexConv(), alexDeconv()
    
    num_feat_map = alex_conv.feature_maps[layer].shape[1] #[1, 256, 6, 6] --> 256
    feat_map = alex_conv.feature_maps[layer].clone() #cloning the mentioned layer
    mean_lst = []
    for i in range(num_feat_map):
        _map = feat_map[0, i, :, :] # taking each feature map out of 256 and finding its mean.
        mean_lst.append(torch.mean(_map).item())
    
    mean_lst = np.array(mean_lst)
    largest_idx = np.argsort(-1 * mean_lst)[:top]#Finding the top 5 mean from 256 feature maps Output: Ex: [4, 56, 76, 119, 200]
    
    fig = plt.figure(figsize=(30,30))
    counter = 1
    for max_act in largest_idx:    
        max_map = torch.zeros((feat_map.shape), dtype = torch.float32) # max_map is zeros tensor with shape [1, 256, 6, 6]
        feat_int = feat_map[0, max_act, :, :] # the first activated feature maps feat_int = current feature of interest (6, 6)
       
        feat_int = torch.where(feat_int>0, feat_int, torch.zeros(feat_int.shape)) # removing all the values < 0 
        max_map[0, max_act, :, :] = feat_int # assigning back to max_map. max_map still has shape [1, 256, 6, 6]
        deconv_output = alex_deconv(max_map, layer, alex_conv.pool_loci)# passing max_map, Conv2d layer number and switches.
        
        new_img = deconv_output[0].clone().detach().numpy().transpose(1,2,0) # changing the output in required format that can
                                                                               # be plotted.
        new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255.
        
        new_img = new_img.astype(np.uint8)
        
        fig.add_subplot(1, top, counter)
        plt.imshow(new_img)
        plt.title(max_act)
        plt.axis(False)
        counter += 1
    plt.show()


# In[8]:


im_path = 'flam_2.jpg'
image, original = get_image(im_path)
alex_conv = alexConv()
alex_conv.eval()
collect(alex_conv) #call hook first and then pass the image
alexConv_output = alex_conv(image)
alex_deconv = alexDeconv()
alex_deconv.eval()
pred = torch.argmax(alexConv_output,1).item()
label = dicti[pred]


# In[9]:


fig = plt.figure(figsize=(10,10))
fig.add_subplot(121)
plt.imshow(original)
plt.title(label)
plt.axis(False)
plt.show()
layer = 10 #[0, 3, 6, 8, 10]
vis_layer(layer, alex_conv, alex_deconv, top = 5)

