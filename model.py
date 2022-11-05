import torch.nn as nn
import torch
import torch.nn.functional as F
#from torchsummary import summary

'''
This script is part of train.py. We define our model as a class
'''


class conv_audio(nn.Module):
    def __init__(self,num_classes):
        super(conv_audio, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1,24,(5,5),padding='same')
        self.conv2 = nn.Conv2d(24,48,(5,5),padding='same')
        self.conv3 = nn.Conv2d(48,48,(5,5),padding='same')
        self.pool1 = nn.MaxPool2d((4,2))
        self.pool2 = nn.MaxPool2d((4,2))
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1))

        self.drop_out_1 = nn.Dropout(p=0.5)
        self.drop_out_2 = nn.Dropout(p=0.5)

        self.flat = nn.Flatten()

        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(1)
        self.norm2 = nn.BatchNorm2d(24)
        self.norm22 = nn.BatchNorm2d(24)
        self.norm3 = nn.BatchNorm2d(48)
        self.norm33 = nn.BatchNorm2d(48)
        self.norm4 = nn.BatchNorm2d(48)

        self.linear1 =  nn.Linear(6912,64)
        self.linear2 =  nn.Linear(64,self.num_classes)

    def forward(self, input):


        x = self.relu(self.norm1(input))

        x = self.pool1(self.relu(self.norm2(self.conv1(x))))

        x = self.relu(self.norm22(x))
        x = self.pool2(self.relu(self.norm3(self.conv2(x))))

        x = self.relu(self.norm33(x))
        x1 = self.conv3(x)
        x2 = self.pool3(x1)
        x = self.norm4(x1)
        x = self.relu(x)

        x = self.flat(x)
        x = self.drop_out_1(x)

        x = self.drop_out_2(self.relu(self.linear1(x)))
        y = self.linear2(x)

        return y,x2

# data = torch.rand(64,1,100,96) # bs, c, t, f
# label = torch.randint(0,19,(64,))

# model = conv_audio(20)
# out = model(data)

# summary(model,(1,100,96))
