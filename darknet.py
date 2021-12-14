import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import utils.network as net_utils
import cfgs.config as cfg


def _make_layers(in_channels, net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        print("net_cfg : ", net_cfg, "isinstance(net_cfg[0], list) : ", net_cfg[0], "list: ",  isinstance(net_cfg[0], list))
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif item == 'M_':
                layers.append(nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)))
            else:
                out_channels, ksize = item
                layers.append(net_utils.Conv2d_BatchNorm(in_channels,
                                                         out_channels,
                                                         ksize,
                                                         same_padding=True))
                # layers.append(net_utils.Conv2d(in_channels, out_channels,
                #     ksize, same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels



class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
#        self.in_channels = 3
#
        net_cfgs = [
            # conv1s
            [(16, 3)],
            ['M', (32, 3)],
            ['M', (64, 3)],
            ['M', (128,3)],
            ['M', (256, 3)],
            ['M', (512, 3)],
            # conv2
            ['M_', (1024, 3)],
            # ------------
            # conv3
            [(512, 3)]
            # conv4
        ]
##
        ##self.conv2d, c1 = _make_layers(3,net_cfgs[0])
        ##self.conv2d_1, c2 = _make_layers(c1,net_cfgs[1])
        ##self.conv2d_2, c3 = _make_layers(c2,net_cfgs[2])
        ##self.conv2d_3, c4 = _make_layers(c3,net_cfgs[3])
        ##self.conv2d_4, c5 = _make_layers(c4,net_cfgs[4])
        ##self.conv2d_5, c6 = _make_layers(c5,net_cfgs[5])
        ##self.conv2d_6, c7 = _make_layers(c6,net_cfgs[6])
        ##self.conv2d_7, c8 = _make_layers(c7,net_cfgs[7])        



        # linear
        ##out_channels = cfg.num_anchors * (cfg.num_classes + 5)
        ##print("out_channels: ", out_channels)
        ##self.conv2d_8 = net_utils.Conv2d(c8,out_channels, 1, 1, relu=False)
        ##self.global_average_pool = nn.AvgPool2d((1, 1))
##    
        

        

##3
#        self.relu = torch.nn.LeakyReLU(0.1, inplace=True)
#        self.maxpool = torch.nn.MaxPool2d(2,2)
#        self.slowpool = torch.nn.MaxPool2d(1,1)
#        self.input0 = nn.Sequential(nn.Conv2d(3,16,3,1,1,bias=False),
#            nn.BatchNorm2d(16,momentum=0.1), self.relu, self.maxpool)
#        self.layer1 = nn.Sequential(nn.Conv2d(16,32,3,1,1,bias=False),
#            nn.BatchNorm2d(32,momentum=0.1), self.relu, self.maxpool)
#        self.layer2 = nn.Sequential(nn.Conv2d(32,64,3,1,1,bias=False),
#           nn.BatchNorm2d(64,momentum=0.1), self.relu, self.maxpool)
#        self.layer3 = nn.Sequential(nn.Conv2d(64,128,3,1,1,bias=False),
#            nn.BatchNorm2d(128,momentum=0.1), self.relu, self.maxpool)
#        self.layer4 = nn.Sequential(nn.Conv2d(128,256,3,1,1,bias=False),
#            nn.BatchNorm2d(256,momentum=0.1), self.relu, self.maxpool)
#        self.layer5 = nn.Sequential(nn.Conv2d(256,512,3,1,1,bias=False),
#            nn.BatchNorm2d(512,momentum=0.1), self.relu, self.slowpool)
#        self.layer6 = nn.Sequential(nn.Conv2d(512,1024,3,1,1,bias=False),
#            nn.BatchNorm2d(1024,momentum=0.1), self.relu)
#        self.layer7 = nn.Sequential(nn.Conv2d(1024,1024,3,1,1,bias=False),
#            nn.BatchNorm2d(1024,momentum=0.1), self.relu)
#        out_channels = cfg.num_anchors * (cfg.num_classes + 5) 
#        self.output0 = nn.Conv2d(1024,out_channels, 1,1,0,bias=False)
        
###3
 
###4       
  #      self.relu = torch.nn.LeakyReLU(0.1, inplace=True)
  #      self.maxpool = torch.nn.MaxPool2d(2,2)
  #      self.slowpool = torch.nn.MaxPool2d(1,1)
  #      self.input0 = nn.Sequential(
  #          nn.Conv2d(3,16,3,1,1,bias=False), nn.BatchNorm2d(16,momentum=0.1), self.relu, self.maxpool,
  #          nn.Conv2d(16,32,3,1,1,bias=False), nn.BatchNorm2d(32,momentum=0.1), self.relu, self.maxpool,
  #          nn.Conv2d(32,64,3,1,1,bias=False), nn.BatchNorm2d(64,momentum=0.1), self.relu, self.maxpool,
  #          nn.Conv2d(64,128,3,1,1,bias=False), nn.BatchNorm2d(128,momentum=0.1), self.relu, self.maxpool,
  #          nn.Conv2d(128,256,3,1,1,bias=False), nn.BatchNorm2d(256,momentum=0.1), self.relu, self.maxpool,
  #          nn.Conv2d(256,512,3,1,1,bias=False), nn.BatchNorm2d(512,momentum=0.1), self.relu, self.slowpool,
  #          nn.Conv2d(512,1024,3,1,1,bias=False), nn.BatchNorm2d(1024,momentum=0.1), self.relu,
  #          nn.Conv2d(1024,512,3,1,1,bias=False), nn.BatchNorm2d(512,momentum=0.1), self.relu,    
  #          )
  #      out_channels = cfg.num_anchors * (cfg.num_classes + 5) 
  #      self.output0 = nn.Sequential(
  #          nn.Conv2d(512,cfg.num_anchors*(5+cfg.num_classes), 1,1,0,bias=False)
  #      )
##4        
##2
        self.layer0, c1 = _make_layers(3,net_cfgs[0])
        self.layer1, c2 = _make_layers(c1,net_cfgs[1])
        self.layer2, c3 = _make_layers(c2,net_cfgs[2])
        self.layer3, c4 = _make_layers(c3,net_cfgs[3])
        self.layer4, c5 = _make_layers(c4,net_cfgs[4])
        self.layer5, c6 = _make_layers(c5,net_cfgs[5])
        self.layer6, c7 = _make_layers(c6,net_cfgs[6])
        self.layer7, c8 = _make_layers(c7,net_cfgs[7])
        out_channels = cfg.num_anchors * (cfg.num_classes + 5) 
        self.output0 = net_utils.Conv2d(c8,out_channels,1,1,relu=False)
##2
      #  self.global_average_pool = nn.AvgPool2d((1,1))
        
   #1
        
   #     self.conv1 = torch.nn.Conv2d(3,16,3,1,1,bias=False)
   #     self.norm1 = torch.nn.BatchNorm2d(16,momentum=0.1)
   #     self.conv2 = torch.nn.Conv2d(16,32,3,1,1,bias=False)
   #     self.norm2 = torch.nn.BatchNorm2d(32,momentum=0.1)
   #     self.conv3 = torch.nn.Conv2d(32,64,3,1,1,bias=False)
   #     self.norm3 = torch.nn.BatchNorm2d(64,momentum=0.1)
   #     self.conv4 = torch.nn.Conv2d(64,128,3,1,1,bias=False)
   #     self.norm4 = torch.nn.BatchNorm2d(128,momentum=0.1)
   #     self.conv5 = torch.nn.Conv2d(128,256,3,1,1,bias=False)
   #     self.norm5 = torch.nn.BatchNorm2d(256,momentum=0.1)
   #     self.conv6 = torch.nn.Conv2d(256,512,3,1,1,bias=False)
   #     self.norm6 = torch.nn.BatchNorm2d(512,momentum=0.1)
   #     self.conv7 = torch.nn.Conv2d(512,1024,3,1,1,bias=False)
   #     self.norm7 = torch.nn.BatchNorm2d(1024,momentum=0.1)
   #     self.conv8 = torch.nn.Conv2d(1024,512,3,1,1,bias=False)
   #     self.norm8 = torch.nn.BatchNorm2d(512,momentum=0.1)
   #     out_channels = cfg.num_anchors * (cfg.num_classes + 5) 
   #     self.conv9 = torch.nn.Conv2d(512,out_channels, 1,1,0)
        
   #     self.relu = torch.nn.LeakyReLU(0.1, inplace=True)
   #     self.maxpool = torch.nn.MaxPool2d(2,2)
   #     self.slowpool = torch.nn.MaxPool2d(1,1)
   #     self.pad = torch.nn.ReflectionPad2d((0,1,0,1))
    #1

        print("out_channels: ", out_channels)
        
        # train
        self.bbox_loss = None
        self.iou_loss = None
        self.cls_loss = None
 #       self.pool = Pool(processes=1)

 #   @property
 #   def loss(self):
 #       return self.bbox_loss + self.iou_loss + self.cls_loss

    def forward(self, im_data, gt_boxes=None, gt_classes=None, dontcare=None,
                size_index=0):

        ##conv2d = self.conv2d(im_data)
        ##conv2d_1 = self.conv2d_1(conv2d)
        ##conv2d_2 = self.conv2d_2(conv2d_1)
        ##conv2d_3 = self.conv2d_3(conv2d_2)
        ##conv2d_4 = self.conv2d_4(conv2d_3)
        ##conv2d_5 = self.conv2d_5(conv2d_4)
        ##conv2d_6 = self.conv2d_6(conv2d_5)
        ##conv2d_7 = self.conv2d_7(conv2d_6)
        ##conv2d_8 = self.conv2d_8(conv2d_7)

        ##global_average_pool = self.global_average_pool(conv2d_8) 

        ##print("global_average_pool.shape: ", global_average_pool.shape) #[16,30,3,3]
        ##print("global_average_pool stride: ", global_average_pool.stride()) #[270,9,3,1]
        ##print("global_average_pool type: ", global_average_pool.type())

   ##1
   #     x = self.conv1(im_data)
   #     x = self.maxpool(self.relu(self.norm1(x)))
   #     x = self.maxpool(self.relu(self.norm2(self.conv2(x))))
   #     x = self.maxpool(self.relu(self.norm3(self.conv3(x))))
   #     x = self.maxpool(self.relu(self.norm4(self.conv4(x))))
   #     x = self.maxpool(self.relu(self.norm5(self.conv5(x))))
   #     x = self.slowpool(self.relu(self.norm6(self.conv6(x))))
   #     x = self.relu(self.norm7(self.conv7(x)))
   #     x = self.relu(self.norm8(self.conv8(x)))
   #     x = self.conv9(x)
   ##1

   ##2,3
        input0 = self.layer0(im_data)
        layer1 = self.layer1(input0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        x = self.output0(layer7)
   ##2,3     
        #global_average_pool = nn.AvgPool2d((1,1))
        #global_average_pool = global_average_pool(out)
        
        


    #    bsize, _, h, w = global_average_pool.size()   
    #    global_average_pool_reshaped = global_average_pool.permute(0,2,3,1).contiguous().view(bsize, -1,
    #                                                               cfg.num_anchors, cfg.num_classes+5)
    
    ##4
     #   x = self.input0(im_data)
     #   x= self.output0(x)
    ##4    
      
       
        #print("global_average_pool: ", global_average_pool.size())
   #     global_average_pool = self.global_average_pool(out)
      #  print("output.size: ", x.size())    
      #  print("x_reshape: ", global_average_pool.size())

        return x
    

if __name__ == '__main__':
    net = Darknet19()
 #   net.utils.load_net('../AlexeyAB/YAD2K/model_data/tiny-yolo-voc.h5',net)
    # net.load_from_npz('models/yolo-voc.weights.npz')
   # net.load_from_npz('models/darknet19.weights.npz', num_conv=18)
