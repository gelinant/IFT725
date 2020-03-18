# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel
from models.CNNBlocks import *

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau IFT725Net.  Le réseau est constitué de

    1) quelques opérations de base du type « conv-batch-norm-relu »
    2) 1 (ou plus) bloc dense inspiré du modèle « denseNet)
    3) 1 (ou plus) bloc résiduel inspiré de « resNet »
    4) 1 (ou plus) bloc de couches « bottleneck » avec ou sans connexion résiduelle, c’est au choix
    5) 1 (ou plus) couches pleinement connectées

    NOTE : le code des blocks résiduels, dense et bottleneck doivent être mis dans le fichier CNNBlocks.py

'''


class IFT725Net(CNNBaseModel):
    """
    Class that mix up several sort of layers to create an original network
    """

    def __init__(self, num_classes=10, init_weights=True):
        """
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(IFT725Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = DenseBlock(32,32)
        self.conv4 = ResidualBlock(32,32)
        self.conv5 = BottleneckBlock(32,32)
        self.FCL= nn.Sequential(
            nn.Linear(32,500),
            nn.ReLU(inplace=True)
        )



    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c2_dense  = self.conv3(c1,c2)
        c3 = self.conv4(c2_dense)
        c4 = self.conv5(c3)
        c5 = self.FCL(c4)
        return c5




'''
FIN DE VOTRE CODE
'''
