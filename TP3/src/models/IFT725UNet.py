# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau IFT725UNet.  Un réseau inspiré de UNet
mais comprenant des connexions résiduelles et denses.  Soyez originaux et surtout... amusez-vous!

'''

class ÌFT725UNet(CNNBaseModel):
    """
     Class that implements a brand new segmentation CNN
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds AlexNet  model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(UNet, self).__init__()
		
		# Encoder
        in_channels = 1  # gray image
        self.conv_encoder1 = self._contracting_block(in_channels=in_channels, out_channels=64)
        self.max_pool_encoder1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder2 = self._contracting_block(64, 128)
        self.max_pool_encoder2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder3 = self._contracting_block(128, 256)
        self.max_pool_encoder3 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder4 = self._contracting_block(256, 512)
        self.max_pool_encoder4 = nn.MaxPool2d(kernel_size=2)
        
		# Transitional block
        self.transitional_block = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
		# Decode
        self.conv_decoder4 = self._expansive_block(1024, 512, 256)
        self.conv_decoder3 = self._expansive_block(512, 256, 128)
        self.conv_decoder2 = self._expansive_block(256, 128, 64)
        self.final_layer = self._final_block(128, 64, num_classes)
		
		# ResidualBlock
		self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        # Encode -----------------------------------------------------------
        encode_block_00 = self.conv_encoder1(x)
        encode_pool_00  = self.max_pool_encoder1(encode_block_00)
		
        encode_block_10 = self.conv_encoder2(encode_pool_00)
        encode_pool_10  = self.max_pool_encoder2(encode_block_10)
		
        encode_block_20 = self.conv_encoder3(encode_pool_10)
        encode_pool_20  = self.max_pool_encoder3(encode_block_20)
		
        encode_block_30 = self.conv_encoder4(encode_pool_20)
        encode_pool_30  = self.max_pool_encoder4(encode_block_30)
		# ------------------------------------------------------------------
		
		# Central encode/decode --------------------------------------------
		cat_layer_30    = self.conv_decoder3(encode_block_30)
		encode_block_21 = torch.cat((cat_layer_30, encode_block_20), 1) 
		
		cat_layer_20    = self.conv_decoder3(encode_block_20)
		encode_block_11 = torch.cat((cat_layer_20, encode_block_10), 1)
		cat_layer_21    = self.conv_decoder3(encode_block_21)
		encode_block_12 = torch.cat((cat_layer_21, encode_block_11, encode_block_10), 1)
		
		cat_layer_10    = self.conv_decoder3(encode_block_10)
		encode_block_01 = torch.cat((cat_layer_10, encode_block_00), 1)
		cat_layer_11    = self.conv_decoder3(encode_block_11)
		encode_block_02 = torch.cat((cat_layer_11, encode_block_01, encode_block_00), 1)
		cat_layer_12    = self.conv_decoder3(encode_block_12)
		encode_block_03 = torch.cat((cat_layer_12, encode_block_02, encode_block_01, encode_block_00), 1)
		# ------------------------------------------------------------------

        # Transitional block -----------------------------------------------
        middle_block    = self.transitional_block(encode_pool_30)
		# ------------------------------------------------------------------

        # Decode -----------------------------------------------------------
        decode_block_31 = torch.cat((middle_block, encode_block_30), 1)
        cat_layer_31    = self.conv_decoder4(decode_block_31)
			
        decode_block_22 = torch.cat((cat_layer_31, encode_block_21, encode_block_20), 1)
        cat_layer_22    = self.conv_decoder3(decode_block_22)
		
        decode_block_13 = torch.cat((cat_layer_22, encode_block_12, encode_block_11,encode_block_10), 1)
        cat_layer_13    = self.conv_decoder2(decode_block_13)
        
		decode_block_04 = torch.cat((cat_layer_13, encode_block_03, encode_block_02, encode_block_01, encode_block_00), 1)
        final_layer = self.final_layer(decode_block_04)
		final_layer += encode_block_01 + encode_block_02 + encode_block_03
		# ------------------------------------------------------------------
		
        return final_layer

    def _contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        Building block of the contracting part
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
		
        return block

    def _expansive_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Building block of the expansive part
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1)
        )
        return block

    def _final_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Final block of the UNet model
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

'''
Fin de votre code.
'''