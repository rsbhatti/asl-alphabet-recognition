# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:41:29 2019

@author: Rajvindra
"""

def result(predict):
    switch={
            0:'A',
            1:'B',
            2:'C',
            3:'D',
            4:'E',
            5:'F',
            6:'G',
            7:'H',
            8:'I',
            9:'J',
            10:'K',
            11:'L',
            12:'M',
            13:'N',
            14:'Place your hand '
            'inside box',
            15:'O',
            16:'P',
            17:'Q',
            18:'R',
            19:'S',
            20:'T',
            21:'U',
            22:'V',
            23:'W',
            24:'X',
            25:'Y',
            26:'Z'
            }
    return switch.get(predict,"error")
        
    