#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:01:50 2017

@author: dinesh
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.widgets as widgets
mutable_object = {} 
def on_press(event):
    print('you pressed', event.button, event.xdata, event.ydata)
    plt.close()
    mutable_object['key'] =  [event.xdata, event.ydata]  
    return([event.xdata, event.ydata])
    
for i in range(3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    filename="/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/0.png"
    im = Image.open(filename)
    arr = np.asarray(im)
    plt_image=plt.imshow(arr)
    cid = fig.canvas.mpl_connect('button_press_event', on_press)
    #rs=widgets.RectangleSelector(
    #    ax, onselect, drawtype='box',
    #    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
    plt.show()
    print(mutable_object)
