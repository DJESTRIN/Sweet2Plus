# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:52:26 2024

@author: Kenneth Johnosn
"""
import numpy as np
import math
import ipdb
# def get_velocity(self, bin = none):
#     self.xint =self.xs[:-1]
#     self.xplusone = self.xs[1:]
#     self.xdis = (self.xint - self.xplusone)**2
    
#     self.yint =self.ys[:-1]
#     self.yplusone = self.ys[1:]
#     self.ydis = (self.yint - self.yplusone)**2
    
#     self.dis = np.sqrt(self.xdis + self.ydis)
    
    
    

    
def get_velocity(xs,ys,frametime = 0.0372):
    
    distance = []
    for x,xn,y,yn in zip(xs[:-1],xs[1:],ys[:-1],ys[1:]):
        distance.append(math.dist([x,y],[xn,yn]))
        

    velocity = []
    for dis in distance:
        velocity.append(dis/frametime)
    
    return velocity

    
xs = np.array([1,3,4,6,5,3,7,8,3,2,7,8])
ys =np.array([2,4,6,8,3,5,6,2,9,3,6,8])


    
    
    
test = get_velocity(xs,ys)

ipdb.set_trace()