# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:12:12 2024

@author: Simone Intingaro
"""
import Projections as P
import numpy as np
from PIL import Image


from datetime import datetime


start_time = datetime.now()

def pixel_to_colour(resX, resY, f, coords, psi, theta, phi):
    
    
    sphere = P.tab_exe(coords,"invscreen", resX,resY,f)
    sphere = P.tab_exe(sphere,"invstereo")


    
    
    tourne = P.tab_rot(P.Rotation(psi,theta,phi), "to_sun", sphere)


    _,Th,Ph = P.tab_exe(tourne, "spherical_coords")
    
    

    color = np.where ((np.degrees(Th)//5+ np.degrees(Ph)//5)%2==0, "0 0 0", "255 255 255" )

    return color



def paint(N):   ## N est le nombre total d'images
    # PPM header
    width = 160
    height = 90
    maxval = 255
    ppm_header = f"P3\n{width} {height}\n{maxval}\n"
    
    H = np.linspace(0,height-1,height)
    W = np.linspace(0,width-1,width)

    ii,jj = np.meshgrid(W,H,indexing='xy')
    
    
    colours_final = []
    for num in range(N):
        for j,i in zip(jj,ii):
            colours_final += pixel_to_colour(width, height, 90, P.tab_point(i,j), 0, num*5, num*15).tolist()
        image = " ".join(str(element) for element in colours_final)


    # Save the PPM image as a binary file
        with open("sphere_celeste_test {}.ppm".format(num), "wb") as f:
            f.write(bytearray(ppm_header, "ascii"))
            f.write(bytearray(image, "ascii"))
            print("{:.1%} completed".format((num+1)/N))





paint(10)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))