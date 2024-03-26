# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:12:12 2024

@author: 21202084
"""
import Projections as P
import numpy as np





# Couleur associe a chaque position
def angle_to_colour(phi, theta):
    phi_tilde = phi // 5
    theta_tilde = theta // 5
    
    if (phi_tilde + theta_tilde) % 2 == 0:
        return 0, 0, 0
    return 255, 255, 255







def pixel_to_colour(resX, resY, f, coords, psi, theta, phi):
    
    sphere = coords.invscreen(resX,resY,f).invstereo()  #passage des coordonnees de l'ecran aux coordonnes sur la sphere
    
    tourne = P.Rotation(psi,theta,phi).to_sun(sphere)  #applique la rotation voulue au point
    
    _,Th, Ph = tourne.spherical_coords()
    
    # x, y = screen_to_plane(resX, resY, X, Y, f)
    # x_n, y_n, z_n = plane_to_sphere(x, y)
    # u_n, v_n, w_n = cam_to_sun(x_n, y_n, z_n, psi, theta, phi)
    # r, theta, phi = cartesian_to_spherical(u_n, v_n, w_n)
    colour = angle_to_colour(np.degrees(Ph), np.degrees(Th))
    return colour

def paint():
    # PPM header
    width = 160
    height = 90
    maxval = 255
    ppm_header = f"P3\n{width} {height}\n{maxval}\n"
    
    # PPM image data
    image = ""
    for i in range(height):
        # Replissage d'une ligne
        for j in range(width-1):
            r, g, b = pixel_to_colour(160, 90, 90, P.point(j,i), 0, 30, 40)
            image = image + f"{r} {g} {b} "

        # Passage a la ligne
        r, g, b = pixel_to_colour(160, 90, 90, P.point(j,i), 0, 30, 40)
        image = image + f"{r} {g} {b}\n"
    
    # Save the PPM image as a binary file
    with open("sphere_celeste_test.ppm", "wb") as f:
        f.write(bytearray(ppm_header, "ascii"))
        f.write(bytearray(image, "ascii"))

paint()
