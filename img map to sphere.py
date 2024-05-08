# Code pour créer une sphère avec une image
## ça marche mais j'arrive pas à l'implémenter dans le code

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2

# Load your equirectangular projection image
img = cv2.imread("starmap.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define the desired view angle (in degrees)
view_angle_phi = 50  # rotation around the z-axis
view_angle_theta = 90  # rotation around the y-axis

# Create a sphere mesh
r = 1  # radius
phi, theta = np.mgrid[0:2*np.pi:img.shape[1]*1j, 0:np.pi:img.shape[0]*1j]  # spherical coordinates

# Adjust the spherical coordinates based on the view angles
phi += np.radians(view_angle_phi)
theta += np.radians(view_angle_theta)

# Convert spherical coordinates to Cartesian coordinates
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Convert Cartesian coordinates to image coordinates
h, w = img.shape[:2]
x_img = ((np.arctan2(y, x) + np.pi) / (2 * np.pi) * (w - 1)).astype(int)
y_img = ((np.arccos(z / r) / np.pi) * (h - 1)).astype(int)

# Map the image onto the sphere
sphere_image = np.zeros_like(img)

# Reshape x_img and y_img to match the shape of the image
x_img_reshaped = np.reshape(x_img, (h, w))
y_img_reshaped = np.reshape(y_img, (h, w))

sphere_image[..., 0] = img[y_img_reshaped, x_img_reshaped, 0]
sphere_image[..., 1] = img[y_img_reshaped, x_img_reshaped, 1]
sphere_image[..., 2] = img[y_img_reshaped, x_img_reshaped, 2]

print(sphere_image)
# Show or save the resulting image
cv2.imshow('Sphere Image', sphere_image[1035:1515,2010:2650])
cv2.waitKey(0)
cv2.destroyAllWindows()
