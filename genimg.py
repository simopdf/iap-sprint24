import numpy as np
import time
import os
from PIL import Image
import cv2


# Cartesien <-> Spherique
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(z, np.sqrt(x**2 + y**2))
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


# Projection stereographique
def sphere_to_plane(x_n, y_n, z_n):
    x = 2 * x_n / (1 + z_n)
    y = 2 * y_n / (1 + z_n)
    return x, y


def plane_to_sphere(x, y):
    x_n = 4 * x / (4 + x**2 + y**2)
    y_n = 4 * y / (4 + x**2 + y**2)
    z_n = (4 - x**2 - y**2) / (4 + x**2 + y**2)
    return x_n, y_n, z_n


# Angle <-> Longueur
def f_to_length(angle_degrees):
    angle_radians = np.radians((180 - angle_degrees) / 2)
    x, y = sphere_to_plane(np.cos(angle_radians), 0, np.sin(angle_radians))
    return 2 * x


def length_to_f(length):
    plane_coords = plane_to_sphere(length / 2, 0)
    angle_radians = np.arctan2(plane_coords[2], plane_coords[0])
    angle_degrees = np.degrees(angle_radians)
    return 180 - 2 * angle_degrees


# Taille de l'ecran
def screen_size(resX, resY, length):
    ratio = resX / resY
    width = length / ratio
    return length, width


# Ecran <-> Plan
def plane_to_screen(resX, resY, x, y, f):
    L = f_to_length(f)
    X = (2 * x / L) * resX + resX / 2
    Y = (2 * y / L) * resX + resY / 2
    return X, Y


def screen_to_plane(resX, resY, X, Y, f):
    L = f_to_length(f)
    x = (X - resX / 2) * L / (2 * resX)
    y = (Y - resY / 2) * L / (2 * resX)
    return x, y


# fmt: off
# Rotations
def R3(psi):
    return np.array([[np.cos(psi),   np.sin(psi),  0],
                     [-np.sin(psi),  np.cos(psi),  0],
                     [0,             0,            1]])

def R2(theta):
    return np.array([[1,             0,                0],
                     [0,  np.cos(theta),  -np.sin(theta)],
                     [0,  np.sin(theta),   np.cos(theta)]])

def R1(phi):
    return np.array([[np.cos(phi),   np.sin(phi),  0],
                     [-np.sin(phi),  np.cos(phi),  0],
                     [0,             0,            1]])
# fmt: on


# Soleil <-> Camera
def cam_to_sun(x_n, y_n, z_n, psi, theta, phi):
    vect_cam = np.array([x_n, y_n, z_n])
    R = R3(psi) @ R2(theta) @ R1(phi)
    vect_sun = np.einsum("ijk,il->ljk", vect_cam, R)
    return vect_sun


def sun_to_cam(u_n, v_n, w_n, psi, theta, phi):
    vect_sun = np.array([u_n, v_n, w_n])
    R = (R3(psi) @ R2(theta) @ R1(phi)).T
    vect_cam = np.einsum("ijk,il->ljk", vect_sun, R)
    return vect_cam


# Transfomation de Lorentz
def lorentz(n_x, n_y, n_z, d_theta, d_phi, v):
    # Beta et Gamma
    c = 1
    beta = v / c
    gamma = (1 - beta**2) ** (-1 / 2)

    # Direction de deplacement
    d_x, d_y, d_z = spherical_to_cartesian(1, d_theta, d_phi)
    direction = np.array([d_x, d_y, d_z])

    # Vecteurs t et u
    th = np.array([1, 0, 0, 0])
    tb = np.array([1, 0, 0, 0])  # Pour plus de clarete
    uh = np.array([gamma, *(gamma * beta * direction)])
    ub = np.array([gamma, *-(gamma * beta * direction)])

    # Matrice
    M = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = (
                np.eye(4)[i, j]
                - (uh[i] + th[i]) * (ub[j] + tb[j]) / (1 + gamma)
                + 2 * uh[i] * tb[j]
            )

    # Quadrivecteur photon et application de la transformation
    k = np.array([np.ones(n_x.shape), -n_x, -n_y, -n_z])
    k = np.einsum("ij,jkl->ikl", M, k)

    # W prime et normalisation
    wp = k[0]
    k = k / wp

    return *k[1:], wp

def modulation_couleur(C,w):  		#C = couleur, w = omega de lorentz

	# Transfo sRGB -> lin
    if C/255 <= 0.04045:
	    C_lin = (C/255)/12.92

    
    else:
        C_lin = ( (C/255 + 0.055)/1.055 ) ** 2.4

	# Transfo fréquence (1/w)

    C_lin = C_lin / w
    
	# Transfo lin -> sRGB

    if C_lin <= 0.0031308:
        C_nouveau = 12.92 * C_lin

    else:
        C_nouveau = 1.055 * ( C_lin ** (1/2.4)) - 0.055


    return min(C_nouveau * 255 , 255)

# Couleur associee a chaque direction
def point_to_colour(resX, resY, theta, phi, w):
    
    dark = 100
    light = 200
    res = np.empty((resY, resX), dtype="object")
    for i in range(resY):
        for j in range(resX):
            if (theta[i, j] // 5 + phi[i, j] // 5) % 2:

                color = modulation_couleur(dark,w[i,j])           
                res[i,j] = color,color,color

            else:

                color = modulation_couleur(light,w[i,j])
                res[i,j] = color,color,color

    return res


# Couleur associee a chaque pixel
def pixel_to_colour(resX, resY, f, X, Y, psi, theta, phi, v):
    global check

    x, y = screen_to_plane(resX, resY, X, Y, f)
    x_n, y_n, z_n = plane_to_sphere(x, y)
    if DEBUG:
        print(f"Ecran -> Plan -> Sphere 🗸 ({time.time() - check:.2f}s)")
        check = time.time()

    u_n, v_n, w_n = cam_to_sun(x_n, y_n, z_n, psi, theta, phi)
    if DEBUG:
        print(f"Rotations 🗸 ({time.time() - check:.2f}s)")
        check = time.time()

    x_l, y_l, z_l, r = lorentz(u_n, v_n, w_n, 0, 0, v)
    if DEBUG:
        print(f"Transformations de Lorentz 🗸 ({time.time() - check:.2f}s)")
        check = time.time()

    _, theta, phi = cartesian_to_spherical(x_l, y_l, z_l)
    colour = point_to_colour(resX, resY, np.degrees(theta), np.degrees(phi), r)
    if DEBUG:
        print(f"Cartesien -> Spherique -> Couleurs 🗸 ({time.time() - check:.2f})s")
        check = time.time()

    return colour


# Dessine l'ecran
def paint(nom, resX, resY, f, psi, theta, phi, v):

    # Tableaux de coordonnees
    x = np.linspace(1, resX, resX)
    y = np.linspace(1, resY, resY)
    X, Y = np.meshgrid(x, y)

    # Tableau de couleurs pour chaque pixel
    image_array = np.empty((resY, resX))  
    image_array = pixel_to_colour(resX, resY, f, X, Y, psi, theta, phi, v)


    if DEBUG:
        print(f"Image data 🗸 ({time.time() - check:.2f}s)")

    #On crée un np.array, on reshape pour unpack les tuples, on convertit en image et on reconvertit en np.array (avec PIL) -> on output un array 3D de pixels
    dt = np.dtype([('x', 'u1'), ('y', 'u1'), ('z', 'u1')])
    img = Image.fromarray(image_array.astype(dt), "RGB")

    if DEBUG:
        img.save(str(nom)+".png") #les images ne sont plus sauvegardées en mémoire

    return np.asarray(img)


# Parametres d'execution
DEBUG = False
tau_tot = 20  # s
c = 1  # c
v_fin = 0.999995  # c
resX = 640  # pixels
resY = 480  # pixels
fps = 1

# Execution ----> créer une fonction pour automatiser?
start_time = time.time()

n = tau_tot * fps
tau = np.linspace(0, tau_tot, n)

a = (c / tau_tot) * np.arctanh(v_fin / c)
v = c * np.tanh(a / c * tau)
video = cv2.VideoWriter("video test.avi", 0, fps=fps, frameSize= (resX,resY))  #ici on crée un objet vidéo de OpenCV
for i in range(n):
    check_loop = time.time()
    check = time.time()
    img = paint(i, resX, resY, 90, 0, 0, 0, v[i]) #ici on crée à chaque itération le np.array de pixels
    video.write(img) #on utilise le array pour créer un frame
    

    if DEBUG:
        print(f"Image {i+1} 🗸 ({time.time() - check_loop:.2f}s)")
        print("\n")
    print(f"Progress: {(i+1)/n*100:6.2f}%")

end_time = time.time()

print(f"\nTemps d'excution moyen: {(end_time - start_time)/n:.2f}s")
print(f"\nTemps d'excution total: {(end_time - start_time):.2f}s")