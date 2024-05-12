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


# Projection equirectangulaire (compatible VR)
def sphere_to_rectangle(r, theta, phi):
    x_r = r * phi
    y_r = r * (np.pi / 2 - theta)
    return x_r, y_r


def rectangle_to_sphere(r, x_r, y_r):
    theta = y_r / r - np.pi / 2
    phi = x_r / r
    return theta, phi


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


# Ecran <-> Plan
def plane_to_screen(resX, resY, x, y, L):
    X = (2 * x / L) * resX + resX / 2
    Y = (2 * y / L) * resX + resY / 2
    return X, Y


def screen_to_plane(resX, resY, X, Y, L):
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


# Transformation sRGB -> RGB
def modulation_couleur(C, w):
    C_lin = np.where(
        C / 255 <= 0.04045, (C / 255) / 12.92, ((C / 255 + 0.055) / 1.055) ** 2.4
    )

    # Transformation frequence (1/w)
    C_lin = C_lin / w

    # Transformation RGB -> sRGB
    C_nouveau = np.where(
        C_lin <= 0.0031308, 12.92 * C_lin, 1.055 * (C_lin ** (1 / 2.4)) - 0.055
    )

    return np.minimum((C_nouveau * 255).astype(int), 255)


# Couleur associee a chaque direction
def point_to_colour(resX, resY, theta, phi, w):
    # Coleurs des cadrillage
    if COLOR:
        dark = np.array([240, 70, 100])
        light = np.array([180, 180, 180])
    else:
        dark = np.array([80, 80, 80])
        light = np.array([130, 130, 130])

    # array 3D: pour chaque couple de coords (= un pixel) on associe un array de dim (1x3) qui correspond aux val RGB des couleurs
    res = np.zeros([resY, resX, 3])

    # on cree le mask correspondant a la disposition des carres
    mask = (theta // 5 + phi // 5) % 2
    # on extend le mask a la 3eme dimension
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # on extend le tableau w (dim=resY,resX) pour qu'il devienne de dim ResY,ResX,3
    w = np.repeat(w[:, :, np.newaxis], 3, axis=2)
    # on cree un tableau 3D de couleurs RGB a partir du mask --> on va l'utiliser pour calculer les couleurs modulees
    color = np.where(mask, dark, light)

    res = modulation_couleur(color, w)  # couleurs modulees

    return res


# Couleur associee a chaque pixel
def pixel_to_colour(resX, resY, f, X, Y, psi, theta, phi, v):
    global check

    # To VR or not to VR
    if EQUIRECTANGULAR:
        r = 1  # Sphere unitaire
        x, y = screen_to_plane(resX, resY, X, Y, r * np.pi)
        theta_r, phi_r = rectangle_to_sphere(r, x, y)
        x_n, y_n, z_n = spherical_to_cartesian(r, theta_r, phi_r)
        if DEBUG:
            print(f"Ecran -> Rectangle -> Sphere 🗸 ({time.time() - check:.2f}s)")
            check = time.time()

    else:
        x, y = screen_to_plane(resX, resY, X, Y, f_to_length(f))
        x_n, y_n, z_n = plane_to_sphere(x, y)
        if DEBUG:
            print(f"Ecran -> Plan -> Sphere 🗸 ({time.time() - check:.2f}s)")
            check = time.time()

    u_n, v_n, w_n = cam_to_sun(x_n, y_n, z_n, psi, theta, phi)
    if DEBUG:
        print(f"Rotations 🗸 ({time.time() - check:.2f}s)")
        check = time.time()

    x_l, y_l, z_l, r = lorentz(u_n, v_n, w_n, *direction, v)
    if DEBUG:
        print(f"Transformations de Lorentz 🗸 ({time.time() - check:.2f}s)")
        check = time.time()

    _, theta, phi = cartesian_to_spherical(x_l, y_l, z_l)
    color = point_to_colour(resX, resY, np.degrees(theta), np.degrees(phi), r)
    if DEBUG:
        print(f"Cartesien -> Spherique -> Couleurs 🗸 ({time.time() - check:.2f})s")
        check = time.time()

    return color


# Dessine l'ecran
def paint(resX, resY, f, psi, theta, phi, v, nom=None):
    # Tableaux de coordonnees
    x = np.linspace(1, resX, resX)
    y = np.linspace(1, resY, resY)
    X, Y = np.meshgrid(x, y)

    # Tableau de couleurs pour chaque pixel
    image_array = np.empty((resY, resX, 3))
    image_array = pixel_to_colour(resX, resY, f, X, Y, psi, theta, phi, v)

    if DEBUG:
        print(f"Image data 🗸 ({time.time() - check:.2f}s)")

    return image_array


# Parametres d'execution
DEBUG = False
DEBUG_FRAME = False
EQUIRECTANGULAR = True
COLOR = True
tau_tot = 10  # S
c = 1  # C
v_fin = 0.999995  # C
orientaion = (0, 0, 0)  # Orientation camera (psi, theta, phi)
direction = (-np.pi / 2, 0)  # Direction de deplacement (theta, phi)
resX = 320  # Pixels
resY = 200  # Pixels
fps = 30  # Fps

# Execution
start_time = time.time()

n = tau_tot * fps
tau = np.linspace(0, tau_tot, n)

a = (c / tau_tot) * np.arctanh(v_fin / c)
v = c * np.tanh(a / c * tau)

# Cree un objet video de OpenCV
video = cv2.VideoWriter("film.avi", 0, fps, (resX, resY))
for i in range(n):
    check_loop = time.time()
    check = time.time()

    # Cree a chaque iteration le np.array de pixels
    img = paint(resX, resY, 90, *orientaion, v[i], i)
    img = cv2.putText(img,"v = {} c".format(v[i]), (10,15),cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1) # on affiche la vitesse en pourcentage de c

    # Array -> Frame
    video.write(np.uint8(img))

    # Messages debug
    if DEBUG:
        print(f"Image {i+1} 🗸 ({time.time() - check_loop:.2f}s)")
        print(f"Progress: {(i+1)/n*100:6.2f}%")
        print("\n")
    else:
        os.system("clear")  # "cls" sous Windows
        print(f"Progress: {(i+1)/n*100:6.2f}%")

    # Les images ne sont plus sauvegardees en memoire, on peut tout de meme les sauvegarder pour le debug
    if DEBUG_FRAME:
        # RGB -> BRG, car c'est le format de Image.fromarray()
        img[:, :, [0, 2]] = img[:, :, [2, 0]]
        image = Image.fromarray(np.uint8(img))
        image.save(str(i) + ".png")


end_time = time.time()

print(f"\nTemps d'excution moyen: {(end_time - start_time)/n:.2f}s")
print(f"Temps d'excution total: {(end_time - start_time):.2f}s")
