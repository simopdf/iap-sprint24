##Projections
import numpy as np

#valeur défault de f:
F = 90
##Dans ce fichier on crée les 3 classes que nous allons utiliser dans le reste du code 
#Tout d'abord la classe 'point' pour manipuler ds points en 2D ou 3D, pour faciliter tous nos changements de coordonnées
#attention : la classe 'Point' existe déjà dans python mais ses attributs ne correspondent pas à nos besoins alors on en crée une nouvelle selon les caractéristiques que nous voulons


class point:
    def __init__(self,x0=0.,y0=0.,z0=None):
        self.x= x0
        self.y= y0
        self.z= z0 # attribut d'instance
    def __repr__(self):
        return f"Point({self.x},{self.y},{self.z})"
        
    def vars(self): #on transforme le point en un tuple [.vars()] -> (x,y,z)
        return (self.x,self.y,self.z)
    def vect(self):
        return np.asarray(self.vars())

    def stereo(self):
        '''Sphere -> To plane'''
        return point(2*self.x/(1+self.z),2*self.y/(1+self.z))
    def invstereo(self):
        '''Plane -> To sphere'''
        return point(4*self.x/(4+self.x**2+self.y**2),4*self.y/(4+self.x**2+self.y**2),(4-self.x**2-self.y**2)/(4+self.x**2+self.y**2))

    
#################################################
#Ensuite la classe 'f', qui hérite des attributs de 'point'.

class f(point):
    #initialisation de la classe
    def __init__(self,f0=F):
        self.f=f0

    #formatage de l'affichage lors d'un print de notre objet
    def __repr__(self):
        return f"f = {self.f} °" 
    

    #fonctions de changement de changement de référentiel
    def to_width(self):   #donne les coordonnées de G et D en sachant l'angle f
        '''demi-angle -> coordonnées des bords de l'écran sur les x'''
        f0_rad = np.radians(self.f)
        return [point(-np.sin(f0_rad/2),0,np.cos(f0_rad/2)).stereo(), point(np.sin(f0_rad/2),0,np.cos(f0_rad/2)).stereo()]   #formule démontrée (voir l'annexe)

    
    def to_angle(self,G,D):
        '''coordonnées des bords de l'écran sur les x (2 points) -> demi-angle f'''
        l = (np.abs(G.invstereo().x)) + np.abs((D.invstereo().x))   #formule démontrée (voir l'annexe)
        return f(360 - 2*np.degrees(np.arcsin(0.5*l)))
    
    def L(self):  #donne la largeur de l'ecran 
        return 2*np.abs(self.to_width()[0].x)
    
    
#################################################


def screen(self, resX, resY):
    #plan -> ecran
    x,y,_ = self.stereo().vars()
    L = f().L()

    # relations démontrées (voir l'annexe)
    X = (2*x/L)*resX + resX/2
    Y = (2*y/L)*resX + resY/2
    return point(X,Y)


def invscreen(self,resX,resY):
    #ecran -> plan
    L = f().L()
    X = self.x
    Y = self.y

    # relations démontrées (voir l'annexe)
    x = (X - resX/2)*L/(2*resX)
    y = (Y - resY/2)*L/(2*resX)

    return point(x,y)


#Ajout des méthodes 'screen' et 'invscreen' à la classe 'point'. On les ajoute de cette manière apres la définition de la classe car elles utilisent des propriétés de la classe 'f', qui doit être définie après la classe 'point'.
point.screen = screen # add methods screen and invscreen to point. methods employ f class
point.invscreen = invscreen





#################################################
#Ensuite la classe 'Rotation', qui hérite des attributs de 'point'. Elle implémente les rotations de la caméra afin de permettre une vue à 360° de la sphère céleste.
#Les matrices de rotation utilisées ainsi que leur combinaison est démontré (voir l'annexe)

class Rotation(point):

    #Initialiation de la classe
    def __init__(self, psi = 0, theta = 0, phi = 0):
        self.psi= np.radians(psi)
        self.theta = np.radians(theta)
        self.phi = np.radians(phi)
        

        R_3 = np.array([[np.cos(self.psi),np.sin(self.psi),0],
                    [-np.sin(self.psi),np.cos(self.psi),0],
                    [0,            0,         1]])

        R_2 = np.array([[1,          0,                0],
                    [0, np.cos(self.theta),-np.sin(self.theta)],
                    [0, np.sin(self.theta), np.cos(self.theta)]])

        R_1 = np.array([[np.cos(self.phi),np.sin(self.phi),0],
                    [-np.sin(self.phi),np.cos(self.phi),0],
                    [0,            0,         1]])
        
        self.R3= R_3
        self.R2= R_2
        self.R1= R_1
    
    def to_sun(self,vect): #Tranformation  de (nx,ny,nz) à (nu,nv,nw)
        return vect.vect() @ (self.R3 @ self.R2 @ self.R1)
    
    def to_cam(self,vect): #Tranformation  de (nu,nv,nw) à (nx,ny,nz)
        return vect.vect() @ np.linalg.inv(self.R3 @ self.R2 @ self.R1)


#####################################################
def cartesian_to_spherical(self):
    r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
    theta = np.arctan2(self.z, np.sqrt(self.x**2 + self.y**2))
    phi = np.arctan2(self.y,self.x)
    return (r, theta, phi)

point.spherical_coords = cartesian_to_spherical


####### Test


print(Rotation(10,20,10).to_sun(point(0,0,1))) ### Testé à la main OK!
print(Rotation(10,20,10).to_cam(point(-0.05939117, 0.33682409, 0.93969262)))

point(1,2,1).screen(720,1920)



         
coords = point()


     

