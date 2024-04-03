##Projections
import numpy as np

#valeur défault de f:
F = 90


class point:

    def __init__(self, x=None, y=None, z=None):

        self.x = x
        self.y = y
        self.z = z

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


#### Implementation avec numpy
"""def tab_point(x_tab,y_tab,z_tab=None):   ## creates array of points
    if isinstance(z_tab,np.ndarray):
        if len(x_tab) != len(y_tab) or len(y_tab) != len(z_tab):
            raise ValueError("Arrays must have the same length") 
        else:
            return np.array([point(x,y,z) for x,y,z in zip(x_tab,y_tab,z_tab)]) 
    else:
        if len(x_tab) != len(y_tab):
            raise ValueError("Arrays must have the same length")
        else:
            return np.array([point(x,y) for x,y in zip(x_tab,y_tab)])"""
    

def tab_exe(points_array, method_name, *args):  ### allows vectorisation of functions for point class
    method = getattr(point, method_name)
    vect_method = np.vectorize(method)
    return vect_method(points_array, *args)

   
#################################################


class f(point): ######## Enlever?
    def __init__(self,f0=F):
        self.f=f0

    def __repr__(self):
        return f"f = {self.f} °" 
    


    def to_width(self):   #donne les coordonnées de G et D en sachant l'angle f
        f0_rad = np.radians(self.f)
        return [point(-np.sin(f0_rad/2),0,np.cos(f0_rad/2)).stereo(), point(np.sin(f0_rad/2),0,np.cos(f0_rad/2)).stereo()]

    
    def to_angle(self,G,D):
        l = (np.abs(G.invstereo().x)) + np.abs((D.invstereo().x))
        return f(360 - 2*np.degrees(np.arcsin(0.5*l)))
    
    def L(self):  #donne la largeur de l'ecran 
        return 2*np.abs(self.to_width()[0].x)
    
    
#################################################


def screen(self, resX, resY, f_screen = F):
    #plan -> ecran
    x,y,_ = self.stereo().vars()
    L = f(f_screen).L()

    X = (2*x/L)*resX + resX/2
    Y = (2*y/L)*resX + resY/2
    return point(X,Y)


def invscreen(self,resX,resY, f_screen = F):
    #ecran -> plan
    L = f(f_screen).L()
    X = self.x
    Y = self.y

    x = (X - resX/2)*L/(2*resX)
    y = (Y - resY/2)*L/(2*resX)

    return point(x,y)


point.screen = screen # add methods screen and invscreen to point. methods employ f class
point.invscreen = invscreen



#L100 L = f(f_screen).L()

#################################################
    
class Rotation(point):

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
        vect_rot = vect.vect() @ (self.R3 @ self.R2 @ self.R1)

        x,y,z = vect_rot.tolist() #conversion np.array -> liste -> point

        return point(x,y,z)
    
    def to_cam(self,vect): #Tranformation  de (nu,nv,nw) à (nx,ny,nz)
        vect_rot = vect.vect() @ np.linalg.inv(self.R3 @ self.R2 @ self.R1)

        x,y,z = vect_rot.tolist() #conversion np.array -> liste -> point
        
        return point(x,y,z)

def tab_rot(points_array, method_name, *args):  ### allows vectorisation of functions for Rotation class
    method = getattr(Rotation, method_name)
    vect_method = np.vectorize(method)
    return vect_method(points_array, *args)

#####################################################
def cartesian_to_spherical(self):
    r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
    theta = np.arctan2(self.z, np.sqrt(self.x**2 + self.y**2))
    phi = np.arctan2(self.y,self.x)
    return (r, theta, phi)

point.spherical_coords = cartesian_to_spherical


####### Test (à finir)


Rotation(10,20,10).to_sun(point(0,0,1)) ### Testé à la main OK!
Rotation(10,20,10).to_cam(point(-0.05939117, 0.33682409, 0.93969262))

point(1,2,1).screen(720,1920)

H = np.linspace(0,5,5)
K = np.linspace(0,5,5)
L = np.linspace(0,5,5)

#tab = tab_point(H,K,L)

#tab_exe(tab,'stereo')
         
#coords = point()
print(point(H,K,L))

print("Projections: OK !")
     

################################

""" x = np.array([1,2,3])
y = np.array([1,2,3])
z = np.array([1,2,3])

concat = np.concatenate([x,y,z],axis=1)
points = np.split(concat, len(concat[:,0],axis=0)) """

PointArray = np.vectorize(point)

# Create an array of Points
points = PointArray(H, K, L)

print(points)