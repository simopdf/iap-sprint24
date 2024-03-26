##Projections
import numpy as np

#valeur défault de f:
F = 90


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
    


class f(point):
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
    
    
class point(point):

    def screen(self, resX, resY):
        x,y,_ = self.stereo().vars()
        L = f().L()

        X = (2*x/L)*resX + resX/2
        Y = (2*y/L)*resX + resY/2
        return point(X,Y)

    def invscreen(self,resX,resY):
        L = f().L()
        X = self.x
        Y = self.y

        x = (X - resX/2)*L/(2*resX)
        y = (Y - resY/2)*L/(2*resX)

        return point(x,y)


######## à tester


def screen(self, resX, resY):
    x,y,_ = self.stereo().vars()
    L = f().L()

    X = (2*x/L)*resX + resX/2
    Y = (2*y/L)*resX + resY/2
    return point(X,Y)

def invscreen(self,resX,resY):
    L = f().L()
    X = self.x
    Y = self.y

    x = (X - resX/2)*L/(2*resX)
    y = (Y - resY/2)*L/(2*resX)

    return point(x,y)


point.screen = screen # add methods screen and invscreen to point. methods employ f class
point.invscreen = invscreen





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
        return vect.vect() @ (self.R3 @ self.R2 @ self.R1)
    
    def to_cam(self,vect): #Tranformation  de (nu,nv,nw) à (nx,ny,nz)
        return vect.vect() @ np.linalg.inv(self.R3 @ self.R2 @ self.R1)


print(Rotation(10,20,10).to_sun(point(0,0,1))) ### Testé à la main OK!
print(Rotation(10,20,10).to_cam(point(-0.05939117, 0.33682409, 0.93969262)))



# Fonction qui associe à un couple d'angles une couleur



point(1,2,1).screen(720,1920)



         
    


     

