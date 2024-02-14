##Projections
import numpy as np


class point:
    def __init__(self,x0=0.,y0=0.,z0=None):
        self.x= x0
        self.y= y0
        self.z= z0 # attribut d'instance
    def __repr__(self):
        return f"Point({self.x},{self.y},{self.z})"
        
    def vars(self): #on transforme le point en un tuple [.vars()] -> (x,y,z)
        return (self.x,self.y,self.z)

    def to_plane(self):
        return point(2*self.x/(1+self.z),2*self.y/(1+self.z))
    def to_sphere(self):
        return point(4*self.x/(4+self.x**2+self.y**2),4*self.y/(4+self.x**2+self.y**2),(4-self.x**2-self.y**2)/(4+self.x**2+self.y**2))
    


class f(point):
    def __init__(self,f0=None):
        self.f=f0

    def __repr__(self):
        return f"f = {self.f} °" 
    


    def to_width(self):
        f0_rad = np.radians(self.f)
        return [point(-np.sin(f0_rad/2),0,np.cos(f0_rad/2)).to_plane(), point(np.sin(f0_rad/2),0,np.cos(f0_rad/2)).to_plane()]
    
    def to_angle(self,G,D):
        l = (np.abs(G.to_sphere().x)) + np.abs((D.to_sphere().x))
        return f(360 - 2*np.degrees(np.arcsin(0.5*l)))
    
    
     

