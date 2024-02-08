##Projections

class point:
    def __init__(self,x0=0.,y0=0.,z0=None):
        self.x= x0
        self.y= y0
        self.z= z0 # attribut d'instance
        
    def vars(self): #on transforme le point en un tuple [.vars()] -> (x,y,z)
        return (self.x,self.y,self.z)

    def to_plane(self):
        return point(2*self.x/(1+self.z),2*self.y/(1+self.z))
    def to_sphere(self):
        return point(4*self.x/(4+self.x**2+self.y**2),4*self.y/(4+self.x**2+self.y**2),(4-self.x**2-self.y**2)/(4+self.x**2+self.y**2))
