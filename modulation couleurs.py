"""on a (R,G,B), pour passer en lin on fait des operation sur 
C = R = G = B si on est en monochromatique sinon il faut le faire pour chaque composante"""

def modulation_couleur(C,w):  		#C = couleur, w = omega de lorentz

	# Transfo sRGB -> lin
  if C/255 <= 0.04045:

		C_lin = (R/255)/12.92

	else:

		C_lin = ( (R/255 + 0.055)/1.055 ) ** 2.4

	# Transfo fréquence (1/w)

	C_lin = C_lin / w

	# Transfo lin -> sRGB

	if C_lin <= 0.0031308:
		C_nouveau = 12.92 * C_lin

	else:
		C_nouveau = 1.055 * ( C_lin ** (1/2.4)) - 0.055


  return min(C_nouveau * 255 , 255)



### C_nouveau est compris entre 0 et 1 donc il faut remultiplier par 255. Le maximum reste évidemment 255 
