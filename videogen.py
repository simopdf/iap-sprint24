import os
import cv2

fps = 30 #nb d'images par seconde

image_folder = "C:\\Users\\Utente\\Documents\\Stage IAP\\video 1min v1"

images = [img for img in os.listdir(image_folder) 
              if img.endswith(".ppm")]


frame = cv2.imread(os.path.join(image_folder, images[0])) 
height, width, layers = frame.shape
video = cv2.VideoWriter("video 1.avi", 0, fps, (width,height)) 

for i,image in enumerate(images):  
        video.write(cv2.imread(os.path.join(image_folder, image)))  
        print("{:.1%} completed".format(i/len(images)))
      
# Deallocating memories taken for window creation 
cv2.destroyAllWindows()  
video.release()  # releasing the video generated 

