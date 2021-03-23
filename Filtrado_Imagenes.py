import cv2
import numpy as np
import os

globalDir = 'C:/Users/juane/OneDrive/Documentos/Cursos/Vision_computacional/Desarrollo'

img = cv2.imread(os.path.join(globalDir, 'Images','Bananos.jpg'))

# %%
cv2.imshow('Banano',img)

# %% Kernel 3X3, este tipo de kernel sirve para difuminar

kerner_3X3 = np.ones((3,3), np.float32)/(3*3)
output_3X3 = cv2.filter2D(img, -1, kerner_3X3) #Es -1 para conservar en la salida de la imágen la original
cv2.imshow('Promedio 3X3', output_3X3)

# %% Kernel 5X5, este tipo de kernel sirve para difuminar

kerner_5X5 = np.ones((5,5), np.float32)/(5*5)
output_5X5 = cv2.filter2D(img, -1, kerner_5X5)
cv2.imshow('Promedio 5X5', output_5X5)

# %% Kernel 20X20, este tipo de kernel sirve para difuminar

kerner_20X20 = np.ones((20,20), np.float32)/(20*20)
output_20X20 = cv2.filter2D(img, -1, kerner_20X20)
cv2.imshow('Promedio 20X20', output_20X20)

# %% Método Gaussiano con desviación o sigma de 3X3

output_gaussiano_3X3 = cv2.GaussianBlur(img, (3,3), 0) #se pone 0 para que calcule el tamalo de la máscara de forma automática
cv2.imshow('Gaussiando sigma 3X3', output_gaussiano_3X3)

# %% Método Gaussiano con desviación o sigma de 11X11
    
output_gaussiano_11X11 = cv2.GaussianBlur(img, (11,11), 0) #se pone 0 para que calcule el tamalo de la máscara de forma automática
cv2.imshow('Gaussiando sigma 11X11', output_gaussiano_11X11)
 