import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

globalDir = 'C:/Users/juane/OneDrive/Documentos/Cursos/Vision_computacional/Desarrollo'

img = cv2.imread(os.path.join(globalDir, 'Images/Lunar.jpg'))

# %%Modelo HSV define el Tono (HUE), Saturación e Intensidad
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# %%Intensidad
I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# %%Imagen binaria, puede ser 0 negros, 1 255 blancos, también es conocida como mascara que es l objeto de interes
umbral,_=cv2.threshold(I,0,255,cv2.THRESH_OTSU) #Se calcula el umbral de manera automática
mascara=np.uint8((I<umbral)*255)

# %% Histograma de una imagen.

#es el valor que más se repite en una imágen, da información importante de la distribución de grises en una imágen y seleccionar lo que nos interesa
histograma = I.flatten()
plt.hist(histograma, bins=100) #bins es la cantidad de barras que vamos a tener en el histograma 

#Se puede hacer el histograma de cada canal y ver como se esta comportando la imágen
rojo = img[:,:,0].flatten()
verde = img[:,:,1].flatten()
azul = img[:,:,2].flatten()

plt.hist(rojo, bins=1000, histtype='stepfilled', color='red')
plt.hist(verde, bins=1000, histtype='stepfilled', color='green')
plt.hist(azul, bins=1000, histtype='stepfilled', color='blue')

# %% Etiquetado de objetos y selección de interes

output = cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
cantidad_objetos = output[0]
labels = output[1]
stats = output[2]

#Se selecciona el objeto con mayor cantidad de pixels que es nuestro objeto de interes,siempre se coge desde la fila 1 ya que la 0 es por lo general el fondo
objeto_interes = (np.argmax(stats[:,4][1:])+1==labels)
objeto_interes = ndimage.binary_fill_holes(objeto_interes).astype(int) #Se rellena los huecos, se retorna como un tipo entero


# %% Medicioón de área y perímetro en la image

objeto_interes_8b = np.uint8(objeto_interes*255) #Ya no va tener valores entre 0 y 1 sino entre 0 y 255
contours,_ = cv2.findContours(objeto_interes_8b,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Encuentra los contornos de la máscara

cnt=contours[0]

#Perímetro
P=cv2.arcLength(cnt,True)

#Área
A=cv2.contourArea(cnt)
A1 = np.sum(objeto_interes_8b/255)


# %% Convex HULL 

# Encuentra el poligono convexo más pequeño que enciarra al objeto

hull = cv2.convexHull(cnt)
puntosConvex = hull[:,0,:] #puntos del convexhull
m,n = objeto_interes_8b.shape # Obtenemos las dimensiones de la imagen
ar = np.zeros((m,n)) #Creamos la imagen como una matriz de ceros
mascaraConvex = np.uint8(cv2.fillConvexPoly(ar,puntosConvex,1)) #La nueva máscara, 1 es el grosor de la línea que vamos a pintar

# %% Bounding Box rotado

#Este se alinea con la dirección del objeto
rotado = cv2.minAreaRect(cnt) #Calcula el cuadrado con área mínima que cubre el objeto
box = np.int0(cv2.boxPoints(rotado)) #Los puntos de la caja

m,n = objeto_interes_8b.shape
ar = np.zeros((m,n))
mascaraRotado = np.uint8(cv2.fillConvexPoly(ar,box,1))


# %% Boundiung Box recto

#Este se alinea con respecto a los ejes
x,y,w,h=cv2.boundingRect(cnt) 
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1) #GREEN BGR

# %% Resultado Convex HULL, Boundiung Box rotado y rectp

#Boundiung Box recto
cv2.imshow('imagen',img)

#Convex HULL
contours,_=cv2.findContours(mascaraConvex,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),1) #RED BGR

#Bounding Box rotado
contours,_=cv2.findContours(mascaraRotado,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(255,0,0),1) #BLUE BGR
 
#Final
cv2.imshow('imagen',img)

#por lo general el convex HULL y BB rotado sirve para mirar características, para medir que tan convexo es el objeto que estoy midiendo, que no tenga entradas
#BB recto para mostrar el objeto que estoy midiendo, solo para cuestiones de visualización. 

# %% Segmentación del objeto de interes

#Segmentación a color 
segColor = np.zeros((m,n,3)).astype('uint8') #matriz de mxn con 3 canales (BRG) y retorna entero de 8 bits de 0 a 255 
segColor[:,:,0] = np.uint8(img[:,:,0]*objeto_interes) #Componente en rojo
segColor[:,:,1] = np.uint8(img[:,:,1]*objeto_interes) #Componente en verde
segColor[:,:,2] = np.uint8(img[:,:,2]*objeto_interes) #Componente en azul

#Segmentado en gris
segGrey = np.zeros((m,n))
segGrey[:,:] = np.uint8(I*objeto_interes)

#Segmentación final
cv2.imshow('imagen', segGrey)




#cv2.imshow('imagen',img)
#cv2.waitKey(0)
#cv2.destryAllWindows()