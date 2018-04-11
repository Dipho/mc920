
# coding: utf-8

# # Trabalho 1 MC920 - Daniel Helu Prestes de Oliveira

# ## Bibliotecas

# In[1]:


import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ## Recepção dos Argumentos e Funções Comuns

# In[14]:


#Recebe o argumento do nome da imagem
imgsource = sys.argv[1]

#Ler imagem
def readimg(imgsource):
    return cv2.imread(imgsource)

#Mostrar imagem
def showimg(img):
    cv2.imshow("Display image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#Criar imagem destino
def createimg(imgsource, flag):
    strSplit = imgsource.split('.')
    imgdest = strSplit[0] + flag + '.' +  strSplit[1]
    return imgdest
    
#Salvar imagem
def writeimg(img, imgdest):
    cv2.imwrite(imgdest,img)
    
#Converter RGB para Gray
def conv_RGB2Gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Gerar contornos
def givecontours(img):
    ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV)
    img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return img, contours


# ## 1.1 Transformação de Cores

# In[5]:


img = readimg(imgsource)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
showimg(img)
imgdest = createimg(imgsource, 'gray')
writeimg(img, imgdest)


# ## 1.2 Contorno dos Objetos

# In[ ]:


img = readimg(imgsource)
gray = conv_RGB2Gray(img)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)
showimg(thresh)
imgdest = createimg(imgsource, 'contour')
writeimg(thresh, imgdest)


# ## 1.3 Extração de Propriedades dos Objetos

# In[21]:


img = readimg(imgsource)
gray = conv_RGB2Gray(img)
gray, contours = givecontours(gray)

print ("Numero de regioes:", len(contours))
for x in range(0, len(contours)):
    M = cv2.moments(contours[x])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroide = (cx,cy)
    print("regiao:", x, "\t\tcentroide:", centroide, "\t\tperimetro:", int(round(cv2.arcLength(contours[x], True),0)), "\t\tarea:", int(cv2.contourArea(contours[x])))
    origem = (cx-3, cy+2)
    cv2.putText(img, str(x), origem, cv2.FONT_HERSHEY_SIMPLEX , 0.25,(0,0,0))

showimg(img)
imgdest = createimg(imgsource, 'region')
writeimg(img, imgdest)


# ## 1.4 Histograma de Area dos Objetos

# In[22]:


img = readimg(imgsource)
gray = conv_RGB2Gray(img)
gray, contours = givecontours(gray)
array = []
peq = 0
med = 0
grd = 0
for x in range(0, len(contours)):
    area = int(cv2.contourArea(contours[x]))
    if(area < 1500):
        peq += 1
    elif(area < 3000):
        med += 1
    else:
        grd += 1
    array.append(area)

print ('Numero de Regioes Pequenas:', peq)
print ('Numero de Regioes Medias:', med)
print ('Numero de Regioes Grandes:', grd)
    
plt.hist(array, 3, facecolor='blue', ec='black')

plt.xlabel('Histograma de Area dos Objetos')
plt.ylabel('Frequencia')
plt.axis([0, 4500, 0, 55])
plt.show()

