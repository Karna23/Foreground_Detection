### Importing things
import numpy as np
import cv2

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


####Reading image


    #for i in range (1300 , 1800):
img = cv2.imread('AirstripRunAroundFeb2006_1300.bmp')
    #img2= cv2.imread('groundMono'+str(i)+'.bmp')
img1 = img.copy()
#### Do processing
height = img.shape[0]
width = img.shape[1]

result = np.zeros((height, width, 3))

for i in range(0, height):
    for j in range(0, width):
        for k in range(0, 3):
            result[i,j,k] = 9.0 

result1 = result.copy()
### Foreground detection
#img2 = cv2.imread('AirstripRunAroundFeb2006_1500.bmp')

#Dif_Img = img2 - img

lamdas  = [0.5,1,2,3,4,5,6,7]


t = 1
Tc = 100
alphaT = 1/t
m = 5
nv = 0.6
counter = 0
falsealarm = [1]

sensitivity = [1]
actuallabels =[]
predictedlabels = []
        

for lamda in lamdas :
    print("lamda = " , lamda)
    print('Image number : ' )

    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, 3):
                result[i,j,k] = 9.0 

    result1 = result.copy()
    img1 = img.copy()

    for iopen in range(1301, 1800):
        
        img2 = cv2.imread('AirstripRunAroundFeb2006_' + str(iopen) + '.bmp')
        img3 = cv2.imread('groundMono' + str(iopen) + '.bmp')
        Dif_Img = img2 - img1
        #cv2.imshow('image' , img)
        #cv2.imshow('Dif_img' , Dif_Img)

        ## step 1:
        for i in range(0 , height):
            for j in range(0, width):
                if np.all(Dif_Img[i,j]**2 <= lamda**2 * result1[i,j]):
                   # print((Dif_Img[i,j])**2 < lamda * result)
                    Dif_Img[i,j] = [0,0,0]
                else:
                    Dif_Img[i,j] = [255,255,255]

        
        ## step 2:
        
        ct = Dif_Img.copy()

        for i in range( m , height - m ) :
            for j in range( m , width - m ):
                counter = 0
                #print((Dif_Img[i,j]) == 255)
                if ct[i,j ,0] == 255:
                    #print((Dif_Img[i,j]) == 255)
                    for inew in range(i-5, i+5) :
                        for jnew in range( j-5, j +5):
                            if ct[inew,jnew , 0] == 255:
                                counter += 1
                    if counter/(m*m) >= nv:
                        pass
                    else :
                        ct[i,j] = [0,0,0]
        
                        



        ## step 3
        
        if t <= Tc :
            alphaT = 1/t
        
        else :
            alphaT = 1/Tc
            

        t = t+1

        img1 = img1.copy() + (1- (ct.copy()/255)) * ( alphaT * Dif_Img.copy())
       

        result1 = result.copy() + (1 -(ct.copy()/255))* ((1 - alphaT )* ( alphaT* (Dif_Img)**2))



        
        ## calculating TPR and FPR

        
        actuallabels.append(img3[:,:, 1].ravel())
        
        predictedlabels.append((ct[:,:,1].ravel()))

        
        
        print(t)
        ### Show image
                    
        cv2.imshow('Given ground images' , img3)
        cv2.imshow('Obtained Foreground images' , ct)
        cv2.waitKey(10)

    
    actuallabels = np.array(actuallabels)
    actuallabels = actuallabels.ravel()

    predictedlabels = np.array(predictedlabels)
    predictedlabels = predictedlabels.ravel()

    p = confusion_matrix(actuallabels, predictedlabels, labels = [255 , 0])
    print(p)

    falsealarm.append( p[1,0]  / (p[1,0] + p[1,1]))
    sensitivity.append(p[0,0] / (p[0,0] + p [0,1]))

    actuallabels =[]
    predictedlabels = []
    t = 1


####Close and exit

#cv2.waitKey(20)
falsealarm.append(0)
sensitivity.append(0)
cv2.destroyAllWindows()
print("done") 
xline = np.arange(0,1,0.01)
yline = np.arange(0,1,0.01)
plt.scatter(falsealarm, sensitivity)
plt.plot(xline,yline,linestyle = '--')
plt.xlabel('False positive rate')
plt.ylabel ( 'True positive rate')
plt.plot(falsealarm, sensitivity)
plt.show()
    


