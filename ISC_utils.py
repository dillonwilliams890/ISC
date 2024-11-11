# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:13:39 2022

@author: will6605
"""
#%%   
import cv2 as cv
from tracker2024 import *
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import warnings
from scipy.ndimage import uniform_filter1d
from PIL import Image
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
import pathlib
from tensorflow.keras.optimizers import Adam
import ipyplot
import tables
from numpy import diff
from scipy.interpolate import UnivariateSpline
#%%

def KV(x, s,v): # Kelvin voigt fit
    return (s/2)*(x[0]**2-1/(x[0]**2))+(2*v/x[0])*(x[1])
    # return (s/2)*(x[0]**2-1)*(x[0]**2+1/x[0]**4)+(2*v/x[0])*(x[1])

def read_h5(file):
     with h5py.File(file, 'r') as f:
         frames = f['dataset'][()]
     return(frames)

def func(x, a, b, c): # Hill sigmoidal equation from zunzun.com
    return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b)) 

def exponential_fit(x, eo, tau, einf):
    return eo * np.exp(-x/tau) + einf

def line_exp(x,uo,tau):
    return uo-x/tau

def mass(b,pt):
    x=pt[0]; y=pt[1]
    parea=5.6
    Hb=(b[int(y-30):int(y+30), int(x-30):int(x+30)])
    # print(Hb)
    base=Hb[1,1]*0.85
    kernel = np.ones((3,3),np.uint8)
    mask = cv.GaussianBlur(Hb, (5, 5), 0)
    mask=cv.erode(mask,kernel,iterations=1)
    _, mask = cv.threshold(mask, base, 255, cv.THRESH_BINARY)
    masked = cv.bitwise_and(Hb, mask)
    average = masked[np.nonzero(masked)].mean()
    Hbnorm=Hb#/average
    Hbnorm[Hbnorm <= 0] = 0.01
    hbmass=((parea*(10**-8)*64500*np.sum(np.sum((-np.log10(Hbnorm))))))
    return hbmass

def P50(oxy, sats):
    xData = oxy
    yData = sats
    # these are the same as the scipy defaults
    initialParameters = np.array([1.0, 1.0, 1.0])
    # do not print unnecessary warnings during curve_fit()
    warnings.filterwarnings("ignore")
    # curve fit the test data
    fittedParameters, pcov = curve_fit(func, xData, yData, initialParameters)
    modelPredictions = func(xData, *fittedParameters) 
    absError = modelPredictions - yData
    xModel = np.linspace(min(xData), max(xData),num=1000)
    yModel = func(xModel, *fittedParameters)
    f = np.full((1000, ), 0.5)
    x = np.linspace(min(xData), max(xData),num=1000)
    g = func(xModel, *fittedParameters)
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    p50=x[idx[0]]
    return p50

def is_outlier(s):
    lower_limit = s.mean() - (s.std() * 2)
    upper_limit = s.mean() + (s.std() * 2)
    return ~s.between(lower_limit, upper_limit)

#Basic roi function to select the four rois necessary for processing
def get_roi(file):
    f = h5py.File(file, 'r')
    key = list(f.keys())[0]
    frame1 = (f[key][0])
    r1 = cv.selectROI(frame1) # squeeze section
    r2 = cv.selectROI(frame1) # sat measurement section
    r3 = cv.selectROI(frame1) # LED measurement section
    # print(r2)
    cv.destroyAllWindows()
    return [r1, r2, r3]

# Object tracking function that finds cells and passes info on their contours, location, and the saturation
#@profile
def track(file, r1, r2, r3, oxy):
    raw = h5py.File(file, 'r')
    key = list(raw.keys())[0]
    myset=raw['dataset']
    size=myset.shape[0]
    #raw_array = read_h5(file)
    #frames=[]
    tracker = EuclideanDistTracker() #distnace tracker that ids cells, from custom tracker_sat file
    frame1 = (raw[key][0]) # get the first frame
    frame2 = (raw[key][1])  # get the second frame
    #need a control frame under both flickering LEDs
    control_1 = frame1
    control_1[control_1 < 1] = 1
    control_2 = frame2
    control_2[control_2 < 1] = 1
    #The brighter frame in the 430nm frame the dimmer is 410
    if np.mean(control_1) > np.mean(control_2):
        control_430=control_1
        control_410=control_2
    else:
        control_430=control_2
        control_410=control_1
    
    #Set background removal object for later
    object_detector = cv.createBackgroundSubtractorMOG2()
    circ=[]
    deform=[]
    parea=5.6 #camera pixel area
    #Molecular absorbtion coefficints of something like that ~chemistry~
    w430_o = 2.1486*(10**8)
    w430_d = 5.2448*(10**8)
    w410_o = 4.6723*(10**8)
    w410_d = 3.1558*(10**8)
    cell_img=[]
    #this is the loop that populates a list with np arrays of all the frames (this is the slow part)
    extinct=[]
    #cell_img=[]
    LED=[0,0]
    light = False #boolean, true for 430nm, false for 410, should switch every frame is hardware is correct
    p=2
    #While loop to iterate through the frames and do image processing
    while True:
        if p>=(size):
        # if p>=(500):
            break
        frame= (raw[key][p])
        if frame is None :
            break

        # Extract Region of interest
        roi = frame
        img = np.zeros_like(roi)
        #Get pixel intesity inLED roi to figure out which light is on
        LED.append(np.mean(frame[int(r3[1]):int(r3[1]+r3[3]), int(r3[0]):int(r3[0]+r3[2])]))
        #Skip frames if flickering goes wrong
        if LED[p]<5:
               p=p+2
               LED.append(LED[p-3])
        elif np.abs(LED[p]-LED[p-1])<3:
                p=p+1
        #If no anaylze the frame
        else:
            if p >= 4:
                #Normalize the frame with control frames
                if LED[p]>LED[p-1]:
                    light = True
                    img=(roi-control_430) 
                    img_sat=(roi/control_430) 
                    a=255    
                else:
                    light = False
                    img=(roi-control_410) 
                    img_sat=(roi/control_410) 
                    a=255     
            
            # mimg=(a-img)
            # mimg=np.zeros_like(img)
            # mimg=cv.normalize(img,  mimg, 0, 255, cv.NORM_MINMAX)
            # im = mimg.astype(np.uint8)
            # bkg = object_detector.apply(im)
            # gauss = cv.GaussianBlur(im, (5, 5), 0)
            im=255-img 
            im = img.astype(np.uint8)
            _, mask = cv.threshold(im, 200, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)
            kernel = np.ones((9,9),np.uint8)
            opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
            
            # thresh= 255 - mask
            # thresh=mask
            contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL ,cv.CHAIN_APPROX_SIMPLE) 
            detections = []
            
            #Loop through the contours found in the image and record the ones of cells
            for cnt in contours:
                # Calculate area and remove small elements
                area = cv.contourArea(cnt)
                x, y, w, h = cv.boundingRect(cnt)   
                if area > 400 and area < 4000 and x>r2[0]-50 and len(cnt)>5:
                    #Get perimiter and area to calculate circularity
                    hull = cv.convexHull(cnt)
                    perimeter = cv.arcLength(hull, True)  
                    # x, y, w, h = cv.boundingRect(cnt)   
                    area = cv.contourArea(hull, True)
                    circ = 2*np.sqrt(np.pi*area)/perimeter
                    rect = cv.minAreaRect(cnt)
                    x, y, w, h = cv.boundingRect(cnt) 
                    (cx, cy), (width, height), angle = rect
                    ellipse = cv.fitEllipse(cnt)
                    (cx, cy), (width, height), angle = ellipse
                    # box = cv.boxPoints(rect)
                    # box = np.int0(box)
                    # minor=height
                    # major=width
                    minor=min(width, height)
                    major=max(width, height)
                    #(xx,yy),(minor,major),angle = cv.fitEllipse(cnt)
                    # if angle < 20 or angle > 160:
                    #     EI=(major-minor)/major
                    # else:
                    #     EI=(minor-major)/major
                    # if angle > 170:
                    #     angle = angle -180
                    detections.append([x, y, w, h, hull, area, light,  major, minor, angle])

            # 2. Object Tracking, basic distance tracker takes in countour info and tracks moving cells
            boxes_ids = tracker.update(detections)
            if len(boxes_ids)>0:
                vol=np.zeros((boxes_ids[-1][4]+1000, 1))
            else:
                vol=[]
               
            for box_id in boxes_ids:
                #pts=[]
                Hb=[]
                x, y, w, h, id, hull, cx, cy, area, light, major, minor, angle = box_id
                # cimg = np.zeros_like(img)
                vol[id]=np.add(vol[id], area)
                #Put a box around the cell
                Hb=(img_sat[int(cy-20):int(cy+20), int(cx-20):int(cx+20)])
                # print(cx)
                if len(Hb)>20 and area>400 and cx>40:
                #     hb_mask=np.zeros_like(Hb)
                # # print(len(hb_mask))
                #     cv.drawContours(hb_mask, [hull], -1,(1,1,1), -1)
                #     Hb = Hb*hb_mask
                #     # Hb = cv.bitwise_and(Hb, mask)
                    Hb[Hb <= 0] = 0.01
                    extinct.append([0])
                #     #for the given cell calcuate the light absorbtion in that box/frame
                    extinct[id].append(parea*(10**-8)*64500*np.sum(np.sum((-np.log10(Hb)))))
                #     # print(id)
                #     # print(extinct[id])
                    # hbmass=mass(img,[cx,cy])
                    # extinct.append([0])
                    # extinct[id].append(hbmass)
                else:
                    extinct.append([0])
                    extinct[id].append(0)
                #Use boolean light to know which light was on during that frame and get that frames absorbtion and the provious one to calculate saturation
                if light == True:
                    f=extinct[id][-1] #430
                    e=extinct[id][-2] #410
                else:
                    e=extinct[id][-1] #410
                    f=extinct[id][-2] #430

                #Set absorbtion values to equation constants
                a=w410_d
                b=w410_o
                c=w430_d
                d=w430_o
                
                #Calcuate mass of oxygenated and deoxygenated hemoglobin
                Mo=(a*f-e*c)/(a*d-b*c)
                Md=(e*d-b*f)/(a*d-b*c)

                #Record saturation is cell is the the roi
                if np.max(vol)>4000:
                    saturation=10000
                    Hgb=-1
                elif cx < int(r1[2]+r1[0]) and cx > int(r2[0]) and abs(Mo) > 0 and abs(Md) > 0:
                    saturation = Mo/(Mo+Md)
                    Hgb=(Mo+Md)
                    # print('mass')
                    # print(e)
                    # print(f)
                    # print('sat')
                    # print(saturation)
                else:
                    saturation = -1
                    Hgb=-1
                if cx < int(r2[0]+r2[2]) and cx > 50 and cy > 50 and cy < 270:
                    
                    cell_mask = np.zeros_like(roi)
                    cv.drawContours(cell_mask, [hull], -1,(255,255,255), -1)
                    # apply mask to input image
                    masked = cv.bitwise_and(roi, cell_mask)
                    # white_mask = ~mask
                    # final = cv.bitwise_or(masked, white_mask)
                    # crop 
                    
                    cell_img=masked[int(cy-50):int(cy+50), int(cx-50):int(cx+50)]
                else:
                    cell_img =np.nan
                #print(saturation)
                #Save circularity, saturation, area, and location for a given cell tagged to it's specific id
                deform.append([oxy, id, circ, cx, p, area, saturation, cell_img, Hgb, major, minor, angle, w, h])
                
            # # For debugging ###
            #     EI=w/h
            #     # cv.putText(roi, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            #     # box = cv.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
            #     # box = int(box)
            #     cv.ellipse(roi, ellipse, (255, 255, 255), 2)
            #     # rminor = min(width, height) / 2
            #     # xtop = cx + math.cos(math.radians(angle)) * rminor
            #     # ytop = cy + math.sin(math.radians(angle)) * rminor
            #     # cv.line(roi, (int(xtop), int(ytop)), (int(cx), int(cy)), (255, 255, 255), 2)
            #     # cv.drawContours(roi,[box],0,(0,0,255),2)
            #     # cv.drawContours(roi, [hull], -1, (255, 255, 255), 1)
            #     cv.putText(roi, str('%f' %saturation), (x, y - 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            #     # cv.putText(roi, str('%f' %angle), (x, y -60), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
              
            # cv.imshow("roi", roi)
            # cv.waitKey(0)

        
            p=p+1
    # cv.destroyAllWindows()
    return deform


#This function takes the cell info from previous function and calculates paramaters, not going to comment too much since it basically just puts info from previous function into dataframes
def shape(deform, r1, r2, r3, oxy):
    value=[]
    df = pd.DataFrame (deform, columns = ['oxy', 'cell', 'circ', 'cx','time','area','saturation','cell_img','hemoglobin', 'major', 'minor', 'angle', 'w', 'h'])
    cells = {k: v for k, v in df.groupby('cell')}
    
    #Go through the info of each cell and calcuate total deformation and median saturation
    for i in range(len(cells)):
        tau=0
        tau_v=0
        sat=[]
        dub = False
        rest=[]
        squeeze=[]
        squeezemajor=[]
        squeezeminor=[]
        recovery=[]
        recoverymajor=[]
        recoveryminor=[]
        tau=np.nan
        tauparm=[]
        Lmajor=[]
        Lminor=[]
        dx=[]
        dt=[]
        ux=[]
        x=[]
        t=[]
        y=[]
        Dij=[]
        tauEID=[]
        time=[]
        start_deform=[]
        cell_shear=[]
        cell_steady=[]
        transit=[]
        angle=[]
        location=[]
        major=[]
        minor=[]
        EI=[]
        images=[]
        deg=[]
        for index, c in cells[i].iterrows():
            if c['saturation']>10:
                dub=True
            # if c['hemoglobin']>-0.5 and c['hemoglobin']<1.5:
            #     hgb.append(c['hemoglobin'])
            if c['cx'] > int(r1[0]+r1[2]): #not using this one right now
                rest.append(c['circ'])
            if c['cx'] > int(r1[0]) and c['cx'] < int(r1[0]+r1[2]-20) and c['saturation']>-0.8: #add circulatiry and EI if the cell was in the squeeze section
                squeeze.append(c['circ'])
                squeezemajor.append(c['major'])
                squeezeminor.append(c['minor'])
                transit.append(c['time'])
                location.append(c['cx'])
                cell_steady.append(c['area'])
            if c['cx'] < int(r2[0]+r2[2]) and c['cx'] > int(r2[0]) and c['saturation']>-0.8 and c['saturation']<1.5: #add circularity and EI if cell was after squeeze
                recovery.append(c['circ'])
                recoverymajor.append(c['major'])
                recoveryminor.append(c['minor'])
                angle.append(c['angle'])
                sat.append(c['saturation'])
            # if c['cx'] < int(r1[0]+r1[2]/3) and c['cx'] > int(r2[0]) and c['saturation']>-0.8: #add circularity and EI if cell was after squeeze
            if  c['cx'] > int(r2[0]) and c['cx'] < int(r1[0]+r1[2]-20): #add circularity and EI if cell was after squeeze
                tauparm.append(c['circ'])
                Lmajor.append(c['major'])
                Lminor.append(c['minor'])
                dt.append(c['time'])
                dx.append(c['cx'])
            if np.isnan(c['cell_img']).any() == False and c['cx'] < int(r2[0]+r2[2]) and c['cx'] > int(r2[0]):
                images.append(c['cell_img'])
            if c['cx'] < int(r1[0]+20) and c['cx'] > int(r1[0]-120) and c['saturation']>-0.8: #add circularity and EI if cell was right after squeeze
                # tauparm.append((c['major']/2-c['minor']/2)/(c['major']/2+c['minor']/2))
                EI.append(c['w']/c['h'])
                major.append(c['major'])
                minor.append(c['minor'])
                time.append(c['time'])
                deg.append(c['angle'])
        if  dub==False and len(squeeze) >= 4 and len(recovery) >=5 and len(sat)>=5 and len(images)>4 and len(major)>1: #make sure cell was sampled enough to give good data
            if np.mean(images[-1])>np.mean(images):
                cell_im430=images[-1]
                cell_im410=images[-2]
            else:
                cell_im430=images[-2]
                cell_im410=images[-1]
            cell_image430 = cv.normalize(cell_im430, None, 0, 255, norm_type=cv.NORM_MINMAX)
            cell_image410 = cv.normalize(cell_im410, None, 0, 255, norm_type=cv.NORM_MINMAX)
            cell_image3=cell_image430/2+cell_image410/2
            #R=sum(rest)/np.count_nonzero(rest)
            D=np.median(squeeze)
            Rc=np.median(recovery)
            D_Rc=abs(D-Rc) #calculate change in circularity
            majorD=np.median(squeezemajor)
            minorD=np.median(squeezeminor)
            majorRC=np.median(recoverymajor)
            minorRC=np.median(recoveryminor)
            degrees=np.median(angle) 
            diff = np.diff(deg)
            change=max(abs(diff))
            ext=np.max(EI)-np.min(EI)
            saturation=np.median(sat)
            #calculate taylor deformation
            # if D_Rc < 0.06: #If low deformation then major and semimajor axis stay correct, undo axis switch
            EID=((majorD/2)-(minorD/2))/((majorD/2)+(minorD/2))
            EIRC=((majorRC/2)-(minorRC/2))/((majorRC/2)+(minorRC/2))
            eo=EID-EIRC
            einf=EIRC
            Ta=abs(EID-EIRC)
            if Ta< 0.1:
                if ext>0.8:# and change>20:  #If low deformation then major and semimajor axis stay correct, undo axis switch
                    EIRC=((minorRC/2)-(majorRC/2))/((majorRC/2)+(minorRC/2))
                    Ta=abs(EID-EIRC)
                    e=1
                elif  ext>0.5 and D_Rc>0.01:   #If low deformation then major and semimajor axis stay correct, undo axis switch
                    EIRC=((minorRC/2)-(majorRC/2))/((majorRC/2)+(minorRC/2))
                    Ta=abs(EID-EIRC)
                    e=1
                else:
                    Ta = Ta
                    e=0
            # if len(tauminor) > 15:
            for j in range(len(dx)-1):
                ux.append(-140*(10**-6)*(dx[j+1]-dx[j]))
                x.append(-dx[j]*(0.014*10**-6))
                t.append(j/1000)
                lamb=Lmajor[j]/majorRC
                y.append(lamb)
                Dij.append(((Lmajor[j]/2)-(Lminor[j]/2))/((Lmajor[j]/2)+(Lminor[j]/2)))
            #shear modulus calc
            mu=0.00826 #[kg/m-s]
            u=-140*(location[-1]-location[0])/(transit[-1]-transit[0]) #[um/s]
            w_d=majorD*.14 #[um]
            h_d=minorD*.14 #[um]
            w_rc=majorRC*.14 #[um]
            h_rc=minorRC*.14 #[um]
            thick=0.05 #[um]
            l=(w_d-w_rc)
            d=7.76-h_d
            d=.5
            shear=mu*(u/(d)) #[N/m^2]
            gamma=(np.arctan(l/(h_d/2)))
            G=thick*(shear/gamma) #[uN/m]
            poisson=0.5
            E=2*(shear/gamma)*(1+poisson)
            speed=u*(10**-6)
            # area=.0196*cell_steady
            area_shear=.0196*(np.median(cell_shear))
            area_steady=.0196*(np.median(cell_steady))
            e=(area_shear-area_steady)/area_steady
            #w=sum(width)/np.count_nonzero(width)
            # e=2*(np.sqrt(area/np.pi))/(6.5) #confinment parameter
            label=[]
            value.append([oxy,D_Rc,Ta,area_steady,u,l,cell_image3,ext,x, t, y, ux, Dij,w_rc, saturation,label])
    return value

#this function normalizes based on cell width
def fractions(data, cluster):
    def fit(data, x, y):
        X = data[x]#-(np.mean(data2.loc[data2['oxy'] == 0]['w'])) # here we have 2 variables for the multiple linear regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example
        Y = data[y]#+(np.mean(data2.loc[data2['oxy'] == 0]['D_Rc']))
        X = sm.add_constant(X) # adding a constant
        mean=np.mean(data[y])
        model = sm.OLS(Y, X).fit()
        coef=model.params[x]
        const=model.params.const
        data[y] = mean*data.apply(lambda row: ((row[y]))/(coef*(row[x])+const), axis=1)
        return data

    data3=data.copy(deep=True)
    p50=[]
    LPF=[]
    data3['sat_norm']=(data3['saturation']-np.mean(data3.loc[data3['oxy'] == 0]['saturation']))
    data3['sat_norm']=(data3['sat_norm'])/np.mean((data3.loc[data3['oxy'] == 21]['sat_norm']))    
    data3['sat_norm']=data3['saturation']
    SO2=[np.mean((data3.loc[data3['oxy'] == 0]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 2]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 3]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 4]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 5]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 7]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 12]['sat_norm'])),np.mean((data3.loc[data3['oxy'] == 21]['sat_norm']))]
    # print(SO2)
    #data3['def']=data3['Ta']/(data3['area']*data3['speed'])
    data3 = data3[data3['Ta'] < 0.5]
    data3 = data3[data3['Ta'] >= 0]
    data3 = data3[data3['sat_norm'] < 1.1]
    data3['D'] = data3['Ta']
    print(len(data3))
    oxy=[0,2,3,4,5,7,12,21]
    # oxy=[0]
    N=len(oxy)
    LPF = []
    num_polys=[]
    num_solys=[]
    polys = pd.DataFrame()
    solys = pd.DataFrame()
    sort=pd.DataFrame()
    for i in range(len(oxy)):
        # df = data3.loc[data3['oxy'] == oxy[i]][['oxy','D','sat_norm','area','speed','cell_img3','label']]
        df = data3.loc[data3['oxy'] == oxy[i]][['oxy','D', 'Ta','sat_norm','area','speed', 'EI','x', 't', 'y', 'ux', 'Dij','majorRC','cell_img3','label']]
        data = data3.loc[data3['oxy'] == oxy[i]][['Ta','sat_norm']]
        # Convert DataFrame to matrix
        mat = data.values
        # Using sklearn
        km = KMeans(n_clusters=cluster[i])
        km.fit(mat)
        # Get cluster assignment labels
        df['cl'] = km.labels_
        if cluster[i]>1:
            df1 = df[df['cl'] == 1]
            df2 = df[df['cl'] < 1]
            df3 = df[df['cl'] == 2]
            df4 = df[df['cl'] == 3]
            # df5 = df[df['cl'] == 4]
            j = [df1,df2,df3,df4]
            j=sorted(j, key=lambda x: x['D'].mean())
            if cluster[i] == 2:
                polys=pd.concat([polys, j[0]])
                solys=pd.concat([solys, j[1]])
            elif cluster[i] == 3:
                polys=pd.concat([polys, j[0]])
                solys=pd.concat([solys, j[2]])
                if (j[1]['D'].mean())>0.1:
                    solys=pd.concat([solys, j[1]])
                else:
                    polys=pd.concat([polys, j[1]]) 


        else:
            if (df['D'].mean())>0.1:
                solys=pd.concat([solys, df])
            else:
                polys=pd.concat([polys, df]) 
        df.plot.scatter('D', 'sat_norm', c='cl', colormap='gist_rainbow') 
        # df.plot.scatter('D', 'sat_norm', c='e', colormap='gist_rainbow')       
    if np.max(clusters) >1:
        solys['sort']=1
        polys['sort']=0
        polys=fit(polys, 'speed', 'D')
        polys=fit(polys, 'area', 'D')
        solys=fit(solys, 'speed', 'D')
        solys=fit(solys, 'area', 'D')
        SO2s=[0,np.mean((solys.loc[solys['oxy'] == 2]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 3]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 4]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 5]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 7]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 12]['sat_norm'])),np.mean((solys.loc[solys['oxy'] == 21]['sat_norm']))]
        SO2p=[np.mean((polys.loc[polys['oxy'] == 0]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 2]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 3]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 4]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 5]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 7]['sat_norm'])),np.mean((polys.loc[polys['oxy'] == 12]['sat_norm'])),1]
        SO2sstd=[0,np.std((solys.loc[solys['oxy'] == 2]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 3]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 4]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 5]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 7]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 12]['sat_norm'])),np.std((solys.loc[solys['oxy'] == 21]['sat_norm']))]
        SO2pstd=[np.std((polys.loc[polys['oxy'] == 0]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 2]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 3]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 4]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 5]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 7]['sat_norm'])),np.std((polys.loc[polys['oxy'] == 12]['sat_norm'])),1]
        oxys = oxy
        oxyp= oxy
        # p50t=P50(oxy,SO2)
        soly_nans = np.isnan(SO2s)
        SO2s=[d for (d, remove) in zip(SO2s, soly_nans) if not remove]
        oxys=[d for (d, remove) in zip(oxys, soly_nans) if not remove]
        poly_nans = np.isnan(SO2p)
        SO2p=[d for (d, remove) in zip(SO2p, poly_nans) if not remove]
        oxyp=[d for (d, remove) in zip(oxyp, poly_nans) if not remove]
        # p50s=P50(oxys,SO2s)
        # p50p=P50(oxyp,SO2p)
        # p50=[p50s, p50p, p50t]
        p50=[]
        sort=pd.concat([polys,solys])
        for i in range(len(oxy)):
            if len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==0]) > 0:
                LPF.append(len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==0])/len(sort.loc[sort['oxy'] == oxy[i]]))
                num_polys.append(len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==0]))
                num_solys.append(len(sort.loc[sort['oxy'] == oxy[i]].loc[sort['sort'] ==1]))
            else:
                LPF.append(0)
    else:
        solys=fit(solys, 'speed', 'D')
        solys=fit(solys, 'area', 'D')
        polys=[]
        SO2s=[]
        SO2p=[]
        SO2sstd=[]
        SO2pstd=[]
        num_solys=[]
        num_polys=[]
        sort=solys
    return data3, sort, SO2, LPF, polys, solys, SO2s, SO2p, p50, SO2sstd, SO2pstd, num_polys, num_solys

#Function to run all the functions for a video
def process(file, r, oxy):
    deform = track(file, r[0], r[1], r[2], oxy)
    data=shape(deform, r[0], r[1], r[2], oxy) 
    return data

def anaylsis(path, oxygens, files):
    roi=[]
    data=[]
    for i in range(len(oxygens)):
        r=get_roi(path+files[i])
        roi.append(r)

    for i in range(len(oxygens)):
        value=process(path+files[i], roi[i], oxygens[i])
        data=data+value
    datas = pd.DataFrame (data, columns = ['oxy', 'D_Rc', 'Ta', 'area','speed','l','cell_img3','EI','x', 't', 'y', 'ux', 'Dij','majorRC', 'saturation','label'])  
    datas = datas[np.abs(datas['saturation']-datas['saturation'].mean()) <= (2*datas['saturation'].std())]    
    return datas

def strain(df, p0):
    def fit(x,y,ux,t,majorRC):
        popt=[]
        if len(x)>10:# and y[-1]<y[1]:
            # plt.plot(t, x)
            # plt.show()
            correct=np.min(y)
            # blur = [i * 150 for i in ux]
            # # blur=[i / majorRC for i in blur]
            x =[-1 *i for i in x]
            y =[i / correct for i in y]
            # x=-1*x
            x=np.flip(x)
            ux=np.flip(ux)
            y=np.flip(y)
            data = {'x': x[0:-2],'t': t[0:-2], 'u': ux[0:-2],  'y': y[0:-2], }
            df = pd.DataFrame(data)
            # df=df.loc[df['y'] <3]
            # df=df.loc[df['y'] >0.5]
            # df=df.rolling(4).mean()
            df=df.loc[df['x']<9e-6]
            df=df.loc[df['x']>2.5e-6]
            # print(df)
            # df=df.loc[df['t']>0.008]
            # df=df.loc[df['t']<0.015]
            df.reset_index(drop=True, inplace=True)

            # ts = np.linspace(df['t'].values[0], df['t'].values[-1], 100)
            # splx = UnivariateSpline(df['t'], df['x'],s=.001)
            # splu = UnivariateSpline(df['t'], df['u'],s=.001) 
            # sply = UnivariateSpline(df['t'], df['y'],s=.005)
            # dudx=diff(df['u'])/diff(df['x'])
            # dydt=diff(sply(ts))/diff(ts)
            # plt.plot(t, y,'ro')
            # plt.plot(ts, sply(ts),'b')
            # # splu.set_smoothing_factor(0.5)
            # # plt.plot(ts, splu(ts),'g')
            # plt.show()
            coef = np.polyfit(df['t'], df['y'],1)
            poly1d_fn = np.poly1d(coef)
            lamb=poly1d_fn(df['t'])
            df.loc[range(len(lamb)),'lamb'] = lamb
            dudx=diff(df['u'])/diff(df['x'])
            dydt=diff(df['lamb'])/diff(df['t'])
            df.loc[range(len(dudx)),'dudx'] = dudx
            df.loc[range(len(dydt)),'dydt'] = dydt
            if df.dydt.mean()<0:
                popt=[np.nan,np.nan]
            else:
            # df['dudxm'] = df['dudx'].rolling(window=3).mean()
            # df['dydtm'] = df['dydt'].rolling(window=3).mean()
            # df.dudx.rolling(5).mean()
            # rolling_mean = df['dudx'].rolling(window=3).mean()
                df=df.rolling(4).mean()
                df=df.dropna()
                # df.reset_index(drop=True, inplace=True)
                # spline = df.interpolate(method='spline', order=3, s=10)
                # df=df.loc[df['x']<5e-6]
                # df=df.loc[df['x']>2.5e-6]
                # df=df.loc[df['t']>0.008]
                # df=df.loc[df['t']<0.015]
                # print(df)
                # print(spline)
                # plt.plot(t[3:],dydt)
                # plt.show()
                # u=uniform_filter1d(ux, size=5)   
                # dudx=diff(ux)/diff(x)
                # dudx=uniform_filter1d(dudx, size=10)
                # y=uniform_filter1d(y, size=7)
                # dydt=diff(y)/diff(t)
                # dydt=uniform_filter1d(dydt, size=10)
                mu=0.00826 #[kg/m-s]
                A=136*10**-12
                majorRC=majorRC*10**-6
                # print(majorRC)
                # print(df['t'].values[-1])
                T=(3*A*mu*df['dudx'])/majorRC
                # plt.plot(df['t'], df['x'])
                # plt.show()
                # plt.plot(df['t'], df['u'])
                # plt.show()
                # plt.plot(df['t'], df['lamb'])
                # plt.show()
                # plt.plot(df['t'], df['dudx'])
                # plt.show()
                # plt.plot(df['t'], df['dydt'])
                # plt.show()
                # plt.plot(df['t'], T)
                # plt.show()
                # T=T[3:-5]
                # print(df['y'])
                x_data=[df['y'], df['dydt']]
                param_bounds=([1e-8,1e-9],[np.inf,2e-5])
                # if np.isinf(T).any() == False and np.isinf(y).any()== False and np.isinf(dydt).any()== False:
                try:
                    popt, pcov = curve_fit(KV, x_data, T, p0=p0,bounds=param_bounds)
                    # print(y)
                    # plt.plot(x_data[0], T)
                    # plt.show()
                    # plt.plot(x_data[0], KV(x_data, *popt))
                    # plt.show()
                    # print(popt*10**6)
                except RuntimeError:
                    print("Error - curve_fit failed")
        return popt
    df['params'] = df.apply(lambda row: fit(row['x'], row['y'],row['ux'],row['t'],row['majorRC']), axis=1)
    return df