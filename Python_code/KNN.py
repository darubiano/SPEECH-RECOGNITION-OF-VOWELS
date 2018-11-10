
# -*- coding: utf-8 -*-
"""
Andrés Santiago Arias Páez
David Andrés Rubiano Venegas

"""
import os

# instalar librerias
os.system('pip install -r requirements.txt')
    
if os.path.exists('Reports\classification_report KNN.txt'):
    os.remove("Reports\classification_report KNN.txt")   

import pandas as pd
import collections
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import svm
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import neighbors
import numpy as np
import time

inicio_de_tiempo = time.time()

def agregartitulos(datos):
    # AGREGAR TITULOS A LOS DATOS
    s=datos.shape
    col=[]
    for x in range(0, s[1]):
        if x==0:
            col.append("name")
        elif x ==s[1]-1:
            col.append("tag")
        else:
            col.append("coef"+str(x))
    #se asigna el vector con los nombres de las columnas creado previamente y se las asignamos a la tabla
    datos.columns = col
    #Remplazar tags por valores numericos 'a':'1', 'e':'2', 'i':'3', 'o':'4', 'u':'5'
    vals_to_replace = {'a':'0', 'e':'1', 'i':'2', 'o':'3', 'u':'4'}
    datos['tag'] = datos['tag'].map(vals_to_replace)
    
    desired_columns=['name','tag']
    #obtener todas las columnas
    all_columns_list=datos.columns.tolist()
    desCol = [x for x in all_columns_list if x not in desired_columns]
    #se obtienen solo los coefficientes
    dCoeffs=datos[desCol]
    #se obtienen las etiquetas
    dTags=datos[col[-1]]
    
    return dCoeffs,dTags
   
def KNN(test):
    #,'MFCCs1','MFCCs2','MFCCs3','LPCs','LPCs1','LPCs2','LPCs3','TOTAL','TOTAL1','TOTAL2','TOTAL3'
    nombre_dataset=['MFCCs','MFCCs1','MFCCs2','MFCCs3','LPCs','LPCs1','LPCs2','LPCs3','TOTAL','TOTAL1','TOTAL2','TOTAL3']
    for i in range(len(nombre_dataset)):
        dataset=pd.read_csv("coefficients//"+nombre_dataset[i]+'.txt',sep=",",header=None)
        nombre=nombre_dataset[i]
        print(nombre)
        dataset = shuffle(dataset, random_state=0)
        X=dataset.iloc[:,1:-1]
        Y=dataset.iloc[:,-1]
        
        standardScaler = preprocessing.StandardScaler()
        X_scaler = standardScaler.fit_transform(X)
        
        # Remplazar los tags por numeros
        vals_to_replace = {'a':'0', 'e':'1', 'i':'2', 'o':'3', 'u':'4'}
        Y=Y.map(vals_to_replace)
        
        #Seleccionar los datos
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaler,Y,test_size=test, random_state=0)
        #, 2, 3
        #, 15, 30
        #,'kd_tree'
        parameters= [
            {
                'p': [1,2,3,5,6,8,10],
                'n_neighbors': [1,2,3,4,5,15,30,50,100,150],
                'algorithm':  ['ball_tree','kd_tree','brute'],
                'weights': ['uniform','distance']
            }    
        ]
        clf =GridSearchCV(neighbors.KNeighborsClassifier(), param_grid=parameters, cv=5)
        clf.fit(X_train,Y_train)
        
        MejorModelo=clf.best_params_
        
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        params = clf.cv_results_['params']
        
        f= open('Reports/means_report_KNN.txt','a+')
        f.write(nombre+' '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train \n')
        
        for m, s, p in zip(means, stds, params):
            print("%0.3f (+/-%0.3f) para %r"%(m, 2*s, p))
            f.write("%0.3f (+/-%0.3f) para %r"%(m, 2*s, p)+'\n')    
        f.close() 
        
        y_pred = clf.predict(X_test)
        target_names=["a","e","i","o","u"]
        table=classification_report(Y_test,y_pred, target_names=target_names)
        f= open('Reports/classification_report KNN.txt','a+')
        f.write(nombre+' '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train \n')
        f.write(str(MejorModelo)+'\n')
        f.write(table+'\n\n')
        f.close()
        
        mat=confusion_matrix(Y_test, y_pred)
        plt.title('Confusion matrices KNN '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre)
        fig=sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
                   xticklabels=target_names, yticklabels= target_names )
        plt.xlabel('Clases de Test')
        plt.ylabel('Clases Predichas')
        plt.show()
        fig.get_figure().savefig('Confusion matrices KNN/Confusion matrices KNN '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        # guardar Modelo
        joblib.dump(clf,'Models KNN/Modelo_entrenado KNN'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.pkl')

if __name__=='__main__':
    # Porcentajes de test 30% 25% 20% ,0.25,0.20
    porcentaje_test=[0.30,0.25,0.20]
    for j in range(len(porcentaje_test)):
        KNN(porcentaje_test[j])
        
    tiempo_final = time.time() 
    tiempo_transcurrido = tiempo_final - inicio_de_tiempo
    
    print("\nTomo %d segundos." % (tiempo_transcurrido))
    print("\nTomo %f minutos." % (tiempo_transcurrido/60))
    print("\nTomo %f horas." % (tiempo_transcurrido/3600))
    
    
    

        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
