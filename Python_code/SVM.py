# -*- coding: utf-8 -*-
"""
Andrés Santiago Arias Páez
David Andrés Rubiano Venegas

"""
import os

# instalar librerias
os.system('pip install -r requirements.txt')

#if os.path.exists('Reports\classification_report.txt'):
    #os.remove("Reports\classification_report.txt")
    
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


def SVM1(test):
    #TAMAÑO DE LOS DATOS
    #'LPCs.txt',
    '''
    Nota probar con pocos coeficientes recomendados MFCCs y MFCCs1,'MFCCs1','MFCCs2','MFCCs3','LPCs','LPCs1','LPCs2','LPCs3','TOTAL','TOTAL1','TOTAL2','TOTAL3'
    '''    
    nombre_dataset=['MFCCs','MFCCs1','MFCCs2','MFCCs3','LPCs','LPCs1','LPCs2','LPCs3','TOTAL','TOTAL1','TOTAL2','TOTAL3']

    for i in range(len(nombre_dataset)):
        dataset=pd.read_csv("coefficients/"+nombre_dataset[i]+'.txt',sep=",",header=None)
        nombre=nombre_dataset[i]
        print(nombre)
        """
        TRATAMIENTO DE LOS DATASETS
        """
        dCoeffs,dTags=agregartitulos(dataset)
        #print(datoslpc.tail())
        #print(datosmfcc.tail())
        """
        VALIDACION CRUZADA
        """ 
    
        X_train, X_test, Y_train, Y_test = train_test_split(dCoeffs,dTags,test_size=test, random_state=0)
        #1e-4, 1e-3, 1e-2, 0.1, 1, 10,100,1000,10000,100000
        C_range = [1e-4, 1e-3, 1e-2, 0.1, 1, 10,100,1000,10000]
        gamma_range = [1e-4, 1e-3, 1e-2, 0.1, 1, 10,100,1000,10000]
        #Parametros
        parameters= [
            {
                'kernel': ['rbf'],
                'gamma': gamma_range,
                'C': C_range
            } 
        ]
        
        # 5 Folios
        clf =GridSearchCV(svm.SVC(decision_function_shape='ovr'), param_grid=parameters, cv=5)
        clf.fit(X_train,Y_train)
        print(clf.best_params_)
        MejorModelo=clf.best_params_
        
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        params = clf.cv_results_['params']    
        
        f= open('Reports/means_report.txt','a+')
        f.write(nombre+' '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train \n')
        
        for m, s, p in zip(means, stds, params):
            print("%0.3f (+/-%0.3f) para %r"%(m, 2*s, p))
            f.write("%0.3f (+/-%0.3f) para %r"%(m, 2*s, p)+'\n')    
        f.close()     
            
        y_pred = clf.predict(X_test)
        
        target_names=["a","e","i","o","u"]
        # Imprimir precision recall f1-score support
        tabla=classification_report(Y_test,y_pred, target_names=target_names)
        tabla=str(tabla)
        #print(tabla)
        f= open('Reports/classification_report.txt','a+')
        f.write(nombre+' '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train \n')
        f.write(str(MejorModelo)+'\n')
        f.write(tabla+'\n\n')
        f.close()
        
        Y_test.groupby(Y_test).count(),collections.Counter(y_pred)
        
        mat=confusion_matrix(Y_test, y_pred)
        plt.title('Matriz de confusion rbf SVM '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre)
        fig=sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
                   xticklabels=target_names, yticklabels= target_names )
        plt.xlabel('Clases de Test')
        plt.ylabel('Clases Predichas')
        print(nombre)
        plt.show(block=False)
        plt.close()

        fig.get_figure().savefig('Confusion matrices SVM/Matriz de confusion rbf'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        scores = clf.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))
                                                             
        #print("The best parameters are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        fig1=plt.title('Validation accuracy')
        
        plt.show(block=False)
        plt.close()
        
        fig1.get_figure().savefig('Parameter matrices/Matriz de parametros rbf '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        
        for ind, i in enumerate(C_range):
            plt.plot(gamma_range, scores[ind], label='C: ' + str(i))
        plt.legend()
        plt.xlabel('Gamma')
        fig2=plt.ylabel('Mean score')
        
        fig2.get_figure().savefig('Cs matrices/Matriz de Cs rbf '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        plt.show(block=False)
        plt.close()
        # guardar Modelo
        joblib.dump(clf,'Models SVM/Modelo_entrenado SVM rbf'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.pkl')

def SVM2(test):
    #TAMAÑO DE LOS DATOS
    #'LPCs.txt',
    '''
    Nota probar con pocos coeficientes recomendados MFCCs y MFCCs1,'MFCCs1','MFCCs2','MFCCs3','LPCs','LPCs1','LPCs2','LPCs3','TOTAL','TOTAL1','TOTAL2','TOTAL3'
    '''    
    nombre_dataset=['MFCCs','LPCs','TOTAL','MFCCs1','MFCCs2','MFCCs3','LPCs1','LPCs2','LPCs3','TOTAL1','TOTAL2','TOTAL3']

    for i in range(len(nombre_dataset)):
        dataset=pd.read_csv("coefficients/"+nombre_dataset[i]+'.txt',sep=",",header=None)
        nombre=nombre_dataset[i]
        print(nombre)
        """
        TRATAMIENTO DE LOS DATASETS
        """
        dCoeffs,dTags=agregartitulos(dataset)
        #print(datoslpc.tail())
        #print(datosmfcc.tail())

        """

        VALIDACION CRUZADA
        """ 
    
        X_train, X_test, Y_train, Y_test = train_test_split(dCoeffs,dTags,test_size=test, random_state=0)
        #1e-4, 1e-3, 1e-2, 0.1, 1, 10,100,1000,10000,100000
        C_range = [1e-4, 1e-3, 1e-2, 0.1, 1, 10,100,1000,10000]
        gamma_range = [1]
        #Parametros
        parameters= [
            {
                'kernel': ['linear'],
                'gamma': gamma_range,
                'C': C_range

            }  
        ]
        
        # 5 Folios
        clf =GridSearchCV(svm.SVC(decision_function_shape='ovr'), param_grid=parameters, cv=5)
        clf.fit(X_train,Y_train)
        
        print(clf.best_params_)
        MejorModelo=clf.best_params_
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        params = clf.cv_results_['params']
        
        f= open('Reports/means_report.txt','a+')
        f.write(nombre+' '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train linear \n')
        
        for m, s, p in zip(means, stds, params):
            print("%0.3f (+/-%0.3f) para %r"%(m, 2*s, p))
            f.write("%0.3f (+/-%0.3f) para %r"%(m, 2*s, p)+'\n')    
        f.close()
        
        y_pred = clf.predict(X_test)
        
        target_names=["a","e","i","o","u"]
        # Imprimir precision recall f1-score support
        tabla=classification_report(Y_test,y_pred, target_names=target_names)
        tabla=str(tabla)
        #print(tabla)
        f= open('Reports/classification_report.txt','a+')
        f.write(nombre+' '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train \n')
        f.write(str(MejorModelo)+'\n')
        f.write(tabla+'\n\n')
        f.close()
        
        Y_test.groupby(Y_test).count(),collections.Counter(y_pred)

        
        mat=confusion_matrix(Y_test, y_pred)
        plt.title('Matriz de confusion SVM linear'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre)
        fig=sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
                   xticklabels=target_names, yticklabels= target_names )
        plt.xlabel('Clases de Test')
        plt.ylabel('Clases Predichas')
        print(nombre)
        plt.show(block=False)
        plt.close()

        fig.get_figure().savefig('Confusion matrices SVM/Matriz de confusion linear'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        scores = clf.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))
                                                             
        #print("The best parameters are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        fig1=plt.title('Validation accuracy')
        
        plt.show(block=False)
        plt.close()
        
        fig1.get_figure().savefig('Parameter matrices/Matriz de parametros linear'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        
        for ind, i in enumerate(C_range):
            plt.plot(gamma_range, scores[ind], label='C: ' + str(i))
        plt.legend()
        plt.xlabel('Gamma')
        fig2=plt.ylabel('Mean score')
        
        fig2.get_figure().savefig('Cs matrices/Matriz de Cs linear'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        plt.show(block=False)
        plt.close()
        # guardar Modelo
        joblib.dump(clf,'Models SVM/Modelo_entrenado SVM linear'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.pkl')



def SVM3(test):
    #TAMAÑO DE LOS DATOS
    #'LPCs.txt',
    '''
    Nota probar con pocos coeficientes recomendados MFCCs y MFCCs1,'MFCCs1','MFCCs2','MFCCs3','LPCs','LPCs1','LPCs2','LPCs3','TOTAL','TOTAL1','TOTAL2','TOTAL3'
    '''    
    nombre_dataset=['MFCCs','LPCs','TOTAL','MFCCs1','MFCCs2','MFCCs3','LPCs1','LPCs2','LPCs3','TOTAL1','TOTAL2','TOTAL3']

    for i in range(len(nombre_dataset)):
        dataset=pd.read_csv("coefficients/"+nombre_dataset[i]+'.txt',sep=",",header=None)
        nombre=nombre_dataset[i]
        print(nombre)
        """
        TRATAMIENTO DE LOS DATASETS
        """
        dCoeffs,dTags=agregartitulos(dataset)
        #print(datoslpc.tail())
        #print(datosmfcc.tail())

        """

        VALIDACION CRUZADA
        """ 
    
        X_train, X_test, Y_train, Y_test = train_test_split(dCoeffs,dTags,test_size=test, random_state=0)
        #1e-4, 1e-3, 1e-2, 0.1, 1, 10,100,1000,10000,100000
        C_range = [1e-4, 1e-3, 1e-2, 0.1, 1, 10,100,1000,10000]
        gamma_range = [1e-4, 1e-3, 1e-2, 0.1, 1, 10,100,1000,10000]
        #Parametros
        parameters= [
            {
                'kernel': ['poly'],
                'gamma': gamma_range,
                'C': C_range

            }  
        ]
        
        # 5 Folios
        clf =GridSearchCV(svm.SVC(decision_function_shape='ovr'), param_grid=parameters, cv=5)
        clf.fit(X_train,Y_train)
        
        print(clf.best_params_)
        MejorModelo=clf.best_params_
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        params = clf.cv_results_['params']
        
        f= open('Reports/means_report_poly.txt','a+')
        f.write(nombre+' '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train \n')
        
        for m, s, p in zip(means, stds, params):
            print("%0.3f (+/-%0.3f) para %r"%(m, 2*s, p))
            f.write("%0.3f (+/-%0.3f) para %r"%(m, 2*s, p)+'\n')    
        f.close()  
            
        y_pred = clf.predict(X_test)
        
        target_names=["a","e","i","o","u"]
        # Imprimir precision recall f1-score support
        tabla=classification_report(Y_test,y_pred, target_names=target_names)
        tabla=str(tabla)
        #print(tabla)
        f= open('Reports/classification_report.txt','a+')
        f.write(nombre+' '+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train \n')
        f.write(str(MejorModelo)+'\n')
        f.write(tabla+'\n\n')
        f.close()
        
        Y_test.groupby(Y_test).count(),collections.Counter(y_pred)

        
        mat=confusion_matrix(Y_test, y_pred)
        plt.title('Matriz de confusion SVM poly'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre)
        fig=sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
                   xticklabels=target_names, yticklabels= target_names )
        plt.xlabel('Clases de Test')
        plt.ylabel('Clases Predichas')
        print(nombre)
        plt.show(block=False)
        plt.close()

        fig.get_figure().savefig('Confusion matrices SVM/Matriz de confusion poly'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        scores = clf.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))
                                                             
        #print("The best parameters are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        fig1=plt.title('Validation accuracy')
        
        plt.show(block=False)
        plt.close()
        
        fig1.get_figure().savefig('Parameter matrices/Matriz de parametros poly'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        
        for ind, i in enumerate(C_range):
            plt.plot(gamma_range, scores[ind], label='C: ' + str(i))
        plt.legend()
        plt.xlabel('Gamma')
        fig2=plt.ylabel('Mean score')
        
        fig2.get_figure().savefig('Cs matrices/Matriz de Cs poly'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.png')
        plt.show(block=False)
        plt.close()
        # guardar Modelo
        joblib.dump(clf,'Models SVM/Modelo_entrenado SVM poly'+str(int(test*100))+'% Test '+str(int((1-test)*100))+'% Train '+nombre+'.pkl')
    
if __name__=='__main__':
    # Porcentajes de test 30% 25% 20% ,0.25,0.20
    porcentaje_test=[0.30,0.25,0.20]
    for j in range(len(porcentaje_test)):
        SVM1(porcentaje_test[j])
        SVM2(porcentaje_test[j])
        SVM3(porcentaje_test[j])

    tiempo_final = time.time() 
    tiempo_transcurrido = tiempo_final - inicio_de_tiempo
    
    print("\nTomo %d segundos." % (tiempo_transcurrido))
    print("\nTomo %f minutos." % (tiempo_transcurrido/60))
    print("\nTomo %f horas." % (tiempo_transcurrido/3600))
    
    
    

        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
