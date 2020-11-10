# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:38:01 2020

@author: user
"""


#===================================================================
#            TP2 ETUDE DE L'ALGORITHME DU PERCEPTRON

#                    ZOUAGHI SABRINE
#                          M1 BIG DATA      
#===================================================================



def perceptron(x_input,w,c,esp,j):
    
    
    # UN COMPTEUR POUR VERIFIER QUE LES POIDS NE CHANGE PAS DE VALEUR A PARTIR D'UN CERTAIN NOMBRE D'ITERATION
    temoin=0 
    
    #UNE LISTE VIDE POUR ENREGISTRER LES VALEUR DES POIDS A CHAQUE BOUCLE ELLA A LA MEME TAILLE QUE LA LISTE DES POIDS W
    t=len(w)*[0] 
     
                   
    #ON A MIS UNE CONDITION D'ARRET Dés QUE LE TEMOIN = AU NOMBRE DES ENTRES 'X' DU PERCEPTRON 
    #c'est a dire il n’y a aucune modification des poids pour tous les entrees
    while(temoin != len(x_input)):
        somme=0
        o=0
        k=j % len(x_input) # j un comteur initialiser a 0 et K prend des valeurs {1,2,3} d'une maniere cyclique 
        for i in range(len(w)):
            # calcul de la sortie o
            somme = somme + w[i] * x_input[k][i]
            if somme > 0:
                o=1
            else:
                o=0 
        # enregistrer les poids dans t[] pour la comparaison avec les poids apres mise a jour        
        t[0]= w[0]        
        t[1]= w[1] 
        t[2]= w[2]

        # MISE A JOUR DES POIDS    
        w[0]= w[0] + esp * (c[k]-o) * x_input[k][0]
        w[1]= w[1] + esp * (c[k]-o) * x_input[k][1]
        w[2]= w[2] + esp * (c[k]-o) * x_input[k][2]
        
        # Affichage de tous les etapes d'apprentissage
        print(w)  
    
        # Incrementation du temoin
        
        if ((t[0]== w[0])and( t[1]== w[1])and( t[2]== w[2])):
            temoin = temoin+1
        else:
            temoin =0
    
        # incrementtion du j pour avancer le K dans le cycle {1,2,3} qui represente les element de input 'x'
        j=j+1
        
    print ("les poids sont : ")
    print(w)
    
#===================================================================    

# APPLICATION POUR LE OU LOGIGUE AVEC AFFICHAGE

#===================================================================
    
x_input=[[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
c=[0,1,1,1]
w=[0,1,-1] # initialisation des poids
esp=0.1    # epsilon
j=0        # compteur j initialiser a 0   

perceptron(x_input,w,c,esp,j)  #appel du fonction

# PARTIE PRESENTATION GRAPHIQUE
import numpy as np
import matplotlib.pyplot as plt
# Set an array of colours, we could call it
# anything but here we call is colormap
# It sounds more awesome
colormap = np.array(['r', 'k'])

# les entrées possibles pour chaque noeud 

x=[0,0,1,1] 
y=[0,1,0,1]

# x presente les valeur sue l'axe des abscisses
# x presente les valeur sue l'axe des ordonnes
# colormap c'est pour donenr le couleur rouge/noir selon la sortie souhaitée c
plt.scatter(x,y, c=colormap[c], s=40)



#===================================================================

# APPLICATION POUR LE ET LOGIGUE AVEC AFFICHAGE

#===================================================================   
x_input=[[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
c=[0,0,0,1]
w=[0,1,-1]
esp=0.1
j=0
perceptron(x_input,w,c,esp,j)

import numpy as np
import matplotlib.pyplot as plt
# Set an array of colours, we could call it
# anything but here we call is colormap
# It sounds more awesome
colormap = np.array(['r', 'k'])
x=[0,0,1,1]
y=[0,1,0,1]
# Plot the data, A is x axis, B is y axis
# and the colormap is applied based on the Targets
plt.scatter(x,y, c=colormap[c], s=40)




#===================================================================

# APPLICATION SUR L'EXERCICE 4 DU TD6

#===================================================================  

# QUESTION 1 : la trace d’exécution de l’algorithme du perceptron

x_input= [[1,1,1,1,1,1,0],[0,1,1,0,0,0,0],[1,1,0,1,1,0,1],[1,1,1,1,0,0,1],[0,1,1,0,0,1,1],[1,0,1,1,0,1,1],[0,0,1,1,1,1,1],[1,1,1,0,0,0,0],[1,1,1,1,1,1,1],[1,1,1,1,0,1,1]]
c=[0,1,0,1,0,1,0,1,0,1]
w=[0,0,0,0,0,0,0]
esp=1
j=0
perceptron(x_input,w,c,esp,j)

# LA PARTIE D'APPRENTISSAGE PREND UN TEMPS FOU !!!!!

