import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

individuels = np.array([0.95704, 0.95504, 0.95504, 0.95504, 0.94905])
moyennes = np.array([0.95704, 0.95704, 0.95704, 0.95704, 0.96004])

"""
Accuracy des modèles sur les sets random: 

Group 0 - acc = 0.956250
Group 1 - acc = 0.957812
Group 2 - acc = 0.965625
Group 3 - acc = 0.964063
Group 4 - acc = 0.971875
Group 5 - acc = 0.962500
Group 6 - acc = 0.965625
Group 7 - acc = 0.962500
Group 8 - acc = 0.950000
Group 9 - acc = 0.962500
Accuracy moyenne: 0.96187
"""

#individuels = np.array([0.961899, 0.963773, 0.958776, 0.961274, 0.959400, 0.962523, 0.960650, 0.963148, 0.959400, 0.962523])
#moyennes=np.array([0.961899, 0.961274, 0.961899, 0.961899, 0.962523, 0.963773, 0.961274, 0.961899, 0.962523, 0.961274])

individuels=np.array([0.965022, 0.958151, 0.956902, 0.961899, 0.960650, 0.961899, 0.960650, 0.960025 ,0.958151, 0.960025])

moyennes=np.array([0.965022, 0.965646, 0.960650, 0.963773, 0.965022, 0.965022, 0.964397, 0.964397, 0.963773, 0.964397])

figure(figsize=(8,4), dpi=150)
individuels*=100
moyennes*=100
nb_groups = len(individuels)
x= [str(i+1) for i in range(nb_groups)]
plt.scatter(x, moyennes, marker="o")
plt.scatter(x, individuels, marker="x")
plt.ylabel('Précision (%)')
plt.xlabel('Nb. de modèles et # de modèle')
plt.legend(['Moyenne', 'Individuel'], loc='lower left')
plt.yticks(np.arange(95.6, 96.61, 0.1))
#plt.grid(which="both")
plt.show()