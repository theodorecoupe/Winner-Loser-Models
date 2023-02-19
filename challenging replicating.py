# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:32:33 2021

@author: theo coupe
"""
"""This code is used in order to run the Odagaki challenging society for a rectangular cell 
instead of a square lattice.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
from random import randint
import time 

stime = time.time()
print(time.time())
#Define Parameters
alpha = 0.05
delta = 1 #Defines the amount of strength transferred following an interaction.
mu = 0.1 #Defines the forgetting rate
number_of_individuals = 10
initial_strength = 0 #Initial strength assigned to each individual
number_of_time_steps = 2000
length = 20
list_of_lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
list_of_densities = []
list_of_average_village_size = []
list_of_variances = []
t = 0
while t < number_of_time_steps :
        length_1 = length #Choosing coordinates of the first agent
        length_2 = length
        if (length_1*length_2) >= number_of_individuals and length_1 >= length_2:
            density = (number_of_individuals)/(length_1*length_2)
            density_rounded = round(density, 5)
            list_of_densities.append(density)
            #Set up initial lattices
            X = np.zeros((length_1, length_2)) #Lattice showing positions of individuals
            Y = np.zeros((length_1, length_2)) #Lattice showing strength of each individual
            A = np.zeros((length_1, length_2)) #Lattice showing number of wins of each individual
            B = np.zeros((length_1, length_2)) #Lattice showing number of defeats of each individual
            C = np.zeros((number_of_individuals, number_of_individuals)) #Lattice used to remember who has interacted with each other.
            
            #Set up occupation of lattices
            b = 0
            while b < number_of_individuals:
                p = randint(0, length_1 - 1)
                q = randint(0, length_2 - 1)
                if X[p][q] == 0:
                    X[p][q] = b + 1
                    Y[p][q] = initial_strength
                    A[p][q] = 1
                    B[p][q] = 1
                    b += 1
            
            #Define interactions
            t = 0
            while t < number_of_time_steps:
                t += 1
                for x in range(0,length_1):
                            for y in range(0,length_2):
                                if Y[x][y] != 0:
                                    Y[x][y] += -mu*np.tanh(Y[x][y]) #Introducing a forgetting rate after each time step.
                for i in range(1, number_of_individuals + 1):
                    for x in range(0, length_1):
                        for y in range(0, length_2):
                            if X[x][y] == i:
                                x_agent1 = x
                                y_agent1 = y
                    list_of_nearest_neighbours = []
                    x_agent2_right = x_agent1 + 1 #Checking if any nearest neighbour sites are occupied.
                    y_agent2_right = y_agent1
                    if x_agent2_right == length_1:
                        x_agent2_right = 0
                    if X[x_agent2_right][y_agent2_right] > 0:
                        list_of_nearest_neighbours.append(Y[x_agent2_right][y_agent2_right])
                    x_agent2_left = x_agent1 - 1
                    y_agent2_left = y_agent1
                    if x_agent2_left == -1:
                        x_agent2_left = length_1 - 1
                    if X[x_agent2_left][y_agent2_left] > 0:
                        list_of_nearest_neighbours.append(Y[x_agent2_left][y_agent2_left])
                    x_agent2_up = x_agent1
                    y_agent2_up = y_agent1 + 1
                    if y_agent2_up == length_2:
                        y_agent2_up = 0
                    if X[x_agent2_up][y_agent2_up] > 0:
                        list_of_nearest_neighbours.append(Y[x_agent2_up][y_agent2_up])
                    x_agent2_down = x_agent1
                    y_agent2_down = y_agent1 - 1
                    if y_agent2_down == -1:
                        y_agent2_down = length_2 - 1
                    if X[x_agent2_down][y_agent2_down] > 0:
                        list_of_nearest_neighbours.append(Y[x_agent2_down][y_agent2_down])
                    list_of_nearest_neighbours.sort(reverse = True)
                    if len(list_of_nearest_neighbours) != 0:
                        if list_of_nearest_neighbours[0] == Y[x_agent2_right][y_agent2_right] and X[x_agent2_right][y_agent2_right] > 0:
                            x_agent2 = x_agent2_right
                            y_agent2 = y_agent2_right
                        elif list_of_nearest_neighbours[0] == Y[x_agent2_left][y_agent2_left] and X[x_agent2_left][y_agent2_left] > 0:
                            x_agent2 = x_agent2_left
                            y_agent2 = y_agent2_left
                        elif list_of_nearest_neighbours[0] == Y[x_agent2_up][y_agent2_up] and X[x_agent2_up][y_agent2_up] > 0:
                            x_agent2 = x_agent2_up
                            y_agent2 = y_agent2_up
                        elif list_of_nearest_neighbours[0] == Y[x_agent2_down][y_agent2_down] and X[x_agent2_down][y_agent2_down] > 0:
                            x_agent2 = x_agent2_down #Choosing strongest nearest neighbour. 
                            y_agent2 = y_agent2_down 
                    if len(list_of_nearest_neighbours) > 1 and (C[(int(X[x_agent1][y_agent1])) - 1][(int(X[x_agent2][y_agent2])) - 1] == 1 or C[(int(X[x_agent2][y_agent2])) - 1][(int(X[x_agent1][y_agent1])) - 1] == 1):
                        if list_of_nearest_neighbours[1] == Y[x_agent2_right][y_agent2_right] and X[x_agent2_right][y_agent2_right] > 0:
                            x_agent2 = x_agent2_right
                            y_agent2 = y_agent2_right
                        elif list_of_nearest_neighbours[1] == Y[x_agent2_left][y_agent2_left] and X[x_agent2_left][y_agent2_left] > 0:
                            x_agent2 = x_agent2_left
                            y_agent2 = y_agent2_left
                        elif list_of_nearest_neighbours[1] == Y[x_agent2_up][y_agent2_up] and X[x_agent2_up][y_agent2_up] > 0:
                            x_agent2 = x_agent2_up
                            y_agent2 = y_agent2_up
                        elif list_of_nearest_neighbours[1] == Y[x_agent2_down][y_agent2_down] and X[x_agent2_down][y_agent2_down] > 0:
                            x_agent2 = x_agent2_down #Choosing strongest nearest neighbour. 
                            y_agent2 = y_agent2_down  
                    if len(list_of_nearest_neighbours) > 2 and (C[(int(X[x_agent1][y_agent1])) - 1][(int(X[x_agent2][y_agent2])) - 1] == 1 or C[(int(X[x_agent2][y_agent2])) - 1][(int(X[x_agent1][y_agent1])) - 1] == 1):
                        if list_of_nearest_neighbours[2] == Y[x_agent2_right][y_agent2_right] and X[x_agent2_right][y_agent2_right] > 0:
                            x_agent2 = x_agent2_right
                            y_agent2 = y_agent2_right
                        elif list_of_nearest_neighbours[2] == Y[x_agent2_left][y_agent2_left] and X[x_agent2_left][y_agent2_left] > 0:
                            x_agent2 = x_agent2_left
                            y_agent2 = y_agent2_left
                        elif list_of_nearest_neighbours[2] == Y[x_agent2_up][y_agent2_up] and X[x_agent2_up][y_agent2_up] > 0:
                            x_agent2 = x_agent2_up
                            y_agent2 = y_agent2_up
                        elif list_of_nearest_neighbours[2] == Y[x_agent2_down][y_agent2_down] and X[x_agent2_down][y_agent2_down] > 0:
                            x_agent2 = x_agent2_down #Choosing strongest nearest neighbour. 
                            y_agent2 = y_agent2_down
                    if len(list_of_nearest_neighbours) > 3 and (C[(int(X[x_agent1][y_agent1])) - 1][(int(X[x_agent2][y_agent2])) - 1] == 1 or C[(int(X[x_agent2][y_agent2])) - 1][(int(X[x_agent1][y_agent1])) - 1] == 1):
                        if list_of_nearest_neighbours[3] == Y[x_agent2_right][y_agent2_right] and X[x_agent2_right][y_agent2_right] > 0:
                            x_agent2 = x_agent2_right
                            y_agent2 = y_agent2_right
                        elif list_of_nearest_neighbours[3] == Y[x_agent2_left][y_agent2_left] and X[x_agent2_left][y_agent2_left] > 0:
                            x_agent2 = x_agent2_left
                            y_agent2 = y_agent2_left
                        elif list_of_nearest_neighbours[3] == Y[x_agent2_up][y_agent2_up] and X[x_agent2_up][y_agent2_up] > 0:
                            x_agent2 = x_agent2_up
                            y_agent2 = y_agent2_up
                        elif list_of_nearest_neighbours[3] == Y[x_agent2_down][y_agent2_down] and X[x_agent2_down][y_agent2_down] > 0:
                            x_agent2 = x_agent2_down #Choosing strongest nearest neighbour. 
                            y_agent2 = y_agent2_down
                    if len(list_of_nearest_neighbours) == 0 or C[(int(X[x_agent1][y_agent1])) - 1][(int(X[x_agent2][y_agent2])) - 1] == 1 or C[(int(X[x_agent2][y_agent2])) - 1][(int(X[x_agent1][y_agent1])) - 1] == 1: #If all sites are empty choose random nearest neighbour site.
                        r = randint(0,3)
                        if r == 0:
                            x_agent2 = x_agent2_right
                            y_agent2 = y_agent2_right
                        elif r == 1:
                            x_agent2 = x_agent2_left
                            y_agent2 = y_agent2_left
                        elif r == 2:
                            x_agent2 = x_agent2_up
                            y_agent2 = y_agent2_up
                        else:
                            x_agent2 = x_agent2_down
                            y_agent2 = y_agent2_down
                    if X[x_agent2][y_agent2] > 0: #Defining a fight between two nearest neighbours.
                        a = Y[x_agent1][y_agent1]
                        b = Y[x_agent2][y_agent2]
                        c = A[x_agent1][y_agent1]
                        d = A[x_agent2][y_agent2]
                        e = B[x_agent1][y_agent1]
                        f = B[x_agent2][y_agent2]
                        k = X[x_agent1][y_agent1]
                        l = X[x_agent2][y_agent2]
                        probability_function = 1/(1 + np.exp(alpha*(b-a)))
                        random_number = np.random.uniform(0,1.0)
                        if probability_function >= random_number: #Updating strength values when agent 1 wins
                            a += delta
                            b -= delta
                            c += 1
                            f += 1
                            Y[x_agent1][y_agent1] = b #Agent 1 takes agent 2's site on the lattice.
                            Y[x_agent2][y_agent2] = a
                            A[x_agent1][y_agent1] = d
                            A[x_agent2][y_agent2] = c
                            B[x_agent1][y_agent1] = f
                            B[x_agent2][y_agent2] = e
                            X[x_agent1][y_agent1] = l
                            X[x_agent2][y_agent2] = k
                        else:
                            Y[x_agent1][y_agent1] -= delta #Updating strength values when agent 2 wins
                            Y[x_agent2][y_agent2] += delta
                            A[x_agent2][y_agent2] += 1
                            B[x_agent1][y_agent1] += 1
                        for i in range(number_of_individuals):
                            C[(int(X[x_agent1][y_agent1])) - 1][i] = 0
                            C[(int(X[x_agent2][y_agent2])) - 1][i] = 0
                        C[(int(X[x_agent1][y_agent1])) - 1][(int(X[x_agent2][y_agent2])) - 1] = 1
                        C[(int(X[x_agent2][y_agent2])) - 1][(int(X[x_agent1][y_agent1])) - 1] = 1
                    if X[x_agent2][y_agent2] == 0: #If nearest neighbour sites are empty.
                        z = Y[x_agent1][y_agent1]
                        g = A[x_agent1][y_agent1]
                        h = B[x_agent1][y_agent1]
                        m = X[x_agent1][y_agent1]
                        X[x_agent1][y_agent1] = 0 #agent 1 moves into the empty space if the chosen
                        X[x_agent2][y_agent2] = m #nearest neighbour site is empty
                        Y[x_agent2][y_agent2] = z
                        Y[x_agent1][y_agent1] = 0
                        A[x_agent2][y_agent2] = g
                        B[x_agent2][y_agent2] = h
                        A[x_agent1][y_agent1] = 0
                        B[x_agent1][y_agent1] = 0
                        for i in range(number_of_individuals):
                            C[int(m) - 1][i] = 0
                        
                        if t == 200:
                             list_of_strength_values = []
                             list_of_dominance_indices = []
                             D = np.zeros((length_1, length_2))
                             for x in range(0,length_1):
                                 for y in range(0,length_2): #adding the strength values to a list for plotting.
                                    k = X[x][y]
                                    v = A[x][y]
                                    w = B[x][y]
                                    if k > 0:
                                        D[x][y] = (v)/(v + w)
                                        list_of_dominance_indices.append((A[x][y])/(A[x][y] + B[x][y]))
                             bounds = [0, 0.0000000001, 0.33, 0.66, 1]
                             cmap = colors.ListedColormap(['white', 'blue','green','red'])
                             norm = colors.BoundaryNorm(bounds, cmap.N)
                             plt.imshow(D, cmap = cmap, norm = norm)
                             plt.colorbar(cmap = cmap, boundaries = bounds, spacing = 'proportional')
                             plt.title("60 individuals, "  + "200 time steps, Density = " + str(density_rounded))
                             plt.show()
                        
                        if t == 600:
                             list_of_strength_values = []
                             list_of_dominance_indices = []
                             D = np.zeros((length_1, length_2))
                             for x in range(0,length_1):
                                 for y in range(0,length_2): #adding the strength values to a list for plotting.
                                    k = X[x][y]
                                    v = A[x][y]
                                    w = B[x][y]
                                    if k > 0:
                                        D[x][y] = (v)/(v + w)
                                        list_of_dominance_indices.append((A[x][y])/(A[x][y] + B[x][y]))
                             bounds = [0, 0.0000000001, 0.33, 0.66, 1]
                             cmap = colors.ListedColormap(['white', 'blue','green','red'])
                             norm = colors.BoundaryNorm(bounds, cmap.N)
                             plt.imshow(D, cmap = cmap, norm = norm)
                             plt.colorbar(cmap = cmap, boundaries = bounds, spacing = 'proportional')
                             plt.title("60 individuals, "  + "1,500 time steps, Density = " + str(density_rounded))
                             plt.show()    
                        
                        if t == 1200:
                             list_of_strength_values = []
                             list_of_dominance_indices = []
                             D = np.zeros((length_1, length_2))
                             for x in range(0,length_1):
                                 for y in range(0,length_2): #adding the strength values to a list for plotting.
                                    k = X[x][y]
                                    v = A[x][y]
                                    w = B[x][y]
                                    if k > 0:
                                        D[x][y] = (v)/(v + w)
                                        list_of_dominance_indices.append((A[x][y])/(A[x][y] + B[x][y]))
                             bounds = [0, 0.0000000001, 0.33, 0.66, 1]
                             cmap = colors.ListedColormap(['white', 'blue','green','red'])
                             norm = colors.BoundaryNorm(bounds, cmap.N)
                             plt.imshow(D, cmap = cmap, norm = norm)
                             plt.colorbar(cmap = cmap, boundaries = bounds, spacing = 'proportional')
                             plt.title("60 individuals, "  + "5000 time steps, Density = " + str(density_rounded))
                             plt.show()
                       
                        if t == 1600:
                             list_of_strength_values = []
                             list_of_dominance_indices = []
                             D = np.zeros((length_1, length_2))
                             for x in range(0,length_1):
                                 for y in range(0,length_2): #adding the strength values to a list for plotting.
                                    k = X[x][y]
                                    v = A[x][y]
                                    w = B[x][y]
                                    if k > 0:
                                        D[x][y] = (v)/(v + w)
                                        list_of_dominance_indices.append((A[x][y])/(A[x][y] + B[x][y]))
                             bounds = [0, 0.0000000001, 0.33, 0.66, 1]
                             cmap = colors.ListedColormap(['white', 'blue','green','red'])
                             norm = colors.BoundaryNorm(bounds, cmap.N)
                             plt.imshow(D, cmap = cmap, norm = norm)
                             plt.colorbar(cmap = cmap, boundaries = bounds, spacing = 'proportional')
                             plt.title("60 individuals, "  + "1000 time steps, Density = " + str(density_rounded))
                             plt.show()
                       
                        if t == 2000:
                             list_of_strength_values = []
                             list_of_dominance_indices = []
                             D = np.zeros((length_1, length_2))
                             for x in range(0,length_1):
                                 for y in range(0,length_2): #adding the strength values to a list for plotting.
                                    k = X[x][y]
                                    v = A[x][y]
                                    w = B[x][y]
                                    if k > 0:
                                        D[x][y] = (v)/(v + w)
                                        list_of_dominance_indices.append((A[x][y])/(A[x][y] + B[x][y]))
                             bounds = [0, 0.0000000001, 0.33, 0.66, 1]
                             cmap = colors.ListedColormap(['white', 'blue','green','red'])
                             norm = colors.BoundaryNorm(bounds, cmap.N)
                             plt.imshow(D, cmap = cmap, norm = norm)
                             plt.colorbar(cmap = cmap, boundaries = bounds, spacing = 'proportional')
                             plt.title("60 individuals, "  + "20000 time steps, Density = " + str(density_rounded))
                             plt.show()
                        
            list_of_strength_values = []
            list_of_dominance_indices = []
            D = np.zeros((length_1, length_2))
            for x in range(0,length_1):
                for y in range(0,length_2): #adding the strength values to a list for plotting.
                    k = X[x][y]
                    v = A[x][y]
                    w = B[x][y]
                    if k > 0:
                        D[x][y] = (v)/(v + w)
                        list_of_dominance_indices.append((A[x][y])/(A[x][y] + B[x][y]))
            bounds = [0, 0.0000000001, 0.33, 0.66, 1]
            cmap = colors.ListedColormap(['white', 'blue','green','red'])
            norm = colors.BoundaryNorm(bounds, cmap.N)
            plt.imshow(D, cmap = cmap, norm = norm)
            plt.colorbar(cmap = cmap, boundaries = bounds, spacing = 'proportional')
            plt.title("32 individuals, " +str(number_of_time_steps) + " time steps, Density = " + str(density_rounded))
            plt.show()
            
            Z = np.zeros((length_1, length_2))
            for x in range(0, length_1):
                for y in range(0, length_2):
                    if X[x][y] > 0:
                        Z[x][y] = 1
            
            def numIslands():
                row = len(Z)
                col = len(Z[0])
                count = 0
                
                for i in range(row):
                    for j in range(col):
                        if Z[i][j] == 1:
                            check_nearest_neighbours(row, col, i, j, count)
                            count += 1
                print("Average size of island = " + str(number_of_individuals/count) + " individuals")
            
            def check_nearest_neighbours(row, col, x, y, z):
                
                if Z[x][y] != 1:
                    return
                Z[x][y] = z + 2
                
                if x != 0:
                    check_nearest_neighbours(row, col, x - 1, y, z)
                
                if x != row - 1:
                    check_nearest_neighbours(row, col, x + 1, y, z)
                    
                if y != 0:
                    check_nearest_neighbours(row, col, x, y - 1, z)
                
                if y != col - 1:
                    check_nearest_neighbours(row, col, x, y + 1, z)
                
                if x == 0:
                    check_nearest_neighbours(row, col, row - 1, y, z)
                    
                if x == row - 1:
                    check_nearest_neighbours(row, col, 0, y, z)
                    
                if y == 0:
                    check_nearest_neighbours(row, col, x, col - 1, z)
                    
                if y == col - 1:
                    check_nearest_neighbours(row, col, x, 0, z)
  
                    
class Graph:

    def __init__(self, row, col, graph, Y, A, B):
        self.ROW = row
        self.COL = col
        self.graph = graph
        self.Y = Y
        self.A = A
        self.B = B
        self.ispop = []
        self.istren = []
        self.avnint = []
       
        
	# A utility function to do DFS for a 2D
	# boolean matrix. It only considers
	# the 8 neighbours as adjacent vertices
    def DFS(self, i, j):
        ween = len(self.graph) - 1 
        numag = 0
        numag +=1
        
        if i < 0 or i >= len(self.graph) or j < 0 or j >= len(self.graph[0]) or self.graph[i][j] != 1:
            return
        if i == ween and j != ween and j != 0: #bottom row 
            self.graph[i][j] = -1

    		# Recur for 8 neighbours
            #self.DFS(i - 1, j - 1)
            self.DFS(i - 1, j)
            #self.DFS(i - 1, j + 1)
            self.DFS(i, j - 1)
            self.DFS(i, j + 1)
           # self.DFS(i + 1, j - 1)
            i = -1
            self.DFS(i + 1, j)
            i = ween
            #self.DFS(i + 1, j + 1)
    		# mark it as visited
            
        if j == ween and i != ween and i != 0: #right col
            self.graph[i][j] = -1

    		# Recur for 8 neighbours
            #self.DFS(i - 1, j - 1)
            self.DFS(i - 1, j)
            #self.DFS(i - 1, j + 1)
            self.DFS(i, j - 1)
            
           # self.DFS(i + 1, j - 1)
            self.DFS(i + 1, j)
            j = -1
            self.DFS(i, j + 1)
            j = ween
            #self.DFS(i + 1, j + 1)
    		# mark it as visited
        if j == ween and i == ween: #bottom right
            self.graph[i][j] = -1

    		# Recur for 8 neighbours
            #self.DFS(i - 1, j - 1)
            self.DFS(i - 1, j)
            #self.DFS(i - 1, j + 1)
            self.DFS(i, j - 1)
            j = -1
            self.DFS(i, j + 1)
           # self.DFS(i + 1, j - 1)
            j = ween
            i = -1
            self.DFS(i + 1, j)
            j = ween
            i = ween
            #self.DFS(i + 1, j + 1)
    		# mark it as visited
            
        
        if j == 0 and i != ween and i != 0: #left col
            self.graph[i][j] = -1

    		# Recur for 8 neighbours
            #self.DFS(i - 1, j - 1)
            self.DFS(i - 1, j)
            #self.DFS(i - 1, j + 1)
            
            self.DFS(i, j+1)
           # self.DFS(i + 1, j - 1)
            self.DFS(i + 1, j)
            j = ween + 1 
            self.DFS(i, j - 1)
            j = 0 
            #self.DFS(i + 1, j + 1)
    		# mark it as visited
        if j == 0 and i == ween:  #bottom left
            self.graph[i][j] = -1

    		# Recur for 8 neighbours
            #self.DFS(i - 1, j - 1)
            self.DFS(i - 1, j)
            #self.DFS(i - 1, j + 1)
            
            self.DFS(i, j+1)
           # self.DFS(i + 1, j - 1)
            j = ween + 1 
            self.DFS(i, j - 1)
            j = 0
            i = -1
            self.DFS(i + 1, j)
            j = 0
            i = ween
            #self.DFS(i + 1, j + 1)
    		# mark it as visited
            
        if i == 0 and j != ween and j != 0: #top row
            self.graph[i][j] = -1

    		# Recur for 8 neighbours
            #self.DFS(i - 1, j - 1)
            
            #self.DFS(i - 1, j + 1)
            self.DFS(i, j-1)
            self.DFS(i, j+1)
           # self.DFS(i + 1, j - 1)
            self.DFS(i + 1, j)
            i = ween + 1
            self.DFS(i - 1, j)
            i = 0
            
            #self.DFS(i + 1, j + 1)
    		# mark it as visited
        if j == 0 and i == 0:  #top left
            self.graph[i][j] = -1

    		# Recur for 8 neighbours
            #self.DFS(i - 1, j - 1)
            
            self.DFS(i, j+1)
           # self.DFS(i + 1, j - 1)
            self.DFS(i + 1, j)
            i = ween + 1
            self.DFS(i - 1, j)
            i = 0
            j = ween + 1
            #self.DFS(i - 1, j + 1)
            self.DFS(i, j - 1)
            i = 0
            j = 0
            #self.DFS(i + 1, j + 1)
    		# mark it as visited    
        if j == ween and i == 0:  #top right
            self.graph[i][j] = -1
            
    		# Recur for 8 neighbours
            #self.DFS(i - 1, j - 1)
            
            #self.DFS(i - 1, j + 1)
            self.DFS(i, j-1)
            
           # self.DFS(i + 1, j - 1)
            self.DFS(i + 1, j)
            j = -1
            self.DFS(i, j + 1)
            i = ween + 1
            j = ween
            self.DFS(i - 1, j)
            i = 0
            
            #self.DFS(i + 1, j + 1)
    		# mark it as visited    
            
        else:   
            self.graph[i][j] = -1
    		# Recur for 8 neighbours
            #self.DFS(i - 1, j - 1)
            self.DFS(i - 1, j)
            #self.DFS(i - 1, j + 1)
            self.DFS(i, j - 1)
            self.DFS(i, j + 1)
           # self.DFS(i + 1, j - 1)
            self.DFS(i + 1, j)
            #self.DFS(i + 1, j + 1)
        return numag
    # The main function that returns
    # count of islands in a given boolean
    # 2D matrix
    def countIslands(self):
        # Initialize count as 0 and traverse
        # through the all cells of
        # given matrix
        count = 0
        prev = 0
        numint = 0
        pnumint = 0
        strencount = 0
        sprev = 0
        
        for i in range(self.ROW):
            for j in range(self.COL):
                # If a cell with value 1 is not visited yet,
                # then new island found
                countt = 0
                if self.graph[i][j] == 1:
                    # Visit all cells in this island
                    # and increment island count
                    self.DFS(i, j)
                    count += 1
                    countt = 0
                    for x in range(self.ROW):
                        for y in range(self.COL):
                            if self.graph[x][y] == -1:
                                countt +=1
                                strencount += (Y[x][y]/1000)
                                numint += ((A[x][y] + B[x][y])/10000)
                    #print("island:",count,"pop:", (countt - prev))
                    self.istren.append((strencount -sprev)/(countt-prev))
                    self.ispop.append(countt - prev)
                    self.avnint.append((numint -pnumint)/(countt - prev))
                    pnumint = numint
                    prev = countt
                    sprev = strencount
# =============================================================================
#                     self.graph[i][j] != 1
#                 if X[i][j] >= 1: 
#                     if nebs(row, col, i, j, length)[0] == 0:
#                         countt+=1
# =============================================================================
 
        return count

#class for agents

graph = Z


# =============================================================================
# 
# 
# row = len(graph)
# col = len(graph[0])
# 
# 
# 
# X = np.zeros((5,5))
# 
# row = 5
# col = 5
# length = 5
# X[4][4] = 1
# X[0][0] = 1
# X[0][2] = 1
# X[0][1] = 1
# X[4][2] = 0
# X[4][0] = 1
# X[0][4] = 1
# 
# print(X)
# 

# 

row = len(graph)
col = len(graph[0])




g = Graph(row, col, graph, Y,A,B)



#avgsize = number_of_individuals/(g.countIslands())


nis = g.countIslands()

islandstrength = np.array(g.istren)
islandpopulation = np.array(g.ispop)
averagenumberofinteractions = np.array(g.avnint)
largest_island = 0



for i in range(0,len(islandpopulation)):
    print("island#:",(i+1)," island pop:", islandpopulation[i], " average strength:", islandstrength[i],"*10^3" , "avg# int:", averagenumberofinteractions[i], "*10^4")
    if islandpopulation[i] > largest_island:
        largest_island = islandpopulation[i]

#print("avstren of is 1:", islandstrength[0], "av pop:", islandpopulation[0])
#print(A)

print("Number of clusters is:", nis)
print("avg size of island is", (number_of_individuals / nis))
print("largest island:", largest_island)
w =1

plt.hist(islandpopulation, bins=np.arange(min(islandpopulation), max(islandpopulation) + 2), edgecolor = 'white')
#plt.xticks(np.arange(0, max(islandpopulation) + 2, 5))
plt.grid(axis='y', color='#3475D0', lw = .4)
plt.grid(axis='x', color='#3475D0',  lw = .4)
plt.title("Island Population") 
plt.xlabel("Number of Agents")
plt.ylabel("Number of Islands")
plt.show()
print(time.time())
print(number_of_individuals, 'agents', ((time.time()-stime)/60.0), 'minutes' )


plt.hist( islandstrength, edgecolor = 'white')
plt.grid(axis='y', color='#3475D0', lw = .4)
plt.grid(axis='x', color='#3475D0',  lw = .4)
plt.title("Island Strength: Average agent strength") 
plt.ylabel("Number of Island")
plt.xlabel("Average strength of island * 10^3")
plt.show


#print("avg size of island is:", avgsize)
# =============================================================================
# 
# if g.countIslands() >= 1:
#      print("minimum island size is 1 agent")
# 
# 
# =============================================================================
#Graph(row, col, Z)

#print(X)  


# =============================================================================
# 
# 
# avgsize = number_of_individuals/(g.countIslands())
# 
# 
# print("Number of clusters is:", g.countIslands())
# print("avg size of island is:", avgsize)
# 
# if g.countIslands()[1] >= 1:
#     print("minimum island size is 1 agent")
# 
# Graph(row, col, Z)
#   
# 
# =============================================================================

# =============================================================================
#             numIslands()
#             list_2 = []
#             for i in range(2,202):
#                 k = 0
#                 list_of_strengths = []
#                 for x in range(0, length_1):
#                     for y in range(0, length_2):
#                         if Z[x][y] == i:
#                             t += 1
#                             k += 1
#                             list_of_strengths.append(Y[x][y])
#                 if k !=0:
#                     list_2.append(k)
#                     variance = 0
#                     for i in list_of_strengths:
#                         average = (sum(list_of_strengths)/len(list_of_strengths))
#                         variance += ((i-average)**2)/len(list_of_strengths)
#             def calculate_variance(my_list):
#                 result = np.var(my_list)
#                 list_of_variances.append(result)
#             calculate_variance(list_of_dominance_indices)
#             print(list_2)
#             plt.hist(list_2, bins = 50, color = "black", label = "density = " + str(density_rounded))
#             plt.xlim([0, number_of_individuals])
#             plt.ylim([0, 10])
#             plt.ylabel("Frequency")
#             plt.xlabel("Village Size")
#             plt.title("Distribution of Village Sizes for 100 individuals (5,000 time steps)")
#             plt.legend()
#             plt.show()
#             print("Density = " + str(density))
#             
#             plt.scatter(list_of_densities, list_of_variances, color = "orange")
#             plt.xlabel("Density")
#             plt.ylabel("Variance")
#             plt.title("Variance versus Density Challenging Society Rectangular Lattice")
#             plt.xlim([0, 1.0])
#             plt.ylim([0, 0.1])
#             plt.show()
#             
# =============================================================================
