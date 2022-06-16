######################################
# Solution par recherche locale      #
# Didier Blach-Laflèche  #xxx	 #
# Marc-Antoine Huet  	 #xxx    #
# 03 novembre 2020	                 #
######################################

from generator_problem import GeneratorProblem

# Si la librarie "random" cause problem:
# Commentez ligne 12 et mettre restart_meta à False, à la ligne 126
from random import randint


class Solve:

    def __init__(self, n_generator, n_device, seed):

        self.n_generator = n_generator
        self.n_device = n_device
        self.seed = seed

        self.instance = GeneratorProblem.generate_random_instance(self.n_generator, self.n_device, self.seed)

    def solve_naive(self):

        print("Solve with a naive algorithm")
        print("All the generators are opened, and the devices are associated to the closest one")

        opened_generators = [1 for _ in range(self.n_generator)]

        assigned_generators = [None for _ in range(self.n_device)]

        for i in range(self.n_device):
            closest_generator = min(range(self.n_generator),
                                    key=lambda j: self.instance.get_distance(self.instance.device_coordinates[i][0],
                                                                      self.instance.device_coordinates[i][1],
                                                                      self.instance.generator_coordinates[j][0],
                                                                      self.instance.generator_coordinates[j][1])
                                    )

            assigned_generators[i] = closest_generator

        self.instance.solution_checker(assigned_generators, opened_generators)
        total_cost = self.instance.get_solution_cost(assigned_generators, opened_generators)
        self.instance.plot_solution(assigned_generators, opened_generators)

        print("[ASSIGNED-GENERATOR]", assigned_generators)
        print("[OPENED-GENERATOR]", opened_generators)
        print("[SOLUTION-COST]", total_cost)


## Méthodes utiles varias --------------------------------------

    # Arguments:
    # - input_array : La liste à copier
    # - output_array : Le contenant de la copy
    # Sortie:
    # - output_array: La liste copiée
    def copyArray(self, input_array, output_array):
        if len(input_array) != len(output_array):
            return -1
        else:
            for i in range(len(input_array)):
                output_array[i] = input_array[i]
            return output_array
    
    # Argument:
    # - nb_opened_generaator (int) :  le nombre de génératrice ouverte pour cette solution
    # Sortie:
    # - opened_generator (liste de int) : un état
    def random_solution(self, nb_opened_generaator):
        opened_generators = [0 for _ in range(self.n_generator)]
        for i in range(nb_opened_generaator):
            opened_generators[randint(0,self.n_generator-1)] = 1
        return opened_generators    

    # Arguments:
        # - instance (instance): l'instance générée par le seed
        # - assigned_generators (liste de int): Les génératrice la plus proche de chaque machine
        # - device_distances (liste de float): Distance de chaque machine avec sa génératrice
        # - opened_generators (liste de int): Les génératrices allumées
        # Sortie:
        # - total_cost (float): coût de cet état
    def get_solution_cost_LocalSearch(self, instance, assigned_generators, device_distances, opened_generators):
        '''
        :param assigned_generators: list where the element at index $i$ correspond to the generator associated to device $i$
        :param opened_generators: list where the element at index $i$ is a boolean stating if the generator $i$ is opened
        '''

        assert len(assigned_generators) == instance.n_device
        assert len(opened_generators) == instance.n_generator

        total_opening_cost = sum([opened_generators[i] * instance.opening_cost[i] for i in range(self.n_generator)])
        total_distance_cost = sum(device_distances)
        total_cost =  total_distance_cost + total_opening_cost
        
        return total_cost


## Méthodes principales de recherche locale -----------

    # Fonction de voisinage
    # Agruments:
    # -opened_generators: liste de int: Les génératrices overtes, courant
    # -previous_states: liste de liste: conteint les combinaisons passées de génératrices 
    # Sorties:
    # -neighbour_list: liste de liste: contient les combiaisons de génératrices voisines
    def neighbour_function(self,opened_generators,previous_states):
        neighbour_list = []
        #Calcule les voisins
        for i in range(0,len(opened_generators)):
            neighbour = [None]*len(opened_generators)
            neighbour = self.copyArray(opened_generators,neighbour)
            neighbour[i] =  int(not bool(neighbour[i])) #flip on or off
            #On ajouter l'état que s'il n'a pas encore été visité récement
            if neighbour not in previous_states:
                neighbour_list.append(neighbour)        
        return neighbour_list      
  
    # Fonction d'évaluation
    # Agruments:
    # -opened_generators: liste de int: Les génératrices overtes, courant
    # Sorties:    
    # -solution_cost: float: Le coût de de cet état
    # -assigned_generators: liste de int: La génératrice associée à chaque machine    
    def evaluation_function(self,opened_generators):  
        assigned_generators = []
        device_distances = []
        #Trouve la généatrice la plus proche de chaque machine, ainsi que la distance
        for devNumber, device in enumerate(self.instance.device_coordinates):
            calculated_distance = 10**23
            index = 0
            for genNumber, generator in enumerate(self.instance.generator_coordinates):
                
                if opened_generators[genNumber]:
                    distance = self.instance.get_distance(device[0], device[1], generator[0], generator[1])
                    if distance<calculated_distance:
                        calculated_distance = distance
                        index = genNumber
            assigned_generators.append(index)
            device_distances.append(calculated_distance)
        #Calcul le coût total selon les distance machienes-génératrices et les coûts de base    
        solution_cost = self.get_solution_cost_LocalSearch(self.instance,assigned_generators, device_distances, opened_generators)
        return solution_cost,assigned_generators
    
    # Fonction de sélection
    # Agruments:
    # -cost: float: coût de la meilleur solution à date
    # -neighbour_cost: liste de float: coûts de tout les états voisins
    # Sorties:
    # -index: int: index di meilleu voisin
    def selection_function(self,order,cost,neighbour_cost):
        valeur, index = min((valeur, index) for (index, valeur) in enumerate(neighbour_cost))
        for i in range (order):
            neighbour_cost[index] = 10**23
            valeur, index = min((valeur, index) for (index, valeur) in enumerate(neighbour_cost))
        if valeur< cost:
            return index
        
        return -1
    

## Méthode de solution pour la recherche locale -----------------------

    def solve(self):
        tabu_meta           = True
        restart_meta        = True
        moreNeighbour_meta  = True
        degradation_meta    = True
        
        deadCounter_restart = 0 
        deadCounter_neighbour = 0 
        deadCounter_degradation = 0 
                
        SOLUTION_OPENED_GENERATORS = [1 for _ in range(self.n_generator)]
        SOLUTION_ASSIGNED_GENERATORS = [None for s_ in range(self.n_device)]
        SOLUTION_COST = 10**23
        
        opened_generators = [1 for _ in range(self.n_generator)]
        assigned_generators = [None for _ in range(self.n_device)]
        cost = 10**23
        previous_states = []
        neighbour_assigned_generators = [None for _ in range(0,self.n_generator)]

        maxit = 300
        for i in range(maxit):
            if i%10 ==0:
                print("Intération : "+str(i)+" of "+str(maxit))
               # print("Time elapsed: "+str(int(tic-toc))+" of "+str(maxsec)+" seconds")
            
# Meta: Nouveau départ --------------------------
            if restart_meta:
                if deadCounter_restart == 30:
                    opened_generators = self.random_solution(2)
                    while opened_generators in previous_states:
                            opened_generators = self.random_solution(2)
                    deadCounter_restart = 0        

            
#Meta: Tabu -----------------------------------
            # On trouve les voisins non-visités
            neighbour_list = self.neighbour_function(opened_generators,previous_states)
            if not tabu_meta:
                previous_states = []
            
#Meta: Second voisins --------------------------
            if moreNeighbour_meta:
                if deadCounter_neighbour > 15:
                    neighbour_list = self.neighbour_function(opened_generators,previous_states)
                    new_list = []
                    for neigbhour in neighbour_list:
                        new_list = new_list + self.neighbour_function(neigbhour,previous_states)
                    neighbour_list = neighbour_list + new_list
                    
 

         
            # On évalue le coût de chaque voisin
            neighbour_cost = [1 for _ in range(0,len(neighbour_list))]
            neighbour_assigned_generators =  [1 for _ in range(0,len(neighbour_list))]
            for index,generators in enumerate(neighbour_list):
                solution_cost,solution_assigned_generators = self.evaluation_function(generators)       
                neighbour_cost[index] = solution_cost
                neighbour_assigned_generators[index] = solution_assigned_generators
               
                
            # On sélection le meilleur voisin 
#Meta ----------------------------------------
            if degradation_meta:
                if deadCounter_degradation < 7:
                    best_neighbhour_index = self.selection_function(0,cost, neighbour_cost)
                else:
                    best_neighbhour_index = self.selection_function(randint(2,5),cost, neighbour_cost)  
                deadCounter_degradation = 0                          
            else:
                 best_neighbhour_index = self.selection_function(0,cost, neighbour_cost)
            
            # Si le meilleur voisin est meilleur que la meilleur solution trouvée, on le garde
            if best_neighbhour_index != -1: 
                opened_generators = neighbour_list[best_neighbhour_index]
                assigned_generators = neighbour_assigned_generators[best_neighbhour_index]
                cost = neighbour_cost[best_neighbhour_index]
                previous_states.append(opened_generators)
                
                SOLUTION_OPENED_GENERATORS = self.copyArray(opened_generators,SOLUTION_OPENED_GENERATORS)
                SOLUTION_ASSIGNED_GENERATORS = self.copyArray(assigned_generators,SOLUTION_ASSIGNED_GENERATORS)
                SOLUTION_COST = cost
                
                deadCounter_restart = 0 
                deadCounter_neighbour = 0 
                deadCounter_degradation = 0 
        
            else:        
                deadCounter_restart +=1
                deadCounter_neighbour +=1
                deadCounter_degradation +=1 


       
### solution trouvée   
        self.instance.plot_solution(assigned_generators, opened_generators)
        
        print("[ASSIGNED-GENERATOR]", SOLUTION_ASSIGNED_GENERATORS)
        print("[OPENED-GENERATOR]", SOLUTION_OPENED_GENERATORS)
        print("[SOLUTION-COST]",'{:.2f}'.format(SOLUTION_COST) )   

