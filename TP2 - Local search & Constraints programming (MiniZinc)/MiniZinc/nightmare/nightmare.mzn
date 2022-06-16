include "globals.mzn"; 
   
int: n; % taille de la grille
% Nombre de tours observables
array[1..n] of int: left_vision;
array[1..n] of int: right_vision;
array[1..n] of int: up_vision;
array[1..n] of int: down_vision;

% Grille
array[1..n,1..n] of var 1..n: city;

% On ne peux pas avoir 2 tours de même taille pour chaque ligne et colonne
constraint forall( [count(city[1..n,j], k) == 1 | j in 1..n, k in 1..n] ); % colonnes
constraint forall( [count(city[i,1..n], k) == 1 | i in 1..n, k in 1..n] ); % lignes

/* 
Toutes les fonctions visible_XX indiquent si une tour est visible ou non selon le point d'observation. 
Ex. visible_up :
  À partir du point d'observation de up_vision, on considère une tour donnée (input) sur une colonne et on vérifie
  si toutes les tours situées avant elle sont de hauteur inférieure.
  
Params : columnIndex -- indice de la colonne (ou de la ligne si lineIndex) où on observe
         tourIndex -- indice de la tour (sur la ligne ou la colonne où on observe) 
Return : un bool si la tour est observable
*/ 


% Pour left_vision
function var bool: visible_left(int: lineIndex, int: tourIndex) = 
  if count( [city[lineIndex,tourIndex] > city[lineIndex, j] | j in 1..tourIndex-1] ) == tourIndex-1 
  then true else false endif;

constraint forall( [count( [visible_left(lineIndex, tourIndex) | tourIndex in 2..n] ) + 1 == left_vision[lineIndex] 
  | lineIndex in 1..n] ); 

% Pour up_vision
function var bool: visible_up(int: columnIndex, int: tourIndex) = 
  if count( [city[tourIndex,columnIndex] > city[i, columnIndex] | i in 1..tourIndex-1] ) == tourIndex-1 
  then true else false endif;

constraint forall( [count( [visible_up(columnIndex, tourIndex) | tourIndex in 2..n] ) + 1 == up_vision[columnIndex] 
  | columnIndex in 1..n] ); 

% Pour down_vision
function var bool: visible_down(int: columnIndex, int: tourIndex) = 
  if count( [city[tourIndex,columnIndex] > city[i, columnIndex] | i in tourIndex+1..n] ) == n-tourIndex 
  then true else false endif;

constraint forall( [count( [visible_down(columnIndex, tourIndex) | tourIndex in 1..n-1] ) + 1 == down_vision[columnIndex] 
  | columnIndex in 1..n] ); 

% Pour right_vision
function var bool: visible_right(int: lineIndex, int: tourIndex) = 
  if count( [city[lineIndex,tourIndex] > city[lineIndex, j] | j in tourIndex+1..n] ) == n-tourIndex 
  then true else false endif;

constraint forall( [count( [visible_right(lineIndex, tourIndex) | tourIndex in 1..n-1] ) + 1 == right_vision[lineIndex] 
  | lineIndex in 1..n] ); 


solve satisfy;

output 
  ["City: \n"] ++
  [show(city[i,j]) ++ if j == n then "\n" else " " endif | i,j in 1..n]