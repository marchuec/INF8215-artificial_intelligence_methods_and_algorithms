include "globals.mzn"; 

int: n;  % Taille de la grille
array[1..n, 1..n] of var 0..1: grid_init_config; % On utilise var pour que les "_" soient considérés comme des variables. Le reste sont des constantes.
 
array[1..n, 1..n] of var 0..1: grid; % Grille à compléter

% On fixe les variables de grid qui sont des constantes dans grid_init_config
constraint forall([grid[i,j] == grid_init_config[i,j] | i,j in 1..n]);

% Le nombre de mines doit être le même pour chaque ligne et chaque colonne
constraint all_equal([sum(grid[i, 1..n]) | i in 1..n] ++ [sum(grid[1..n, j]) | j in 1..n]);

% Pas de lignes identiques ou de colonnes identiques
constraint forall([ grid[1..n,i] != grid[1..n,j] | i,j in 1..n where j>i ]); % colonnes
constraint forall([ grid[i,1..n] != grid[j,1..n] | i,j in 1..n where j>i ]); % lignes

% Pas plus de 2 cases consécutives sans mines pour les lignes et les colonnes
constraint forall([ count(grid[i..i+2,j], 0) <= 2 | j in 1..n, i in 1..n-2]); % colonnes
constraint forall([ count(grid[i,j..j+2], 0) <= 2 | i in 1..n, j in 1..n-2]); % lignes 

% Pas plus de 2 cases consécutives avec mines pour les lignes et les colonnes
constraint forall([ count(grid[i..i+2,j], 1) <= 2 | j in 1..n, i in 1..n-2]); % colonnes
constraint forall([ count(grid[i,j..j+2], 1) <= 2 | i in 1..n, j in 1..n-2]); % lignes 

solve satisfy;

output
   [show(grid[i,j]) ++ if j == n then "\n" else " " endif | i,j in 1..n];
   