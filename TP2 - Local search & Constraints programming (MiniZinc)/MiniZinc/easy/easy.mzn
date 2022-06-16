include "globals.mzn";

int: n;  % Taille de la grille et nb. de drones
array[1..n] of var 1..n: columns;  % 1 drone par colomne

constraint alldifferent(columns);  % 1 drone par ligne maximum
constraint forall([ abs(columns[i+j] - columns[i]) != j | i in 1..n-1, j in 1..n-i ]); % diagonales

solve satisfy;

% i = colomne, j = ligne
output	
  [if fix(columns[i]) == j then "D " else ". " endif 
   ++
   if j == n then "\n" else "" endif	
   | i, j in 1..n];
   
   
