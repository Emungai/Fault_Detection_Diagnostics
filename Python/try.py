import sys
sys.path.append('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/utils')

from similarity_mat import sim_matrix

x=[[3,4,3,2,1],
   [4,3,5,1,1],
   [3,5,3,3,3],
   [2,1,3,3,2],
   [1,1,3,2,3]]

S=sim_matrix(x)

