from basic_hopfield import *
from DG import *
import numpy as np
import sys

np.random.seed(42)


# SIMPLE HOPFIELD NETWORK
def basic_process():
    X = np.array([one,three,six]);
    W = base_weights(X)
    
    # Test pattern 1 error corrects to 3
    print("test1")
    seven_segment(test1)
    converge_energy(test1,W,printend=True,printenergy=True)
    print('________')
    # Test pattern 2 error corrects to 6
    print("test2")
    seven_segment(test2)
    converge_energy(test2,W,printend=True,printenergy=True)
    print('________')
    # see what incomplete 4 corrects to 
    print("test3")
    seven_segment(test3)
    converge_energy(test3,W,printend=True,printenergy=True)
 
 # TEST SIMILARITIES BETWEEN ALL patterns when encoded.(MEASURES OVERLAP)   
def print_sims(DG):
    for i in range(len(DG.pairs)):
        print(f'pattern {hex(i)} cosine similarities :')
        DG.classify(DG.pairs[i][1],printval=True);print('\n')
    
 
# PATTERN SEPARATED     
def DGattempt():
    X = all_pats
    DG = Dentate_Gyrus(X)
    W = DG.weights(X)
    
            
    print_sims(DG)
          
        
    #Test pattern 1 error corrects to 3
    print("\ntest1")
    seven_segment(test1)
    tf_test1 = DG.forward(test1)
    out1 = converge_energy(tf_test1,W,printenergy=True)
    DG.classify(out1,printval=True,printit=True)
    
    # Test pattern 2 error corrects to 6
    print("\ntest2")
    seven_segment(test2)
    tf_test2 = DG.forward(test2)
    out2 = converge_energy(tf_test2,W,printenergy=True)
    DG.classify(out2,printval=True,printit=True)
    
    # see what incomplete 4 corrects to 
    print("\ntest3")
    seven_segment(test3)
    tf_test3 = DG.forward(test3)
    out3 = converge_energy(tf_test3,W,printenergy=True)
    DG.classify(out3,printval=True,printit=True)
    

if __name__ =='__main__':
    # basic_process()
    print('----------------------------------------------')
    DGattempt()