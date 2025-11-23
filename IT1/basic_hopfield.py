from math import *
import numpy as np

fancy=True ; np.random.seed(42)

def seven_segment(pattern):
    global fancy;fancy = True
    left,lower,right = \
        ('▕','__','▏') if fancy else \
        ('|','__','|')
    def vert(d1,d2,d3):
        print(
        (left  if d1 else " ")+
        (lower if d3 else "  ")+
        (right if d2 else " "))
    bits = [(ai==1) for ai in pattern]
    print(" "+lower+" " if bits[0] else "   ")
    vert(*bits[1:4])
    vert(*bits[4:7])
    number=0
    for i in range(0,4):
        if bits[7+i]:
            number+=pow(2,i)
    print('%X'%int(number))

hexdigits = np.int8([\
    [1,1,1,0,1,1,1, 0,0,0,0],
    [0,0,1,0,0,1,0, 1,0,0,0],
    [1,0,1,1,1,0,1, 0,1,0,0],
    [1,0,1,1,0,1,1, 1,1,0,0],
    [0,1,1,1,0,1,0, 0,0,1,0],
    [1,1,0,1,0,1,1, 1,0,1,0],
    [1,1,0,1,1,1,1, 0,1,1,0],
    [1,0,1,0,0,1,0, 1,1,1,0],
    [1,1,1,1,1,1,1, 0,0,0,1],
    [1,1,1,1,0,1,1, 1,0,0,1],
    [1,1,1,1,1,1,0, 0,1,0,1],
    [0,1,0,1,1,1,1, 1,1,0,1],
    [1,1,0,0,1,0,1, 0,0,1,1],
    [0,0,1,1,1,1,1, 1,0,1,1],
    [1,1,0,1,1,0,1, 0,1,1,1],
    [1,1,0,1,1,0,0, 1,1,1,1]])*2-1

zero, one, two, three, four, five, six, seven, eight, nine, hexA, hexB, hexC, hexD, hexE, hexF = hexdigits
all_pats = np.array([zero,one, two, three, four, five, six, seven, eight, nine, hexA, hexB, hexC, hexD, hexE, hexF]);

test1 = np.int8([1,-1,1,1,-1,1,1,-1,-1,-1,-1])
test2 = np.int8([1,1,1,1,1,1,1,-1,-1,-1,-1])
test3 = np.int8([-1,1,1,1,-1,1,-1,-1,-1,-1,-1])

#funcs
sgn = lambda x:(np.array(x)>0)*2-1
E   = lambda x,W:-0.5*x.T@W@x
base_weights = lambda X: (X.T@X/X.shape[0])-np.eye(X.shape[1])



def converge_energy(s,W,maxiter=20,printend=False,printenergy=False):
    previous_energy = E(s,W)
    for i in range(maxiter):
        if printenergy: print('iteration %2d: E = %6.2f'%(i,previous_energy))
        s = sgn(W@s); energy = E(s,W)
        if energy>=previous_energy: break
        previous_energy = energy
    if printend: seven_segment(s); print(previous_energy); print()
    return s

