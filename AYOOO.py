from math import *
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA

fancy=False ; np.random.seed(42)

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

test1 = np.int8([1,-1,1,1,-1,1,1,-1,-1,-1,-1])
test2 = np.int8([1,1,1,1,1,1,1,-1,-1,-1,-1])
test3 = np.int8([-1,1,1,1,-1,1,-1,-1,-1,-1,-1])

sgn = lambda x:(np.array(x)>0)*2-1
E   = lambda x,W:-0.5*x.T@W@x

class Dentate_Gyrus():
    def __init__(self,pats,):
        self.n = len(pats[0]); self.m = 10*len(pats[0])
        self.W = (np.random.rand(self.n,self.m)*2 - 1).T
        # print(self.W.shape);
        # print(np.dot(self.W.T,self.W))
        
        #processes
        self.ExDims = lambda x: self.W @ x; 
        self.Kwins =  lambda x: (x >= np.partition(x, -self.n)[-self.n]).astype(float)*2 - 1
        self.hebb_up = lambda x: np.outer(x,x)
        
    def forward(self, x): 
        # print(x);
        it = self.Kwins(self.ExDims(x))
        # print(self.ExDims(x).shape)
        return it
        
    def weights(self,X):
        W = np.zeros([self.m,self.m])
        inc = np.zeros([len(X[:,0]),self.m])
        for i in range(len(X[:,0])):
            inc[i] = self.forward(X[i])
        X, _ = sp.linalg.qr(inc.T);X = X.T; print(X.shape)    
        #update weight matrix with tranformed patterns
        for i in range(len(X[:,0])):
            W += self.hebb_up(X[i,:].T)
        np.fill_diagonal(W, 0)
        return W,X
        
        
def converge_states(s,W,maxiter=50):
    ls = E(s,W); ps = None;
    for i in range(maxiter):
        print(f'iter: {i} ///  energy = {ls}')
        s_ = sgn(W@s); energy = E(s_,W)
        if energy >= ls: print(f'final energy  = {energy}'); break
        ps = s;s = s_;ls = energy;
    return s_


def classify(vec,pats):
    ls = np.zeros(len(pats[:,0]))
    for i in range(len(ls)):
        ls[i] = np.dot(vec , pats[i,:])/(nm(vec)*nm(pats[:,i])+ 1e-9)
    print(ls)
    return np.argmax(ls)

# def check(vec,pat):
#     return np.dot(vec , pat)/(nm(vec)*nm(pat))

nm = lambda x :np.linalg.norm(x)

if __name__=="__main__":
    X = np.array([one,three,six]);
    #DG process
    DG = Dentate_Gyrus(hexdigits)
    tf_test1 = (DG.forward(test1));tf_test2 = DG.forward(test2);tf_test3 = DG.forward(test3)
    W,encpats = DG.weights(hexdigits)
    
    
    

    print("\n\ntest1")
    seven_segment(test1)
    print('--------------')
    xt1 = converge_states(tf_test1,W)
    # print(np.all(xt1 == DG.forward(three)))
    # print(np.sum((xt1 - DG.forward(three))**2))
    max_i = classify(xt1 ,encpats)
    seven_segment(hexdigits[max_i])
    
    
    print("\n\ntest2")
    seven_segment(test2)
    print('--------------')
    xt2 = converge_states(tf_test2,W)
    max_i = classify(xt2 ,encpats)
    seven_segment(hexdigits[max_i])
    
    print("\n\ntest3")
    seven_segment(test3)
    print('--------------')
    xt3 = converge_states(tf_test3,W)
    max_i = classify(xt3 ,encpats)
    seven_segment(hexdigits[max_i])
    
    


    
    
    # # #tries all patterns
    
    # for a,s in enumerate(encpats):
    #     for i in range(20):
    #         s_ = sgn(W@s)
    #         if np.all(s==s_): break
    #         s = s_
    #     # print("\nThis pattern should be 0x%X:"%a)



   




