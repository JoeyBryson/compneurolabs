import numpy as np
from basic_hopfield import seven_segment
inb = lambda x: (x + 1)/2
nm = lambda x :np.linalg.norm(x)

class Dentate_Gyrus():
    def __init__(self,pats):
        self.n = len(pats[0]); self.m = 25*len(pats[0]) # SPARSITY IS N/M for simplicity and to keep it at 0.04
        self.W = (np.random.rand(self.n,self.m)*2 - 1).T; self.W = self.W - np.mean(self.W)
        self.pairs = []
        #processes
        self.ExDims = lambda x: self.W @ x; 
        self.Kwins =  lambda x: (x >= np.partition(x, -self.n)[-self.n]).astype(float)*2 - 1
        self.hebb_up = lambda x: np.outer(x,x)
     
    # Vector Transformation (general)
    def forward(self, x): 
        return self.Kwins(self.ExDims(x))
    
    # Vector Transformation (learnt patterns) and keeps a link between x and x_enc
    def pat_forward(self,x):
        enc = self.forward(x)
        self.pairs.append([x, enc])
        return enc
     
    # SET WEIGHTS IN MxM matrix
    def weights(self,X):
        W = np.zeros([self.m,self.m])
        inc = np.zeros([len(X[:,0]),self.m])
        for i in range(len(X[:,0])):
            inc[i] = self.pat_forward(X[i])    
        #create weight matrix with transformed patterns
        for i in range(len(X[:,0])):
            # print(X.shape);print(W.shape)
            W += self.hebb_up(inc[i,:])
        np.fill_diagonal(W, 0); W /= self.n
        return W
    
    # COMPARE A VECTOR TO ENCODED PATTERNS
    def classify(self,vec,printval=False,printit = False):
        ls = np.zeros(len(self.pairs));
        for i in range(len(ls)):
            val = self.pairs[i][1]
            ls[i] = np.dot(vec , val)/(nm(vec)*nm(val)+ 1e-9)
        if printval==True:print(ls)
        if printit ==True: seven_segment(self.pairs[np.argmax(ls)][0])
        return ls