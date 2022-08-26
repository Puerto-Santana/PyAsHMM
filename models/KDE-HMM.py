# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:05:11 2022

@author: fox_e
"""

import pickle
import datetime
import os
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from AR_ASLG_HMM  import AR_ASLG_HMM  as hmm
class KDE_HMM:    
    def __init__(self,train,N,A=None,pi=None,h= None,v=None):
        self.y = train
        self.x = None
        self.T = None
        self.N = N
        self.L = len(train)
        
        mod = hmm(self.y,lengths,self.N,P=0)
        mod.EM()
        aa = mod.A
        vv = mod.forBack[0].gamma
        vv = vv/np.sum(vv,axis=0)
        
        if v is None:
            self.v = vv
        else:
            self.v = v
        self.forback = []
        self.C = None
        if h is None:
            self.h = np.ones([1,self.N])*(4*np.std(train)**5/(3*self.L))**(1./5.)
        else:
            self.h = h
        if  A is None:
            self.A = aa
        else:
            self.A = A
        if pi is None:
            self.pi = np.ones(N)/N
        else:
            self.pi= pi

    def act_probt(self):
        """
        Updates the temporal probability ( i.e., the mean for each time instance, hidden state and variable,) parameter of each forBack object

        Parameters
        ----------
        G : TYPE list
            DESCRIPTION. list with numpy arrays of size KxK
        B : TYPE list
            DESCRIPTION. list of lists with the emission parameters
        p : TYPE numpy array of size NxK
            DESCRIPTION. matrix of number of AR values
        """ 
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forback[i].prob_t(self.y,self.O[aux_len[i]:aux_len[i+1]],self.v,self.h,self.N)

    def act_gamma(self):
        """
        Updates for each forBack object its gamma parameter or latent probabilities

        Parameters
        ----------
        ps : TYPE, optional int
            DESCRIPTION. The default is None. index of the initial distibution
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forback[i].forward_backward(self.A,self.pi,self.O[aux_len[i]:aux_len[i+1]])
            
    def act_params(self):
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forback[0].update_hw(self.O[aux_len[0]:aux_len[1]],self.y,self.N,self.h,self.v)
        numh =self.forback[0].numh
        denh =self.forback[0].denh
        numw =self.forback[0].numw
        denw =self.forback[0].denw
        for i in range(1,len(self.lengths)):
            self.forback[i].prob_t(self.y,self.O[aux_len[i]:aux_len[i+1]],self.v,self.h,self.N)
            numh +=self.forback[0].numh
            denh +=self.forback[0].denh
            numw +=self.forback[0].numw
            denw +=self.forback[0].denw
            
        h = np.sqrt(numh/denh)
        w = numw/denw
        return [h,w]
    
    def act_A(self):
        """
        Updates the parameter A

        Parameters
        ----------
        ta : TYPE boolean list
            DESCRIPTION. list that gives the indices of rows and columns of A to be updated.

        Returns
        -------
        A : TYPE numpy array of size NxN
            DESCRIPTION. The updated transition matrix

        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forback[0].act_aij(self.A,self.N,self.O[aux_len[0]:aux_len[1]])
        numa = self.forback[0].numa
        dena = self.forback[0].dena
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_aij(self.A,self.N,self.O[aux_len[i]:aux_len[i+1]])
            numa = numa +self.forback[i].numa 
            dena = dena +self.forback[i].dena 
            
        A = np.ones([self.N,self.N])
        for j in range(self.N):
            A[j] = numa[j]/dena[j]
            for k in range(self.N):
                A[k][j] = numa[k][j]/dena[k]
        for n in range(self.N):    
            A[n] = A[n]/np.sum(A[n]) # Normalizo para asegurar que cada fila de A sea una distribucion
        return A
    
    def act_pi(self):
        """
        Updates the initial distribution parameters

        Returns
        -------
        npi : TYPE numpy array of size N
            DESCRIPTION. The updated  initial distribution

        """
        api = self.forback[0].gamma[0]
        for i in range(1,len(self.lengths)):
            api = api+self.forback[i].gamma[0]
        npi =  api/len(self.lengths)
        # ind = np.argwhere(npi==0).T[0]
        # npi[ind] = np.exp(-740)
        return npi
    

    
    def fit(self,x,lengths,its=25,err=9e-1):
        tempa =self.A
        tempi =self.pi
        temph = self.h
        tempv = self.v
        for l in range(len(lengths)):
            self.forback.append(forBack())
        self.lengths = lengths
        self.O = x
        self.act_probt()      
        likeli= -1e10
        eps =1
        it = 0
        while eps >err and it<its:
            self.act_gamma()
            self.A = self.act_A()
            self.pi = self.act_pi()
            h, v =  self.act_params()
            self.v = v
            self.h = h
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-self.forback[i].ll)
            self.LogLtrain = likelinew
            self.act_probt()
            eps = likelinew-likeli
            print("it: "+str(it)+", error: "+str(eps) +" h: "+str(self.h)+ ", log-likelihood: "+str(likelinew))
            likeli = likelinew
            if eps >0:
                tempa =self.A
                tempi =self.pi
                temph = self.h
                tempv = self.v
#            print likeli
            it = it+1   
        self.A = tempa
        self.pi = tempi
        self.h = temph
        self.v = tempv
    
    def viterbi(self,O,plot=True,xlabel="Time units",ylabel="index"): 
        """
        Computes the most likely sequence of hidden states 

        Parameters
        ----------
        O : TYPE numpy array of size TxK
            DESCRIPTION. testing dataset
        plot : TYPE, optional bool
            DESCRIPTION. The default is True. 
        maxg : TYPE, optional bool
            DESCRIPTION. The default is False. Uses max in the labelling 
        absg : TYPE, optional bool
            DESCRIPTION. The default is True. Uses absolute value in the labelling
        indexes : TYPE, optional bool
            DESCRIPTION. The default is False. Report indexes instead of labels
        ps : TYPE, optional int
            DESCRIPTION. The default is None. Uses a different initial distribution

        Returns 
        -------
        TYPE numpy array of size T 
            DESCRIPTION. Most likely sequence of hidden states

        """
        T =len(O)
        fb = forBack()
        fb.prob_t(self.y,O,self.v,self.h,self.N)
        delta = np.log(self.pi)+fb.probt[0]
        psi = [np.zeros(len(self.pi))]
        for i in range(1,T):
            delta = self.viterbi_step(delta,fb,i)
            psi.append(np.argmax(delta+np.log(self.A.T),axis=1))
            
        psi = np.flipud(np.array(psi)).astype(int)
        Q = [np.argmax(delta)]
        for t in np.arange(0,T-1):
            Q.append(psi[t][Q[t]])
        Q=np.array(Q)
        if plot==True:
            plt.figure("Sequence of labeled hidden states",figsize=(4.5,3))
            plt.clf()
            plt.title("Sequence of labeled hidden state")
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.plot(Q[::-1])
            plt.grid("on")
            plt.show()
            plt.tight_layout()
            plt.pause(0.02)
        return Q[::-1]
    
    def viterbi_step(self,delta,fb,t):
        """
        Computes a viterbi step at time t
        
        Parameters
        ----------
        delta : TYPE numpy array
            DESCRIPTION. delta_t(i) = max_j [delta_t-1(j)a_ji]*log(P(o|q_t=i,lambda))
        fb : TYPE forBack object
            DESCRIPTION. contains information about latent probabilities for a dataset
        t : TYPE int
            DESCRIPTION. time 

        Returns 
        -------
        TYPE numpy array of size N 
            DESCRIPTION. Information of delta_{t+1} 

        """
        return np.max(delta+np.log(self.A.T),axis=1)+fb.probt[t]
    
    def log_likelihood(self,O,xunit=False,ps = None):
        """
        Computes the log-likelihood of a dataset O.
        it uses the  current learned parameters of the model

        Parameters
        ----------
        O : TYPE Numpy array of size T'xK
            DESCRIPTION. Observations to be tested
        pena : TYPE, optional bool
            DESCRIPTION. The default is False. it applies the BIC penalization
        xunit : TYPE, optional bool
            DESCRIPTION. The default is False. it applies the BIC per unit of data
        ps : TYPE, optional  int
            DESCRIPTION. The default is None. It changes the initial base distribution

        Returns
        -------
        TYPE float
            DESCRIPTION. The score of the observations O

        """ 
        fb = forBack()
        fb.prob_t(O,self.v,self.h,self.N)
        fb.forward_backward(self.A,self.pi,O)
        Clist = fb.ll
        log = np.sum(-Clist)
        del fb
        if xunit == False:
            return log
        else:
            return log/float(len(O))
        
        
    def plot_densities(self,nombre=""):
        y_min = np.min(self.y)
        y_max = np.max(self.y)
        xline = np.linspace(y_min,y_max,10000)
        densidades = np.zeros([len(xline),self.N])
        for t in range(len(xline)):
            densidades[t] = np.sum(self.v/self.h*self.gaussian_kernel((xline[t]-self.y)/self.h),axis=0)
        plt.figure("Hidden densities")
        plt.clf()
        for i in range(self.N):
            plt.plot(xline,densidades[:,i].T,label="state: "+str(i)+ " bandwidth:"+str(round(self.h.T[i][0],3)))
        plt.legend()
        plt.xlabel("$X$")
        plt.ylabel("$P$")
        plt.grid("on")
        plt.tight_layout()
        
    
    def gaussian_kernel(self,x):
       return  np.exp(-0.5*x**2)/(np.sqrt(2*np.pi)) 

    
    
    
class forBack:
    
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.rho = None
        self.ll = None
        self.probt = None
        self.numh = None
        self.denh =None
        self.numw = None
        self.denw = None
        
        
    def gaussian_kernel(self,x):
       return  np.exp(-0.5*x**2)/(np.sqrt(2*np.pi)) 
        
    def prob_t(self,y,x,v,h,N):
        T = len(x)
        data = np.zeros([T,N])
        for t in range(T):
            zt = (x[t]-y)/h
            data[t] = np.sum(v*self.gaussian_kernel(zt)/h,axis=0)
        self.probt = np.log(data)
        
    def forward_backward(self,A,pi,O,ps=None): 
        """
        Does the scaled forward-backward algorithm using logarithms

        Parameters check AR_ASLG_HMM for more information 
        ----------
        ps : TYPE, optional int
            DESCRIPTION. The default is None. index of the initial distibution
        """
        T = len(O)
        if ps == None:
            alfa = np.log(pi) + self.probt[0]
        else:
            alfa = np.log(A[ps]) + self.probt[0]
        Cd = -np.max(alfa)-np.log(np.sum(np.exp(alfa-np.max(alfa))))
        Clist = np.array([Cd])
        Clist = np.array([Cd])
        alfa = alfa+Cd
        ALFA = [alfa]
        for t in range(1,T):
            alfa = self.forward_step(alfa,A,t)
            Cd = -np.max(alfa)-np.log(np.sum(np.exp(alfa-np.max(alfa))))
            Clist = np.concatenate([Clist,[Cd]])
            alfa = Cd + alfa
            ALFA.append(alfa)
        ALFA= np.array(ALFA)
        self.alpha = ALFA
        
        beta =np.zeros(len(pi))
        Clist =Clist[::-1]
        beta = beta + Clist[0]
        BETA = [beta]
        for t in range(1,T):
            beta = self.backward_step(beta,A,T-t)
            beta = beta + Clist[t]
            BETA.append(beta)
        BETA= np.flipud(np.array(BETA))
        self.beta = BETA
        self.ll = Clist
        self.gamma = np.exp(self.alpha+self.beta)/np.sum(np.exp(self.alpha+self.beta),axis=1)[np.newaxis].T
        # self.gamma = np.exp(self.gamma)
        
        
        
    
    def checkzero(self,z):
        """
        Returns a modified vector z with no zero instances
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)
        y[inds] = np.exp(-740)
        return y
    
    def forward_step(self,alfa,A,t,k_eval=False,prob=None):
        """
        Does an inductive step in the alfa variable
        alfa_t(i) = P(O1,..., Ot, q_t=i|lambda)

        Parameters check AR_ASLG_HMM for more information 
        ----------
        alfa : TYPE numpy array of size N
            DESCRIPTION. alfa_t(i)
        t : TYPE int
            DESCRIPTION. time isntance
        k_eval : TYPE, optional bool
            DESCRIPTION. The default is False. Use proposed or current probabilities
        prob : TYPE, optional numpy array fo size N
            DESCRIPTION. The default is None. proposed probabilities

        Returns
        -------
        TYPE numpy array  of size N
            DESCRIPTION. [alfa_{t+1}(i)]_i

        """
        if k_eval == False:
            maxi =np.max(alfa)
            arg = np.dot(np.exp(alfa-maxi),A)
            arg = self.checkzero(arg)
            logaa = np.log(arg)
            return self.probt[t] +maxi+logaa
        else:
            maxi =np.max(alfa)
            logaa = np.log(np.dot(np.exp(alfa-maxi),A))
            return prob[t] +maxi+logaa
        # return np.exp(np.log(np.dot(alfa,A))+self.prob_comp[t])
        
    
    
    def backward_step(self,beta,A,t): 
        """
        Does an inductive step in the beta variable
        beta_t(i) = P(Ot+1,..., OT| q_t=i,lambda)

        Parameters check AR_ASLG_HMM for more information 
        ----------
        beta : TYPE numpy array of size N
            DESCRIPTION.  beta_t(i)
        t : TYPE int
            DESCRIPTION. time isntance

        Returns
        -------
        TYPE numpy array  of size N
            DESCRIPTION. [beta_{t+1}(i)]_i

        """
        maxi = np.max(beta)
        arg = np.dot(A,np.exp(self.probt[t]+beta-maxi))
        arg = self.checkzero(arg)
        logba = np.log(arg)
        return maxi+logba
    
    def act_aij(self,A,N,O): 
        """
        Updates the parameter A

        Parameters check AR_ASLG_HMM for more information 
        ----------
        """
        T = len(O)
        bj = self.probt
        nume = []
        deno = []
        for i in range(N):
            alfat = (self.alpha.T[i])[:T-1] 
            betat = self.beta[1:].T 
            num = A[i]*np.sum(np.exp(alfat+betat+ bj[1:].T),axis=1)
            den = np.sum(num)
            nume.append(num)
            deno.append(den)
        self.numa = np.array(nume)
        self.dena = np.array(deno)
        
        # opcional
        
        for i in range(len(self.dena)):
            if self.dena[i] == 0:
                self.dena[i] = np.exp(-740)
    
    def update_hw(self,x,y,N,h,v):
        """
        

        Returns
        -------
        None.

        """
        T = len(x)
        L= len(y)
        self.numh = np.zeros([1,N])
        self.numw = np.zeros([L,N])
        self.denh = np.zeros([1,N])
        self.denw = np.zeros([1,N])
        for t in range(T):
            tmons  = v*self.gaussian_kernel((x[t]-y)/h)
            psitl = tmons/(h*np.exp(self.probt[t]))*self.gamma
            self.numh += np.sum(psitl*(x[t]-y)**2,axis=0)
            self.numw += psitl
            self.denh += np.sum(psitl,axis=0) 
        self.denw = self.denh
        

#%% Datos Gamma
# import random
y = np.concatenate([-(np.random.gamma(1,6,3000)-15),-np.random.gamma(3,6,3000)-50])[np.newaxis].T
x = np.concatenate([-(np.random.gamma(1,6,3000)-15),-np.random.gamma(3,6,3000)-50])[np.newaxis].T

order = np.concatenate([np.arange(500),np.arange(3000,3500),np.arange(500,1500),np.arange(3500,4500),np.arange(1500,3000),np.arange(4500,6000)])
 
# order1 = np.arange(6000)
# order2 = np.arange(6000)
# random.shuffle(order1)
# random.shuffle(order2)

y = y[order]
x = x[order]
# 

lengths = np.array([len(x)])
model1 = KDE_HMM(y,2)
model1.fit(x,lengths)



segm = model1.viterbi(x)
index1 = np.argwhere(segm==0).T[0]
index2 = np.argwhere(segm==1).T[0]
plt.figure("Distribution separation")
plt.clf()
plt.title("OMG")
plt.hist(y[index1],bins=20,label="Test Group1",alpha=0.5,color="blue",density=True)
plt.hist(y[index2],bins=20,label="Test Group2",alpha=0.5,color="orange",density=True)
plt.legend()
plt.grid("on")
plt.tight_layout()

model1.plot_densities()
#%%%
def simul_GMM(mix,mus,sigmas,n):
    data = np.zeros(n)
    macul = np.concatenate([[0],np.cumsum(mix)])
    zs = np.random.uniform(0,1,n) 
    for l in range(n):
        index = np.argmax(macul-zs[l]>0)-1
        data[l] = np.random.normal(mus[index],sigmas[index],1)
    return data[np.newaxis].T
        
#%% datos Gauss
y0 = simul_GMM([0.1,0.2,0.3,0.4],[8.,9.,10.,11.],[0.5,0.5,0.5,0.5],3000)
x0 =simul_GMM([0.1,0.2,0.3,0.4],[8.,9.,10.,11.],[0.5,0.5,0.5,0.5],3000)
y1 = simul_GMM([0.2,0.3,0.1,0.4],[-1.,-2.,-4.,-5.],[0.5,0.5,0.5,0.5],3000)
x1 =simul_GMM([0.2,0.3,0.1,0.4],[-1.,-2.,-4.,-5.],[0.5,0.5,0.5,0.5],3000)

order = np.concatenate([np.arange(500),np.arange(3000,3500),np.arange(500,1500),np.arange(3500,4500),np.arange(1500,3000),np.arange(4500,6000)])

x = np.concatenate([x0,x1],axis=0)
y = np.concatenate([y0,y1],axis=0)
order = np.concatenate([np.arange(500),np.arange(3000,3500),np.arange(500,1500),np.arange(3500,4500),np.arange(1500,3000),np.arange(4500,6000)])
 
y = y[order]
x = x[order]

lengths = np.array([len(x)])
model2 = KDE_HMM(y,2)
model2.fit(x,lengths)
model2.viterbi(x)
model2.plot_densities(nombre="GMM")

segm = model2.viterbi(x)
index1 = np.argwhere(segm==0).T[0]
index2 = np.argwhere(segm==1).T[0]
plt.figure("Distribution separation")
plt.clf()
plt.title("OMG 2")
plt.hist(y[index1],bins=20,label="Test Group1",alpha=0.5,color="blue",density=True)
plt.hist(y[index2],bins=20,label="Test Group2",alpha=0.5,color="orange",density=True)
plt.legend()
plt.grid("on")
plt.tight_layout()





 
