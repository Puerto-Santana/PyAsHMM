# -*- coding: utf-8 -*-
"""
Created on Fri August 24 14:07:03 2022
@author: Carlos Puerto-Santana at KTH royal institute of technology, Stockholm, Sweden
@Licence: CC BY 4.0
"""
import pickle
import datetime
import time
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
class KDE_AsHMM:    
    def __init__(self,O,N,A=None, pi=None, h= None, M=None, v=None, 
                  p=None, P=None, G=None, p_cor=5,struc=True,lags=True):
        """
        Generates an object of type KDE-AsHMM 

        Parameters
        ----------
        O : TYPE numpy array 2D
            DESCRIPTION training data, files are instances and columns variables
        N : TYPE integer positive
            DESCRIPTION. Number of hidden states
        A : TYPE, optional  numpy array 2D
            DESCRIPTION. The default is None. Transition Matrix of size NxN
        pi : TYPE, optional numpy array 1D
            DESCRIPTION. The default is None. Initial distribution vector of size N
        h : TYPE, optional  numpy array 2D
            DESCRIPTION. The default is None. Initial bandwidths of size NxK
        M : TYPE, optional list of lists
            DESCRIPTION. The default is None. List of size NxK lists with the coefficients of dependencies
        v : TYPE, optional numpy array 2D
            DESCRIPTION. The default is None. Kernel mixture weights, size (T-P)xK, recommended to fix P also
        p : TYPE, optional numpy array 2D
            DESCRIPTION. The default is None. matrix of AR dependencies of size NxK
        P : TYPE, optional numpy array 2D
            DESCRIPTION. The default is None. Maximum lag 
        G : TYPE, optional list of lists
            DESCRIPTION. The default is None. list of size N with binary matrices of size KxK
        p_cor : TYPE, optional int
            DESCRIPTION. The default is 5. maximum correlation order to determine P
        struc : TYPE, optional bool
            DESCRIPTION. The default is True. Enables structural optimization 
        lags : TYPE, optional bool
            DESCRIPTION. The default is True. Enables AR order optimization

        Returns
        -------
        None.

        """
        
        # Insert training data
        if type(O) is np.ndarray:
            if len(O.shape) == 2:
                self.O = O
            else:
                raise Exception("Invalid data dimension, it must has 2 dimensions")
        else:
           raise Exception("Invalid data input")
           
         # Insert number of hidden states
        if type(N) is int:
            if N> 0:
                self.N = N
            else:
                raise Exception("Number of hidden states must be positive")
        else:
            raise Exception("Number of hidden states must be a positive integer")
            
        # Insert data length and number of variables
        self.length = len(O)
        self.K       = len(O[0]) 
        self.forback = []
        
        # Optimization parameters
        if type(struc) is bool :
            self.struc   = struc    
        else:
            raise Exception("Struc must be a boolean")
            
        if type(lags) is bool:   
            self.lags    = lags   
        else:
            raise Exception("lags must be a boolean")
        
        # Setting maximum allowed AR values
        if P is None:
            self.P = self.det_lags(p_cor=p_cor)
        else:
            if type(P) is int:
                if P>=0:
                    self.P = P
                else:
                    raise Exception("Maximum lag P must be greater or equal to zero")
            else:
                raise Exception("Maximum lag  P must be an integer")
                
        # Auxiliary model for transition matrix
        # if v is None or A is None:
        #     aux_model = hmm(O,np.array([self.length]),N,P=self.P)
        #     aux_model.EM()
            
        
        # Setting the bandwidths    
        if h is None:
            self.h = np.ones([self.N,self.K])*(4*np.std(O,axis=0)**5/(3*self.length))**(1./5.)
        else:
            if type(h) is np.ndarray:
                if len(h.shape) ==2:
                    if len(h) == self.N and len(h[0]) ==self.K:
                        self.h = h
                    else:
                        raise Exception("Input h does not match the dimensions NxK")
                else:
                    raise Exception("Input h has more or less than 2 dimensions")
            else:
                raise Exception("Input h must be a numpy array of 2 dimensions")
                
        #  Setting the AR orders
        if p is None:
            if lags == True:
                self.p = np.zeros([N,self.K]).astype(int)
            if lags == False:
                self.p = (self.P*np.ones([N,self.K])).astype(int)
        else:
            if type(p)  is np.ndarray:
                if len(p.shape) == 2:
                    if np.sum(p<0) == 0:
                        self.p = p
                    else:
                        raise Exception("No negative values in p are allowed")
                else:
                    raise Exception("p must hast two dimensions")
            else:
                raise Exception("p must be a numpy array")
            
        # Setting the initial transition matrix
        if  A is None:
            
            # Ap = aux_model.A
            # self.A = Ap
            A = np.ones([self.N,self.N])
            np.fill_diagonal(A,999)
            self.A = A/np.sum(A,axis=1)
        else:
            if type(A) is np.ndarray:
                if len(A.shape) == 2:
                    if np.sum(p<0) == 0:
                        self.A = A
                    else:
                        raise Exception("No negative values in A are allowed")
                else:
                    raise Exception("A must hast two dimensions")
            else:
                raise Exception("A must be a numpy array")
            
        # Setting the initial distribution vector
        if pi is None:
            self.pi = np.ones(N)/N
        else:
            if type(pi) is np.ndarray:
                if len(pi.shape) == 1:
                    if np.sum(pi<0) == 0:
                        self.pi = pi
                    else:
                        raise Exception("No negative values in pi are allowed")
                else:
                    raise Exception("pi must hast one dimensions")
            else:
                raise Exception("pi must be a numpy array")
            
        # Setting the list of graphs 
        if G is None:
            G = self.prior_G(N,self.K)
            L= []
            for i in range(N):
                L.append(np.arange(self.K))
            self.G = G
            self.L = L
        else:
            self.L = []
            if len(G) != self.N:
                 raise Exception("G must be a list of N components")
            else:
                for i in range(self.N):
                    if len(G[i]) != self.K or  len(G[i].T) != self.K  :
                        raise Exception("G at component "+ str(i) +"  is not of shape KxK")
            for i in range(self.N):
                booli, Li = self.dag_v(G[i])
                if booli == True:
                    self.L.append(Li)
                else:
                   raise Exception("G at component "+ str(i) +"  is not a DAG")
            self.G = G
                
        # setting the dependencies coefficients
        if M is None:
            M = []
            for i in range(self.N):
                Mi = []
                for k in range(self.K):
                    Mik = np.zeros([1,int(np.sum(self.G[i][k])+self.p[i][k])])
                    Mi.append(Mik)
                M.append(Mi)
            self.M = M
        else:
            self.M= M
            
        # Setting the kernel mixture coefficients
        if v is None:
            # self.v = self.init_v(self.O, self.N, self.h[0], self.P)
            # vv = aux_model.forBack[0].gamma
            # vv = vv/np.sum(vv,axis=0)
            # self.v  = vv
            vv = np.random.uniform(0.4,0.6,[self.length-self.P,self.N])
            self.v = vv/np.sum(vv,axis=0)
        else:
            if type(v) is np.ndarray:
                if len(v.shape) == 2:
                    if np.sum(v<0) == 0:
                        self.v = v 
                        print("v must be of size T-P, it is recommended to fix P as well")
                    else:
                        raise Exception("No negative values in pi are allowed")
                else:
                    raise Exception("pi must hast one dimensions")
            else:
                raise Exception("pi must be a numpy array")
                    
        
    def gaussian_kernel(self,x):
        """
        Computes the exponential kernel for the given argument x

        Parameters
        ----------
        x : TYPE numpy ndarray
            DESCRIPTION. argument for the Gaussian kernel

        Returns The Gaussian kernel evaluation for the given input
        -------
        TYPE numpy ndarray
            DESCRIPTION.

        """
        return  np.exp(-0.5*x**2)/(np.sqrt(2*np.pi)) 
   
    def init_v(self,x,N,h,P):
        """
        A possible way to star the v parameters. Uses the likelihood of the data
        to determine if an observation belongs to state or to another. 
        Parameters
        ----------
        x : TYPE numpy ndarray
            DESCRIPTION. training data
        N : TYPE int
            DESCRIPTION. number of hidden states
        h : TYPE numpy ndarray
            DESCRIPTION. bandwidths
        P : TYPE  int
            DESCRIPTION. maximum lag

        Returns
        -------
        v : TYPE numpy ndarray
            DESCRIPTION. instance rlevancy for each cluster
        """
        index = []
        K = len(x[P])
        for n in range(N):
            index.append([])
        T = len(x)
        index[0].append(P)
        e_empty = True
        
        llb = []
        for t in range(P+1,T-P):
            ll = np.zeros(N)        
            dg = np.zeros(N)
            for i in range(N):
                if len(index[i]) != 0:
                    ll[i] = np.sum(np.prod(1/h*self.gaussian_kernel((x[t+P]-x[index[i]])/h),axis=1))/float(len(index[i]))
                    if ll[i] > (0.01)**(K):
                        dg[i]= 1
            if e_empty == False or np.sum(dg) != 0:
                j = np.argmax(ll)
                index[j].append(t)
            
            if e_empty == True and np.sum(dg)==0:
                emp = 0
                for i in range(N):
                    if len(index[i]) == 0:
                        index[i].append(t)
                        break
                for i in range(N):
                    if len(index[i]) == 0:
                        emp = emp +1
                if emp ==0:
                    e_empty = False 
                llb.append(ll)
        
            v = np.ones([T-P,N])*np.exp(-744)
            for i in range(N):
                v[index[i],i] = 1
            v = v/np.sum(v,axis=0)
        return v
    
    def correlogram(self,O,p_cor=0):
        """
        Computes the correlogram of a variable with p_cor past values

        Parameters
        ----------
        O : TYPE Numpy array of size TxK
            DESCRIPTION. Observations  to be used in training. T is the number of instances and K is the number of variables
            In case of using two or more datatsets for training, concatenate them such that T=T1+T2+T3+... Ti is the length of dataset i
        p_cor : TYPE, int
            DESCRIPTION. The default is 0. number of AR values for the Box-Jenkins methodology

        Returns an array of size p_cor with the correlogram 

        """
        if p_cor==0:
            p_cor = int(len(O)/4.)
        n = len(O)
        y = O[::-1]
        mean = np.mean(y)
        var = [np.sum((y-mean)**2)/n]
        for k in range(1,p_cor+1):
            ylag = np.roll(y,-k)
            num = np.sum((y[:-k]-mean)*(ylag[:-k]-mean))*(n-k)**(-1)
            var.append(num)
        rho = np.array(var)/var[0]
        return rho[1:]

    def pcorrelogram(self,O,p=0,title="",plot=True):
        """
        Computes the partial correlogram function of a variable with p past values

        Parameters
        ----------
        O : TYPE Numpy array of size TxK
            DESCRIPTION. Observations  to be used in training. T is the number of instances and K is the number of variables
            In case of using two or more datatsets for training, concatenate them such that T=T1+T2+T3+... Ti is the length of dataset i
        p : TYPE, optional int
            DESCRIPTION. The default is 0.
        title : TYPE, optional str
            DESCRIPTION. The default is "".
        plot : TYPE, optional bool
            DESCRIPTION. The default is True.

        Returns a lsit with the correlogram and partial-correlogram, both of size p
        """
        if p==0:
            p = int(len(O)/4.)
        n = len(O)
        rho = self.correlogram(O,p)
        prho = [rho[0]]
        for k in range(1,p):
            a = rho[:k+1]
            tor = np.concatenate([rho[:k][::-1],[1.],rho[:k]])
            b = [tor[-k-1:]]
            for j in range(1,k+1):
                b.append(np.roll(tor,j)[-k-1:])
            b = np.array(b)
            coef = np.linalg.solve(b,a)
            prho.append(coef[k])
        prho = np.array(prho)
        lags = np.arange(1,p+1)
        ucof = np.ones(p)*1.96*(1./np.sqrt(n))
        icof = np.ones(p)*-1.96*(1./np.sqrt(n))
        if( plot == True):
            plt.figure("PCorrelogram"+title)
            plt.subplot(211)
            plt.title("Partial correlogram"+title)
            plt.bar(lags,prho, label = "Partial correlations")
            plt.plot(lags,ucof,linestyle=":",color="black",label ="Confidence bands")
            plt.plot(lags,icof,linestyle=":",color="black")
            plt.plot(lags,np.zeros(p),color = "black",linewidth =0.8)
            plt.xlabel("Time lags")
            plt.ylabel("Partial Correlation" + r"$\phi_{kk}$")
            plt.ylim([-1,1])
            plt.legend()
            plt.subplot(212)
            plt.title("Correlogram"+title)
            plt.bar(lags,rho, label = "Correlations")
            plt.plot(lags,ucof,linestyle=":",color="black",label ="Confidence bands")
            plt.plot(lags,icof,linestyle=":",color="black")
            plt.plot(lags,np.zeros(p),color = "black",linewidth =0.8)
            plt.xlabel("Time lags")
            plt.ylabel("Correlation "+ r"$\rho_k$" )
            plt.ylim([-1,1])
            plt.legend()
            plt.show
        prho = prho[np.newaxis:]
        return [rho,prho]
    
    def det_lags(self,p_cor=50,plot=False):
        """
        Determines the maximum number of lag parameters that can be used by the model.
        Uses a normality test over the partial correlogram function

        Parameters
        ----------
        p_cor : TYPE, int
            DESCRIPTION. The default is 50.
        plot : TYPE, bool
            DESCRIPTION. The default is False.

        Returns suggested number of lags
        """
        n,d = self.O.shape
        indrho =  []
        energy = []
        thresh = 1.96*(1/np.sqrt(n))
        for i in range(d):
            Oi =self.O.T[i]
            [rhoi,prhoi] = self.pcorrelogram(Oi,p=p_cor,plot=False,title ="variable: "+ str(i))
            prhoi = np.abs(prhoi)
            energy.append(np.sum(prhoi**2))
        ind = np.argmax(np.array(energy))
        Om =self.O.T[ind]
        [rhoi,prhoi] = self.pcorrelogram(Om,p=p_cor,plot=plot,title = " variable: "+ str(ind))
        verdad = True
        j=0
        while verdad == True:
            if np.abs(prhoi[j])< thresh or j+1 == p_cor:
                indrho.append(j-1)
                verdad = False  
            j=j+1
        indrho = np.array(indrho)
        return int(np.max(indrho)+1)

        
# Graphs and DAGS
  
    def prior_G(self,N,K):
        """
        Generates a new graph assuming a naive structure 

        Parameters
        ----------
        N : TYPE int
            DESCRIPTION. number of hidden states
        K : TYPE int 
            DESCRIPTION. number of variables

        Returns a list of naive graphs in the form of KxK matrices
        """
        G = []
        for i in range(N):
            Gi = np.zeros([K,K])
            G.append(Gi)
        return G
    
    def my_fathers(self,G,j):
        """
        Give the indexes of the parents of the variable with index j in the graph G
        
        Parameters
        ----------
        G : TYPE numpy array of size KxK
            DESCRIPTION. a graph matrix
        j : TYPE int
            DESCRIPTION. index of variable whose parents want to be found

        Returns
        -------
        TYPE List
            DESCRIPTION. list with the indexes of the parents of the varible j

        """
        index = np.where(G[j]==1)[0]
        return np.sort(index).astype(int)
    
    def dag_v(self,G):
        """
        Executes the Kahn topological sorting  algorithm. 
    
        Parameters
        ----------
        G : TYPE numpy array size KxK
            DESCRIPTION. a graph with K variables

        Returns
        -------
        list
            DESCRIPTION. A bool indicating if it is a acyclic graph and a topological sort in L. 

        """
        G2 = np.copy(G)
        L = []
        booli = False
        ss = np.sum(G2,axis=1)
        s = np.where(ss==0)[0]
        while len(s)>0:
            si = s[0]
            s = s[1:]
            L.append(si)
            indexi = np.where(G2.T[si]==1)[0]
            for x in indexi:
                G2[x][si] = 0
                if np.sum(G2[x])==0:
                    s =np.concatenate([s,[x]])
        if np.sum(G2)==0:
            booli = True
        return [booli,np.array(L)]
    
      
    def plot_graph(self,route=None,label=None,coef=False,node_siz=800,lab_font_siz=9,edge_font_siz=9):
        """
        plot all the context-specific Bayesian networks

        Parameters
        ----------
        route : TYPE, optional str
            DESCRIPTION. The default is None. route where to save the plots
        label : TYPE, optional list
            DESCRIPTION. The default is None. names of the nodes or variables
        coef : TYPE, optional bool
            DESCRIPTION. The default is False.
        node_siz : TYPE, optional int
            DESCRIPTION. The default is 800.
        lab_font_siz : TYPE, optional int
            DESCRIPTION. The default is 9.
        edge_font_siz : TYPE, optional int
            DESCRIPTION. The default is 9.

        Returns
        -------
        None.

        """
        if route is not  None:
            try:
                os.mkdir(route)
            except:
                pass
        if label == None:
            label = range(self.K)
        dic= range(self.K)
        g = range(self.K)
        for index in range(self.N):
            plt.figure("Graph_"+str(index))
            plt.clf()
            D = nx.DiGraph()
            G = self.G[index]
            B = self.M[index]
            K = len(G)
            for i in range(K):
                pa = self.my_fathers(G,i)
                for j in range(len(pa)):
                    if coef==True:
                        D.add_edges_from([("$"+str(label[pa[j]])+"$","$"+str(label[i])+"$")],weight= round(B[i][j+1],2))
                    else:
                        D.add_edges_from([("$"+str(label[pa[j]])+"$","$"+str(label[i])+"$")],weight ="")
                for r in range(self.p[index][i]):
                    if coef==True:
                        D.add_edges_from([("$"+str(label[i])+":AR_{"+str(r+1)+"}$","$"+str(label[i])+"$")],weight= round(B[i][len(pa)+1+r],2))
                    else:
                        D.add_edges_from([("$"+str(label[i])+":AR_{"+str(r+1)+"}$","$"+str(label[i])+"$")],weight = "")
#                    D.add_edges_from([(str(label[i])+"_AR_"+str(r+1),str(label[i]))])
            pos=nx.circular_layout(D)
            for cc in pos:
                pos[cc] = 2*pos[cc]
            node_labels = {node:node for node in D.nodes()}
            nx.draw_networkx_labels(D, pos, labels=node_labels,font_size=lab_font_siz)
            edge_labels=dict([((u,v,),d['weight']) for u,v,d in D.edges(data=True)])
            nx.draw_networkx_edge_labels(D,pos,edge_labels=edge_labels,font_size=edge_font_siz)
            nx.draw(D,pos, node_size=node_siz,edge_cmap=plt.cm.Reds,arrows=True,node_color="silver",linewidths=2)
            plt.tight_layout()
            if route is not None:
                plt.savefig(route+r"\Graph_g_"+str(round(g[dic[index]],2))+".pdf",dpi=600,bbox_inches="tight")
            plt.show()
            

    
    
    
    def climb_ar(self):
        """
        Performs the AR optimization 

        Returns
        -------
        None.

        """
        for i in range(self.N):
            for k in range(self.K):
                pena= 0.5*(np.sum(self.G[i][k])+self.p[i][k])*np.log(self.length)**2
                sm = self.forback[0].score_i_k(self.O,self.O,self.M,self.p,self.G,self.P,self.v,self.h,i,k)-pena
                while self.p[i][k] +1 <= self.P :
                    p2 = np.copy(self.p)
                    p2[i][k] = p2[i][k] +1
                    [h2,v2,M2] = self.act_params(self.G,self.G,self.p,p2)
                    for j in range(len(self.forback)):
                        pena= 0.5*(np.sum(self.G[i][k])+p2[i][k])*np.log(self.length)**2
                        s2 = self.forback[0].score_i_k(self.O,self.O,M2,p2,self.G,self.P,self.v,self.h,i,k)-pena
                    if s2 > sm:
                        self.p = p2
                        self.M = M2
                        self.h  = h2
                        sm  = s2
                    else:
                        break
            
    def pos_ads(self,G):
        """
        Used to look for the next arcs to add to a graph

        Parameters
        ----------
        G : TYPE numpy array of size KxK
            DESCRIPTION. matrix of a graph

        Returns
        -------
        index : TYPE
            DESCRIPTION. A list where
        index[.][0] is a node which can recieve edges
        index[.][1] is a list  of potential fathers

        """
        index = []
        for i in range(self.K):
            indexi = [i]
            indexj =[]
            for j in range(self.K):
                G2 = np.copy(G)
                if G2[i][j] != 1:
                    G2[i][j] = 1
                    [fool,L] = self.dag_v(G2)
                    if fool ==True:
                        indexj.append(j)
            indexi.append(indexj)
            index.append(indexi)
        return index
    
    def climb_struc(self): 
        """
        Performs the structural non AR optimization 

        Returns
        -------
        None.

        """
        for i in range(self.N):
            for k in range(self.K):
                possi = self.pos_ads(self.G[i])
                son = possi[k][0]
                if len(possi[k][1])!=0:
                    pena= 0.5*(np.sum(self.G[i][k])+self.p[i][k])*np.log(self.length)**2
                    sm = self.forback[0].score_i_k(self.O,self.O,self.M,self.p,self.G,self.P,self.v,self.h,i,k)-pena
                    for j in possi[k][1]:
                        G2 = np.copy(self.G)
                        G2[i][son][j] =1
                        L2 = []
                        for nn in range(self.N):
                            L2.append(self.dag_v(G2[nn])[1])
                        [h2,v2,M2] = self.act_params(self.G,G2,self.p,self.p)
                        pena= 0.5*(np.sum(G2[i][k])+self.p[i][k])*np.log(self.length)**2
                        s2 = self.forback[0].score_i_k(self.O,self.O,M2,self.p,G2,self.P,self.v,self.h,i,k)-pena
                        if s2>sm: 
                            sm= s2
                            self.M = M2
                            self.G = G2
                            self.L = L2
                            self.h = h2
                        else:
                            break


                    
               
    def hill_climb(self):
        """
        Perform the M structural optimization step during the SEM algorithm

        Returns
        -------
        None.

        """
        if self.lags is True:
            self.climb_ar()
        if self.struc is True:
            self.climb_struc()
    

    def act_probt(self):
        """
        Updates the forbck list with all the temporal probabilities

        Returns
        -------
        None.

        """
        self.forback[0].prob_t(self.O,self.O,self.M,self.p,self.G,self.P,self.v,self.h)



    def act_gamma(self):
        """
        Does the forward-backward algorithm for the current forBack object

        Returns
        -------
        None.

        """
        self.forback[0].forward_backward(self.A,self.pi)
            
    def act_params(self,curr_G,G,curr_p,p):
        """
        Updates the parameters v, h and M

        Parameters
        ----------
        curr_G : TYPE list of matrices
            DESCRIPTION. Current G model
        G : TYPE list of matrices
            DESCRIPTION. Proposed G model, same as curr_G in EM
        curr_p : TYPE numpy ndarray
            DESCRIPTION. Current AR orders
        p : TYPE numpy ndarray
            DESCRIPTION. Poposed AR orders, same as curr_p in EM
  
        Returns
        -------
        list with parameters h,w,M
            DESCRIPTION. h bandwidths, w new instance relevancies, M weights

        """
        self.forback[0].update_params(self.O,curr_G,G,curr_p,p,self.P,self.M,self.h)
        numh =self.forback[0].numh
        denh =self.forback[0].denh
        numw =self.forback[0].numw
        denw =self.forback[0].denw
        numm = self.forback[0].numm
        denm = self.forback[0].denm

            
        h = np.sqrt(numh/denh[np.newaxis].T)
        w = numw/denw
        
        M = []
        for i in range(self.N):
            Mi = []
            for k in range(self.K):
                if np.sum(G[i][k]) != 0 or p[i][k]!=0:
                    Mik = np.linalg.solve(denm[i][k],numm[i][k].T).T
                    Mi.append(Mik.reshape([1,len(Mik)]))
                else:
                    Mik = self.M[i][k]
                    Mi.append(Mik)
            M.append(Mi)
        
        return [h,w,M]
    
    def act_A(self):
        """
        Updates the transition matrix parameter

        Returns 
        -------
        A : TYPE numpy ndarray
            DESCRIPTION. Transition matrix

        """
        self.forback[0].act_aij(self.A,self.N)
        numa = self.forback[0].numa
        dena = self.forback[0].dena
            
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
        Updated the pi parameter

        Returns
        -------
        None.

        """
        api = self.forback[0].gamma[0]
        return api
    

    
    def EM(self,its=50,err=9e-1,plot=False):
        """
        Performs the EM algorithm

        Parameters
        ----------
        its : TYPE, optional int
            DESCRIPTION. The default is 50. number of iterations of the EM algorithm
        err : TYPE, optional float
            DESCRIPTION. The default is 9e-1. error bound to end the algorithm
        plot : TYPE, optional bool
            DESCRIPTION. The default is False. plot samples from the learned model for each EM ireration (suitable for dimensions <4)

        Returns
        -------
        None.

        """
        if type(its) is not int:
            raise Exception("The parameter its  must be an integer")
        if type(err) is not float:
            raise Exception("The parameter err must be a float")
        print("EM optimization...")
        tick = time.time()
        tempa =self.A
        tempi =self.pi
        temph = self.h
        tempv = self.v
        tempM = self.M
        self.forback = []
        self.forback.append(forBack())
        self.act_probt()      
        likeli = -1e10
        eps = 1
        it = 0
        while eps >err and it<its:
            self.act_gamma()
            self.A = self.act_A()
            self.pi = self.act_pi()
            h, v, M =  self.act_params(self.G,self.G,self.p,self.p)
            self.v = v
            self.h = h
            self.M = M
            likelinew = np.sum(-np.log(self.forback[0].ll))
            self.LogLtrain = likelinew
            self.act_probt()
            eps = likelinew-likeli
            if np.mod(it,10)==0:
                print("EM it: "+str(it)+", error: "+str(eps) + ", log-likelihood: "+str(likelinew))
            likeli = likelinew
            if eps >0 :
                tempa =self.A
                tempi =self.pi
                temph = self.h
                tempv = self.v
                tempM = self.M
            if plot == True:
                for gg in range(self.K):
                    for ff in range(gg+1,self.K):
                        self.scatter_pair_kl(gg,ff)
            it = it+1   
        self.A = tempa
        self.pi = tempi
        self.h = temph
        self.v = tempv
        self.M = tempM
        tock = time.time()
        print("EM it: "+str(it-1)+", error: "+str(eps) + ", log-likelihood: "+str(likelinew))
        print("EM optimization ended, it took: "+str(round(tock-tick,5))+"s or: " + str(round((tock-tick)/60.,5))+" min")
        
        
    def SEM(self,err1=9e-1,err2=9e-1,its1=1,its2=50,plot=False): 
        """
        Computes the SEM algorithm
        
        Parameters
        ----------
        err1 : TYPE, optional float
            DESCRIPTION. The default is 9e-1. error bound to end SEM algorithm
        err2 : TYPE, optional float
            DESCRIPTION. The default is 9e-1. error bound to end EM algorithm
        its1 : TYPE, optional int
            DESCRIPTION. The default is 1. Number of iterations of the SEM algorithm
        its2 : TYPE, optional int
            DESCRIPTION. The default is 50. Number of iterations of the EM algorithm
        plot : TYPE, optional bool
            DESCRIPTION. The default is False. plot samples from the learned model for each EM ireration (suitable for dimensions <4)

        Returns
        -------
        None.

        """
        if type(its1) is not int:
            raise Exception("The parameter its  must be an integer")
        if type(err1) is not float:
            raise Exception("The parameter err must be a float")
        if type(its2) is not int:
            raise Exception("The parameter its  must be an integer")
        if type(err2) is not float:
            raise Exception("The parameter err must be a float")
        eps = 1e100
        self.git = 0
        likelihood  = []
        print("SEM optimization...")
        tick = time.time()
        self.EM(its2,err2,plot)
        likelihood.append(self.LogLtrain)
        while eps > err1 and self.git < its1:
            self.hill_climb()
            self.git = self.git+1
            self.EM(its2,err2,plot=plot)
            eps = (self.LogLtrain-likelihood[-1])
            print("SEM it: " +str(self.git) +" error: " +str(eps))
            likelihood.append(self.LogLtrain)
        tock = time.time()
        print("SEM ended, it took: " +str(round(tock-tick,6))+"s or "+str(round((tock-tick)/60.,6))+"min")
    
    def viterbi(self,x,plot=True,xlabel="Time units",ylabel="index"): 
        """
        Performs the viterbi algorithm or hard segmentation.

        Parameters
        ----------
        x : TYPE numpy ndarray
            DESCRIPTION. input data to be segmentated
        plot : TYPE, optional bool
            DESCRIPTION. The default is True. Plots the segmentation
        xlabel : TYPE, optional str
            DESCRIPTION. The default is "Time units". x axis plot label
        ylabel : TYPE, optional str
            DESCRIPTION. The default is "index". y axis plot label

        Returns
        -------
        TYPE numpy array
            DESCRIPTION. The Viterbi path of the data

        """
        if type(x) is not np.ndarray:
            raise Exception("Input must be a numpy ndarray")
        if len(x.shape) !=2:
            raise Exception("Input numpy ndarray must be 2 dimensional")
        if len(x[0]) != self.K:
            raise Exception("Input number of columns does not match models dimensionality")
        T =len(x)
        fb = forBack()
        fb.prob_t(x,self.O,self.M,self.p,self.G,self.P,self.v,self.h,train=False)
        delta = np.log(self.pi)+np.log(fb.probt[0])
        deltat = [delta]
        psi = [np.zeros(len(self.pi))]
        for i in range(1,T-self.P):
            delta = self.viterbi_step(delta, fb, i)
            deltat.append(delta)
            psi.append(np.argmax(delta+np.log(self.A.T),axis=1))
            
        psi = np.flipud(np.array(psi)).astype(int)
        Q = [np.argmax(delta)]
        for t in np.arange(0,T-1-self.P):
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
        return np.max(delta+np.log(self.A.T),axis=1)+np.log(fb.probt[t])
    
    def log_likelihood(self,y,xunit=False):
        """
        Computes the log-likelihood of a dataset O.
        it uses the  current learned parameters of the model

        Parameters
        ----------
        y : TYPE Numpy array of size T'xK
            DESCRIPTION. Observations to be tested
        xunit : TYPE, optional bool
            DESCRIPTION. The default is False. it applies the BIC per unit of data
        Returns
        -------
        TYPE float
            DESCRIPTION. The score of the observations O

        """ 
        if type(y) is not np.ndarray:
            raise Exception("Input must be a numpy ndarray")
        if len(y.shape) !=2:
            raise Exception("Input numpy ndarray must be 2 dimensional")
        if len(y[0]) != self.K:
            raise Exception("Input number of columns does not match models dimensionality")
        fb = forBack()
        fb.prob_t(y,self.O,self.M,self.p,self.G,self.P,self.v,self.h,train=False)
        fb.forward_backward(self.A,self.pi)
        Clist = fb.ll
        log = np.sum(-np.log(Clist))
        del fb
        if xunit == False:
            return log
        else:
            return log/float(len(y))
        
        
    def sample_state(self,n,i):
        """
        Sample n instances from hidden state i

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. number of samples
        i : TYPE int
            DESCRIPTION. hidden state index

        Returns
        -------
        data : TYPE numpy ndarray
            DESCRIPTION. n samples from the distribution from hidden state i

        """
        if type(n) is not int:
            raise Exception("Number of samples n must be an integer")
        if type(i) is not int:
            raise Exception("Hidden state index i must be an integer")
        if  i<0 or i>=self.N:
            raise Exception("Hidden state index i does not exist")
        semi = self.O[:self.P]
        data = np.zeros([n,self.K])
        data = np.concatenate([semi,data],axis=0)
        vi = self.v[:,i]
        Gi = self.G[i]
        Li = self.L[i]
        Mi = self.M[i]
        macul = np.concatenate([[0],np.cumsum(vi)])
        zs = np.random.uniform(0,1,n) 
        for l in range(n):
            indexl = np.argmax(macul-zs[l]>0)-1
            yl = self.O[indexl+self.P][np.newaxis]
            datal = np.zeros([1,self.K])
            for k in Li:
                ulk = np.zeros([1,0])
                rlk = np.zeros([1,0])
                paik = np.argwhere(Gi[k]==1).T[0]
                pik =self.p[i][k]
                Mik = Mi[k]
                hik = self.h[i][k]
                if len(paik) != 0:
                    if len(paik) ==1:
                        ulk = np.concatenate([ulk,[datal[0,paik]]],axis=1)
                        rlk = np.concatenate([rlk,[yl[0,paik]]],axis=1)
                    else:
                        ulk = np.concatenate([ulk,datal[0,[paik]]],axis=1)
                        rlk = np.concatenate([rlk,[yl[0,paik]]],axis=1)
                        
                if pik !=0:
                    for j in range(pik):
                        ulkar = data[l+self.P-j,k]
                        rlkar = self.O[indexl+self.P-j,k]
                        rlk = np.concatenate([rlk,[[rlkar]]],axis=1)
                        ulk = np.concatenate([ulk,[[ulkar]]],axis=1)
                mean = yl[0][k] + np.dot(Mik,(ulk-rlk).T)
                datal[0,k] = np.random.normal(mean[0][0],hik,1)
            data[l] = datal
        return data
        
        
    def plot_densities_k(self,k,leng=500,nombre="",root = None):
        """
        Plots the densities of k for all the hidden states 
        
        Parameters
        ----------
        k : TYPE int
            DESCRIPTION. variable index 
        leng : TYPE, optional int
            DESCRIPTION. The default is 500. Number of smaples per scatter plot
        nombre : TYPE, optional str
            DESCRIPTION. The default is "". Name for the plot
        root : TYPE, optional str
            DESCRIPTION. The default is None. route to save plot

        Returns
        -------
        None.

        """
        if k<0 or k>= self.K:
            raise Exception("Variable index k not found")
        if type(leng) is not int:
            raise Exception("Leng must be a positive integer")
        dat = self.O
        y = dat[:,k]
        y_min = np.min(y)
        y_max = np.max(y)
        L = len(y)
        xline = np.linspace(y_min,y_max,leng)
        densidades = np.zeros([len(xline)-self.P,self.N])
        for i in range(self.N):
            vi = self.v[:,i]
            pa_ik = np.argwhere(self.G[i][k]==1).T[0]
            p_ik = self.p[i][k]
            Mik = self.M[i][k]
            hik = self.h[i][k]
            rt = (dat[self.P:,pa_ik]).reshape([L-self.P,len(pa_ik)])
            for j in range(1,p_ik+1):
                rt = np.concatenate([rt, (np.roll(y[:,k],j)[self.P:])[np.newaxis].T],axis=1)
            uline  = np.zeros([leng-self.P,0])
            for k in pa_ik:
                uk = self.O[:,k]
                uk_min = np.min(uk)
                uk_max = np.max(uk)
                ukline = np.linspace(uk_min,uk_max,leng-self.P)
                uline = np.concatenate([uline,ukline],axis=1)
            for j in range(1,p_ik+1):
                uline = np.concatenate([uline, (np.roll(xline[:,k],j)[self.P:])[np.newaxis].T],axis=1)
            for t in range(leng-self.P):
                mut = dat[self.P:,k] + np.dot(Mik,(uline[t]-rt).T)
                zt = (xline[t+self.P]- mut)/hik
                tomult = vi/hik*self.gaussian_kernel(zt).reshape([L-self.P])
                densidades[t:i] = np.sum(tomult)
                
        if nombre == None:
            nombre = "Densities by hidden states for vairable " + str(k)
            
        plt.figure(nombre)
        plt.clf()
        for i in range(self.N):
            plt.plot(xline[self.P:],densidades[:,i],label="state: "+str(i)+ " bandwidth:"+str(round(self.h.T[i][0],3)))
        plt.legend()
        plt.xlabel("$X$")
        plt.ylabel("$P$")
        plt.grid("on")
        plt.tight_layout()
        if root is not None:
            plt.savefig(root, dpi=600)
    
    
    def scatter_BN_ik(self,i,k,samples=5000,root=None):
        """
        

        Parameters
        ----------
        i : TYPE
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.
        samples : TYPE
            DESCRIPTION.
        root : TYPE, optional str
            DESCRIPTION. The default is None. route to save plot

        Returns
        -------
        None.

        """
        Gik = self.G[i][k]
        n_plots = np.sum(Gik)+self.p[i][k]
        n_rows = int(np.ceil(np.sqrt(n_plots)))
        parents = self.my_fathers(self.G[i], k)
        data = self.sample_state(samples,i)
        plt.figure("Scatter structure vairble "+ "$X_"+str(k)+"$ at hidden state "+ str(i))
        plt.clf()
        for n,pa in enumerate(parents):
            plt.subplot(n_rows,n_rows,n+1)
            plt.scatter(data[:,k],data[:,pa])
            plt.xlabel("$X^t_"+str(k)+"$")
            plt.ylabel("$X^t_"+str(pa)+"$")
            plt.grid("on")
        for w in range(1,self.p[i][k]+1):
            plt.subplot(n_rows,n_rows,int(np.sum(self.G[i][k]))+w+1)
            plt.scatter(data[w:,k],data[:-w,pa])
            plt.xlabel("$X^{t-"+str(w)+"}_"+str(k)+"$")
            plt.ylabel("$X^t_"+str(k)+"$")
            plt.grid("on")
        plt.tight_layout()
        if root is not None:
            plt.savefig(root, dpi=600)
            

    def scatter_pair_kl(self,k,l,samples=5000,name="",root =""):
        """
        Plot the scatter plot of the distribution from a pair of variables k,l 

        Parameters
        ----------
        k : TYPE int
            DESCRIPTION. Index of a variable
        l : TYPE int
            DESCRIPTION. Index of a variable
        samples : TYPE, optional int
            DESCRIPTION. The default is 5000. Number of samples
        name : TYPE, optional str
            DESCRIPTION. The default is "". Name of the plot
        root : TYPE, optional str 
            DESCRIPTION. The default is None. route to save plot

        Returns
        -------
        None.

        """
        if l<0 or l>= self.K:
            raise Exception("Variable index l not found")
        if k<0 or k>= self.K:
            raise Exception("Variable index k not found")
        if type(samples) is not int:
            raise Exception("Leng must be a positive integer")
        plt.figure("Scatter variable "+ "$X_"+str(k)+"$ and  variable "+ "$X_"+str(l)+name)
        plt.clf()
        plt.subplot(1,2,1)
        for i in range(self.N):
            data = self.sample_state(samples,i)
            plt.scatter(data[:,k],data[:,l],label ="state: " +str(i))
        plt.xlabel("$X^t_"+str(k)+"$")
        plt.ylabel("$X^t_"+str(l)+"$")
        plt.legend()
        plt.grid("on")
        plt.subplot(1,2,2)
        plt.scatter(self.O[:,k], self.O[:,l])
        plt.xlabel("$X^t_"+str(k)+"$")
        plt.ylabel("$X^t_"+str(l)+"$")
        plt.grid("on")
        plt.tight_layout()
        plt.pause(0.1)
        if root is not None:
            plt.savefig(root, dpi=600)
        
    def plot_all_pairs_scatter(self,samples=500,root =None,name =""):
        """
        Plot scatter plots from the learned parameters for all pairs of variables

        Parameters
        
        ----------
        samples : TYPE, optional int
            DESCRIPTION. The default is 500. Number of instances for each sample for each hidden state
        root : TYPE, optional str
            DESCRIPTION. The default is None. route to save plot
        name : TYPE, optional str
            DESCRIPTION. The default is "". Name of the plot
        Returns
        -------
        None.

        """
        if type(samples) is not int:
            raise Exception("Leng must be a positive integer")
        n_plots = self.K*(self.K-1)
        if n_plots<6:
            n_cols = n_plots
            n_rows = 1
        else:
            n_cols = 6
            n_rows = np.ceil(n_plots/6)
        plt.figure("Scatter plots "+name)
        plt.clf()
        for i in range(self.N):
            data = self.sample_state(samples,i)
            count = 1
            for gg in range(self.K):
                for ff in range(gg+1,self.K):
                    plt.subplot(n_rows,n_cols,count)
                    plt.scatter(data[:,gg],data[:,ff],label ="state: " +str(i))
                    plt.xlabel("$X^t_"+str(gg)+"$")
                    plt.ylabel("$X^t_"+str(ff)+"$")
                    plt.legend()
                    plt.grid("on")
                    plt.subplot(n_rows,n_cols,count+1)
                    plt.scatter(self.O[:,gg], self.O[:,ff],color="black")
                    plt.xlabel("$X^t_" + str(gg)  + "$")
                    plt.ylabel("$X^t_" + str(ff)  + "$")
                    plt.grid("on")
                    count +=2
        plt.tight_layout()
        if root is not None:
            plt.savefig(root, dpi=600)

    def plot_AR_k_scatter(self,k,samples=500,root=None):
        """
        Plot scatter plots from the learned parameters for all AR self.P values fro variable k

        Parameters
        ----------
        k : TYPE int
            DESCRIPTION. index of variable
        samples : TYPE, optional int
            DESCRIPTION. The default is 500. Number of instances per sample

        Returns
        -------
        None.

        """
        if k<0 or k>= self.K:
            raise Exception("Variable index k not found")
        if type(samples) is not int:
            raise Exception("Leng must be a positive integer")
        n_plots = 2*self.P
        if n_plots<6:
            n_cols = n_plots
            n_rows = 1
        else:
            n_cols = 6
            n_rows = np.ceil(n_plots/6)
        plt.figure("Scatter plots  ")
        plt.clf()
        for i in range(self.N):
            data = self.sample_state(samples,i)
            count = 1
            for p in range(1,self.P+1):
                plt.subplot(n_rows,n_cols,count)
                plt.scatter(data[:-p,k],data[p:,k],label ="state: " +str(i))
                plt.ylabel("$X^t_"+str(k)+"$")
                plt.xlabel("$X^{t-"+str(p)+"}_"+str(k)+"$")
                plt.legend()
                plt.grid("on")
                plt.subplot(n_rows,n_cols,count+1)
                plt.scatter(self.O[:-p,k], self.O[p:,k],color="black")
                plt.ylabel("$X^t_" + str(k)  + "$")
                plt.xlabel("$X^{t-"+str(p)+"}_" + str(k)  + "$")
                plt.grid("on")
                count +=2
        plt.tight_layout()
        if root is not None:
            plt.savefig(root, dpi=600)


    def save(self,root=None, name = "kde-ashmm_"+str(datetime.datetime.now())[:10]):
        """
        Saves the current model

        Parameters
        ----------
        root : TYPE, optional str
            DESCRIPTION. The default is None. direction in pc to save the model
        name : TYPE, optional str
            DESCRIPTION. The default is "hmm_"+str(datetime.datetime.now())[:10].Name of the saved file
        """
        if root == None:
            root = os.getcwd()
        try:
            os.mkdir(root)
        except:
            pass
        itemlist = [self.N,self.K,self.P,self.length,self.O,self.A,self.pi,self.M,self.h,self.v,self.p
                    ,self.G,self.L]
        with open(root+"\\"+name+".kdehmm", 'wb') as fp:
            pickle.dump(itemlist, fp)
    
    def load(self,rootr):
        """
        Loads a model, must be an kdehmm file 

        Parameters
        ----------
        rootr : TYPE str
            DESCRIPTION. direction of the saved model

        Returns
        -------
        str
            DESCRIPTION. in case of  wrong filename, returns error

        """
        if rootr[-7:] != ".kdehmm":
            raise Exception("The file is not an ashmm file")
        with open(rootr, "rb") as  fp:
            loaded = pickle.load(fp)
        self.N          = loaded[0]
        self.K          = loaded[1]
        self.P          = loaded[2]
        self.length     = loaded[3]
        self.O          = loaded[4]
        self.A          = loaded[5]
        self.pi         = loaded[6]
        self.M          = loaded[7]
        self.h          = loaded[8]
        self.v          = loaded[9]
        self.p          = loaded[10]
        self.G          = loaded[11]
        self.L          = loaded[12]
        
    def copy(self,model):
        """
        Copy the parameters of another KDE-AsHMM model

        Parameters
        ----------
        model : TYPE KDE-AsHMM
            DESCRIPTION. a KDE-AsHMM object

        Returns
        -------
        None.

        """
        if type(model) is not KDE_AsHMM:
            raise Exception("Model is not a KDE_AsHMM object")
        self.N          = model.N
        self.K          = model.K
        self.P          = model.P
        self.length     = model.length
        self.O          = np.copy(model.O)
        self.A          = np.copy(model.A)
        self.pi         = np.copy(model.pi)
        self.M          = []
        for i in range(self.N):
            Mi = []
            for k in range(self.K):
                Mi.append(np.copy(model.M[i][k]))
            self.M.append(Mi)
        self.h          = np.copy(model.h)
        self.v          = np.copy(model.v)
        self.p          = np.copy(model.p)
        self.G          = np.copy(model.G)
        self.L          = np.copy(model.L)

            
class forBack:
    
    def __init__(self):
        """
        Creates a forBack object, useful to deal with probabilities and latent probabilities 
        
        Returns
        -------
        None.

        """
        self.alpha = None
        self.beta  = None
        self.gamma = None
        self.rho   = None
        self.ll    = None
        self.probt = None
        self.numh  = None
        self.denh  = None
        self.numw  = None
        self.denw  = None
        self.mnum  = None
        self.mden  = None
        
        
    def gaussian_kernel(self,x):
        """
        Computes the exponential kernel for the given argument x

        Parameters
        ----------
        x : TYPE numpy ndarray
            DESCRIPTION. argument for the Gaussian kernel

        Returns 
        -------
        TYPE numpy ndarray
            DESCRIPTION. The Gaussian kernel evaluation for the given input

        """
        return  np.exp(-0.5*x**2)/(np.sqrt(2*np.pi)) 
   
  
    def prob_t(self,x,y,M,p,G,P,v,h,train=True):
        """
        Computes the probabililty of the input sequence x for each of the N hidden states,
        given the parameters M, p, G, P, v, h and the training dataset y

        Parameters
        ----------
        x : TYPE numpy ndarray
            DESCRIPTION. testing dataset
        y : TYPE numpy ndarray
            DESCRIPTION. training dataset
        M : TYPE list
            DESCRIPTION. list with the list of coefficients
        p : TYPE numpy ndarray
            DESCRIPTION. array with the AR order for each variable for each hidden state
        G : TYPE list
            DESCRIPTION. list of matrices with the graphs
        P : TYPE int
            DESCRIPTION. maximum lag
        v : TYPE numpy ndarray
            DESCRIPTION. instance relevancy for each kernel
        h : TYPE numpy ndarray
            DESCRIPTION. kernel bandwidths
        train : TYPE, optional bool
            DESCRIPTION. The default is True. if training is being performed

        Returns
        -------
        None.

        """
        N = len(p)
        K = len(x[0])
        T = len(x)
        L = len(y)
        data   =  np.zeros([T-P,N])
        psi = np.zeros([T-P,N,L-P])
        for i in range(N):
            kern = np.ones([T-P,L-P])
            vi = v[:,i]
            for k in range(K):
                diff = x[P:,k][np.newaxis].T- y[P:,k][np.newaxis]
                pa_ik  = np.argwhere(G[i][k]==1).T[0]
                p_ik   = p[i][k]
                Mik    = M[i][k]
                hik    = h[i][k]
                if len(Mik[0])>0:
                    ut = np.zeros([T-P,0])
                    rt = np.zeros([L-P,0])
                    if len(pa_ik>0):
                        utx     = (x[P:,pa_ik]).reshape([T-P,len(pa_ik)])
                        ut = np.concatenate([ut,utx],axis=1)
                        rtx     = (y[P:,pa_ik]).reshape([L-P,len(pa_ik)])
                        rt = np.concatenate([rt,rtx],axis=1)
                    for j in range(1,p_ik+1):
                        ut = np.concatenate([ut, (np.roll(x[:,k],j)[P:])[np.newaxis].T],axis=1)
                        rt = np.concatenate([rt, (np.roll(y[:,k],j)[P:])[np.newaxis].T],axis=1)
                    mu = np.zeros([T-P,L-P])
                    mu = np.dot(ut.reshape([T-P,1,len(Mik[0])])-rt.reshape([1,L-P,len(Mik[0])]),Mik.T).reshape([T-P,L-P])
                    mu = (diff- mu)/hik
                else:
                    mu = diff/hik
                tomult = 1./hik*self.gaussian_kernel(mu)
                kern = kern*tomult
            arg = kern*vi.reshape([1,L-P])
            for t in range(T-P):
                arg[t] = self.checkzero(arg[t])
            if train == True:
                np.fill_diagonal(arg,0)
            data[:,i] = np.sum(arg,axis=1)
            psi[:,i,:] = arg/(data[:,i].reshape([T-P,1]))
        self.probt   = data
        self.psi     = psi

        
    def score_i_k(self,x,y,M,p,G,P,v,h,i,k,train=True):
        """
        Score used to compare strcutures duering the SEM algorithm

        Parameters
        ----------
        x : TYPE numpy ndarray
            DESCRIPTION. testing dataset
        y : TYPE numpy ndarray
            DESCRIPTION. training dataset
        M : TYPE list
            DESCRIPTION. list with the list of coefficients
        p : TYPE numpy ndarray
            DESCRIPTION. array with the AR order for each variable for each hidden state
        G : TYPE list
            DESCRIPTION. list of matrices with the graphs
        P : TYPE int
            DESCRIPTION. maximum lag
        v : TYPE numpy ndarray
            DESCRIPTION. instance relevancy for each kernel
        h : TYPE numpy ndarray
            DESCRIPTION. kernel bandwidths
        i : TYPE int
            DESCRIPTION. hidden state index 
        k : TYPE int
            DESCRIPTION. variable index
        train : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE int
            DESCRIPTION. score of the current structure

        """
        dif = x[P:,k][np.newaxis].T- y[P:,k][np.newaxis]
        T = len(x)
        L = len(y)
        pa_ik = np.argwhere(G[i][k]==1).T[0]
        p_ik = p[i][k]
        Mik = M[i][k]
        hik = h[i][k]
        len_nm = len(pa_ik)+p_ik
        if len(Mik[0])> 0:
            ut = (x[P:,pa_ik]).reshape([T-P,len(pa_ik)])
            rt = (y[P:,pa_ik]).reshape([L-P,len(pa_ik)])
            for j in range(1,p_ik+1):
                ut = np.concatenate([ut, (np.roll(x[:,k],j)[P:])[np.newaxis].T],axis=1)
                rt = np.concatenate([rt, (np.roll(y[:,k],j)[P:])[np.newaxis].T],axis=1)
            mu = np.dot(ut.reshape([T-P,1,len_nm])-rt.reshape([1,L-P,len_nm,]),Mik.T).reshape([T-P,L-P])
            mu = (dif- mu)/hik
        else:
            mu = dif/hik
        mu = -0.5*(mu**2+np.log(2*np.pi))
        score =  np.sum((self.psi[:,i,:])*mu)  
        return score
        
                
        
    def forward_backward(self,A,pi):    
        """
        Performs the forward-backward algorithm 

        Parameters
        ----------
        A : TYPE numpy ndarray
            DESCRIPTION. Transition matrix
        pi : TYPE numpy ndarray
            DESCRIPTION. initial distribution
        Returns
        -------
        None.

        """
        T = len(self.probt)
        alfa = pi*self.probt[0]
        Cd = 1./np.sum(alfa)
        Clist = np.array([Cd])
        alfa = alfa*Cd
        ALFA = [alfa]
        for t in range(1,T):
            alfa = self.forward_step(alfa,A,t)
            Cd = 1./np.sum(alfa)            
            Clist = np.concatenate([Clist,[Cd]])
            alfa = Cd*alfa
            ALFA.append(alfa)
        ALFA= np.array(ALFA)
        self.alpha = ALFA
        
        beta =np.ones(len(pi))
        Clist =Clist[::-1]
        beta = beta*Clist[0]
        BETA = [beta]
        for t in range(1,T):
            beta = self.backward_step(beta,A,T-t)
            beta = beta*Clist[t]
            BETA.append(beta)
        BETA= np.flipud(np.array(BETA))
        self.beta = BETA
        self.ll = Clist
        self.gamma = self.alpha*self.beta/np.sum(self.alpha*self.beta,axis=1)[np.newaxis].T
        self.psi = self.psi*self.gamma.T[np.newaxis].T
        
    def checkzero(self,z):
        """
        Returns a modified vector z with no zero instances
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)
        y[inds] = np.exp(-740)
        return y
    
    def forward_step(self,alfa,A,t):
        """
        Does a forward step

        Parameters
        ----------
        alfa : TYPE numpy ndarray
            DESCRIPTION. current alfa values
        A : TYPE numpy ndarray
            DESCRIPTION. transition matrix
        t : TYPE int
            DESCRIPTION. time instance

        Returns
        -------
        TYPE numpy ndarray
            DESCRIPTION. alfa at time step t

        """
        arg = np.dot(alfa,A)
        return self.probt[t]*arg


    def backward_step(self,beta,A,t): 
        """
        Does a backward step

        Parameters
        ----------
        beta : TYPE numpy ndarray
            DESCRIPTION. Current beta values
        A : TYPE numpy ndarray
            DESCRIPTION. transition matrix
        t : TYPE int
            DESCRIPTION. time instance

        Returns
        -------
        arg : TYPE
            DESCRIPTION.

        """
        arg = np.dot(A,self.probt[t]*beta)
        return arg
    
    def act_aij(self,A,N,): 
        """
        Updates the transition matrix A

        Parameters
        ----------
        A : TYPE numpy ndarray
            DESCRIPTION. Transition matrix
        N : TYPE int
            DESCRIPTION. number of hidden states
        O : TYPE
            DESCRIPTION.
        P : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        T = len(self.probt)
        bj = self.probt
        nume = []
        deno = []
        for i in range(N):
            alfat = (self.alpha.T[i])[:T-1] 
            betat = self.beta[1:].T 
            num = A[i]*np.sum(alfat*betat*bj[1:].T,axis=1)
            den = np.sum(num)
            nume.append(num)
            deno.append(den)
        self.numa = np.array(nume)
        self.dena = np.array(deno)
        
        # opcional
        
        for i in range(len(self.dena)):
            if self.dena[i] == 0:
                self.dena[i] = np.exp(-740)
                
    def update_params(self,x,curr_G,G,curr_p,p,P,M,h):
        """
        Perform the M step to update the v, h and M parameters

        Parameters
        ----------
        x : TYPE numpy ndarray
            DESCRIPTION. training data
        curr_G : TYPE list of matrices
            DESCRIPTION. Current G matrices
        G : TYPE list of matrices
            DESCRIPTION. Tested G matrices, equal to curr_G in EM
        curr_p : TYPE numpy ndarray
            DESCRIPTION. Current AR orders
        p : TYPE numpy ndarray
            DESCRIPTION. Tested AR orders, equal to curr:p in EM
        P : TYPE int
            DESCRIPTION. Maximum lag
        M : TYPE list
            DESCRIPTION. List of the list of the parameters
        h : TYPE numpy ndarray
            DESCRIPTION. Bandwidths

        Returns
        -------
        None.

        """
        # Constantes a usar
        T = len(x)
        K = len(x[0])
        N = len(p)
        
        # Donde se guarda la info
        numh = np.zeros([N,K])
        numm = []
        denm = []
        for i in range(N):
            nummi = []
            denmi = []
            for k in range(K):
                lpa = np.sum((G[i][k]))+p[i][k]
                numik = np.zeros([1,int(lpa)])
                denmik = np.zeros([int(lpa),int(lpa)])
                nummi.append(numik)
                denmi.append(denmik)
            numm.append(nummi)
            denm.append(denmi)
        numw = np.zeros([T-P,N])
        
        #computando cosas :v 
        for i in range(N):
            psil = self.psi[:,i]
            numw[:,i] = np.sum(self.psi[:,i,:],axis=0)
            for k in range(K):
                diffx = x[P:,k][np.newaxis].T- x[P:,k][np.newaxis]
                pa_ik = np.argwhere(G[i][k]==1).T[0]
                curr_pa_ik = np.argwhere(curr_G[i][k]==1).T[0]
                p_ik = p[i][k]
                curr_p_ik = curr_p[i][k]
                Mik = M[i][k]
                ut = (x[P:,pa_ik]).reshape([T-P,len(pa_ik)])
                cut = (x[P:,curr_pa_ik]).reshape([T-P,len(curr_pa_ik)])
                for j in range(1,p_ik+1):
                    ut = np.concatenate([ut, (np.roll(x[:,k],j)[P:])[np.newaxis].T],axis=1)
                for j in range(1,curr_p_ik+1):
                    cut = np.concatenate([cut, (np.roll(x[:,k],j)[P:])[np.newaxis].T],axis=1)
                                        
                len_nm = int(len(pa_ik)+p_ik)
                len_cm = int(len(curr_pa_ik)+curr_p_ik)

                ubar = ut.reshape([T-P,1,len_nm])-ut.reshape([1,T-P,len_nm])
                cut_bar = cut.reshape([T-P,1,len_cm])-cut.reshape([1,T-P,len_cm])

                ubarx = ubar*(diffx.reshape([T-P,T-P,1]))
                mut = np.dot(cut_bar,Mik.T).reshape([T-P,T-P])

                ubar2 = np.zeros([T-P,T-P,len_nm,len_nm])   
                for l in range(len_nm):
                    ubar2[:,:,l] = ubar*(ubar[:,:,l]).reshape([T-P,T-P,1])
                
                if p[i][k] != 0 or np.sum(G[i][k]) !=0 :
                    denm[i][k] = np.sum(np.sum(ubar2*psil.reshape([T-P,T-P,1,1]),axis=0),axis=0)
                    numm[i][k] = np.sum(np.sum(ubarx*psil.reshape([T-P,T-P,1]),axis=0),axis=0)
                numh[i][k] = np.sum((diffx-mut)**2*psil)
        self.denw = np.sum(self.gamma,axis=0)
        self.denh = self.denw
        self.numm = numm
        self.denm = denm
        self.numh = numh
        self.numw = numw
        
        

        
        
        
        