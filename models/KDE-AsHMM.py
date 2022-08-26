# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:19:45 2022

@author: fox_e
"""
import pickle
import datetime
import time
import os
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from AR_ASLG_HMM  import AR_ASLG_HMM  as hmm
from scipy.special import logsumexp
from scipy.spatial import distance_matrix as distmat
# import ctypes 
# root = os.getcwd()
# so_file = root+"/aux_k.so"
# fun =  ctypes.CDLL(so_file)
# print(type(fun))
# print(fun)
# return_val = fun.suma(16,14)
# print(return_val)


class KDE_AsHMM:    
    def __init__(self,O,lengths,N, A=None, pi=None, h= None, M=None, v=None, 
                  p=None, P=None, G=None, sigma=None,p_cor=5,struc=True,lags=True):
        self.O = O
        self.N = N
        self.lengths = lengths
        self.struc   = struc    
        self.lags    = lags    
        self.K       = len(O[0]) 
        self.git     = 0
        self.min_h = np.zeros(self.K)
        
        

        if P is None:
            self.P = self.det_lags(p_cor=p_cor)
        else:
            self.P = P
        if v is None or A is None:
            aux_model = hmm(O,lengths,N,P=self.P)
            aux_model.EM()
            
        self.forback = []
        self.C = None
        if h is None:
            self.h = np.ones([self.N,self.K])*(4*np.std(O,axis=0)**5/(3*np.sum(lengths)))**(1./5.)
        else:
            self.h = h
        if p is None:
            if lags == True:
                self.p = np.zeros([N,self.K]).astype(int)
            if lags == False:
                self.p = (self.P*np.ones([N,self.K])).astype(int)
        else:
            self.p = p
        if  A is None:
            Ap = aux_model.A
            self.A = Ap
        else:
            self.A = A
        if pi is None:
            self.pi = np.ones(N)/N
        else:
            self.pi= pi
        if G is None:
            G = self.prior_G(N,self.K)
            L= []
            for i in range(N):
                L.append(np.arange(self.K))
            self.G = G
            self.L = L
        else:
            self.G = G
            self.L = []
            if len(G) != self.N:
                print( "Invalid G")
            else:
                for i in range(self.N):
                    if np.prod(G[i].shape) != self.K**2 or len(G[i]) != self.K or  len(G[i].T) != self.K  :
                        print("Invalid G")
            for i in range(self.N):
                booli, Li = self.dag_v(G[i])
                if booli == True:
                    self.L.append(Li)
                else:
                   print("G at state "+ str(i) + " is not a DAG")
                
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
        if v is None:
            # self.v = self.init_v(self.O, self.N, self.h[0], self.P)
            # vv = aux_model.forBack[0].gamma
            # vv = vv/np.sum(vv,axis=0)
            # self.v  = vv
            vv = np.random.uniform(0.01,0.99,[np.sum(lengths)-self.P,3])
            self.v = vv/np.sum(vv,axis=0)
        else:
            self.v = v
                    
        print("Priors generated...")
        
    def gaussian_kernel(self,x):
        return  np.exp(-0.5*x**2)/(np.sqrt(2*np.pi)) 
   
    def init_v(self,x,N,h,P):
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
        Plots all the learned graphs

        Parameters
        ----------
        route : TYPE, optional str
            DESCRIPTION. The default is None. direction where to save the graphs
        label : TYPE, optional list
            DESCRIPTION. The default is None. labels for the variables. It must be of size K
        coef : TYPE, optional  bool
            DESCRIPTION. The default is False. Adds the weights of the arcs to the plots

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
        Looks for the best structure in AR components. 
        Uses a greedy search

        Parameters
        ----------
        tb : TYPE boolean list 
            DESCRIPTION. indices to be updated of the parameter B
        ts : TYPE boolean list
            DESCRIPTION. indices to be updated of the parameter sigma
        """
        # tick = time.time()
        # print("Performing AR optimization")
        for i in range(self.N):
            for k in range(self.K):
                sm = 0
                aux_len = np.concatenate([[0],self.lengths.cumsum()])
                for j in range(len(self.forback)):
                    pena= 0.5*(np.sum(self.G[i][k])+self.p[i][k])*(self.lengths[j])
                    sm+= self.forback[j].score_i_k(self.O[aux_len[j]:aux_len[j+1]],self.O[aux_len[j]:aux_len[j+1]],self.M,self.p,self.G,self.P,self.v,self.h,i,k)-pena
                    # print(str(i)+":"+str(k)+", current score: " +str(sm))
                while self.p[i][k] +1 <= self.P :
                    p2 = np.copy(self.p)
                    p2[i][k] = p2[i][k] +1
                    [h2,v2,M2] = self.act_params(self.G,self.G,self.p,p2)
                    s2 = 0
                    aux_len = np.concatenate([[0],self.lengths.cumsum()])
                    for j in range(len(self.forback)):
                        pena= 0.5*(np.sum(self.G[i][k])+p2[i][k])*(self.lengths[j])
                        s2+= self.forback[j].score_i_k(self.O[aux_len[j]:aux_len[j+1]],self.O[aux_len[j]:aux_len[j+1]],M2,p2,self.G,self.P,self.v,self.h,i,k)-pena
                        # print(str(i)+":"+str(k)+", Score to compare: "+str(s2))
                    if s2 > sm:
                        self.p = p2
                        self.M = M2
                        sm  = s2
                        # print(str(i)+":"+str(k)+", current score: " +str(sm))
                    else:
                        break
        # tock = time.time()
        # print("AR optimization ended, took: "+str(round(tock-tick,6))+"s")
        
                

    
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
        Looks for the best graphical structure, uses an upward greedy algorithm

        Parameters
        ----------
        tb : TYPE boolean list 
            DESCRIPTION. indices to be updated of the parameter B
        ts : TYPE boolean list
            DESCRIPTION. indices to be updated of the parameter sigma
        """
        # tick = time.time()
        # print("Performing BN optimization")
        for i in range(self.N):
            for k in range(self.K):
                possi = self.pos_ads(self.G[i])
                son = possi[k][0]
                if len(possi[k][1])!=0:
                    sm = 0
                    aux_len = np.concatenate([[0],self.lengths.cumsum()])
                    for r in range(len(self.forback)):
                        pena= 0.5*(np.sum(self.G[i][k])+self.p[i][k])*(self.lengths[r])
                        sm+= self.forback[r].score_i_k(self.O[aux_len[r]:aux_len[r+1]],self.O[aux_len[r]:aux_len[r+1]],self.M,self.p,self.G,self.P,self.v,self.h,i,k)-pena
                    for j in possi[k][1]:
                        G2 = np.copy(self.G)
                        G2[i][son][j] =1
                        L2 = []
                        for nn in range(self.N):
                            L2.append(self.dag_v(G2[nn])[1])
                        [h2,v2,M2] = self.act_params(self.G,G2,self.p,self.p)
                        s2 = 0
                        aux_len = np.concatenate([[0],self.lengths.cumsum()])
                        for l in range(len(self.forback)):
                            pena= 0.5*(np.sum(G2[i][k])+self.p[i][k])*(self.lengths[l])
                            s2 =+ self.forback[l].score_i_k(self.O[aux_len[l]:aux_len[l+1]],self.O[aux_len[l]:aux_len[l+1]],M2,self.p,G2,self.P,self.v,self.h,i,k)-pena
                        if s2>sm: 
                            sm= s2
                            self.M = M2
                            self.G = G2
                            self.L = L2
                        else:
                            break
        # tock = time.time()
        # print("BN optimization ended, took: "+str(round(tock-tick,6))+"s")

                    
               
    def hill_climb(self):
        """
        Executes a greedy forward algorithm to explore the Bayesian networks space.

        Parameters
        ----------
        tb : TYPE boolean list 
            DESCRIPTION. indices to be updated of the parameter B
        ts : TYPE boolean list
            DESCRIPTION. indices to be updated of the parameter sigma
        """
        if self.lags is True:
            self.climb_ar()
        if self.struc is True:
            self.climb_struc()
    

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
            self.forback[i].prob_t(self.O[aux_len[i]:aux_len[i+1]],self.O[aux_len[i]:aux_len[i+1]],self.M,self.p,self.G,self.P,self.v,self.h)



    def act_gamma(self):
        """
        Updates for each forBack object its gamma parameter or latent probabilities

        Parameters
        ---------- x,G,p,P,h,v
        ps : TYPE, optional int
            DESCRIPTION. The default is None. index of the initial distibution
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forback[i].forward_backward(self.A,self.pi,self.O[aux_len[i]:aux_len[i+1]],self.P)
            
    def act_params(self,curr_G,G,curr_p,p,method=0):
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forback[0].update_params(self.O[aux_len[0]:aux_len[1]],curr_G,G,curr_p,p,self.P,self.M,self.h)
        numh =self.forback[0].numh
        denh =self.forback[0].denh
        numw =self.forback[0].numw
        denw =self.forback[0].denw
        numm = self.forback[0].numm
        denm = self.forback[0].denm
        for i in range(1,len(self.lengths)):
            self.forback[i].updata_params(self.y,self.O[aux_len[i]:aux_len[i+1]],curr_G,G,curr_p,p,self.P,self.M,self.h)
            numh += self.forback[i].numh
            denh += self.forback[i].denh
            numw += self.forback[i].numw
            denw += self.forback[i].denw
            numm += self.forback[i].numm
            denm += self.forback[i].denm
            
        h = np.sqrt(numh/denh[np.newaxis].T)
        w = numw/denw
        
        M = []
        for i in range(self.N):
            Mi = []
            for k in range(self.K):
                if np.sum(G[i][k]) != 0 or p[i][k]!=0:
                    Mik = np.linalg.solve(denm[i][k],numm[i][k].T).T
                    Mi.append(Mik.reshape([1,len(Mik)]))
                    # Mi.append(Mik)
                else:
                    Mik = self.M[i][k]
                    Mi.append(Mik)
            M.append(Mi)
            
        # for i in range(self.N):
        #     for k in range(self.K):
        #         if h[i][k] <self.min_h[k]:
        #             h[i][k] = self.min_h[k]
                    
            
        return [h,w,M]
    
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
        self.forback[0].act_aij(self.A,self.N,self.O[aux_len[0]:aux_len[1]],self.P)
        numa = self.forback[0].numa
        dena = self.forback[0].dena
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_aij(self.A,self.N,self.O[aux_len[i]:aux_len[i+1]],self.P)
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
    

    
    def EM(self,its=50,err=9e-1,plot=False):
        print("EM optimization...")
        tick = time.time()
        tempa =self.A
        tempi =self.pi
        temph = self.h
        tempv = self.v
        tempM = self.M
        self.forback = []
        for l in range(len(self.lengths)):
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
            # if it == 0 and self.git==0:
            self.h = h
            self.M = M
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-np.log(self.forback[i].ll))
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
#            print likeli
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
        Does the SEM algorithm for parameter and structure learning

        Parameters
        ----------
        err1 : TYPE, optional float
            DESCRIPTION. The default is 1e-2. Maximum SEM error allowed
        err2 : TYPE, optional float
            DESCRIPTION. The default is 1e-2. Maximum EM error allowed
        its1 : TYPE, optional int
            DESCRIPTION. The default is 100. Maximum SEM iterations
        its2 : TYPE, optional int
            DESCRIPTION. The default is 100. Maximum EM iterations 
        ta : TYPE, optional  boolean list
            DESCRIPTION. The default is "all".  list that gives the indices of rows and columns of A to be updated.
        tpi : TYPE, optional bool
            DESCRIPTION. The default is True. Updates or not the initial distribution
        tb : TYPE, optional boolean list
            DESCRIPTION. The default is "all". indices to be updated of the parameter B
        ts : TYPE, optional boolean list
            DESCRIPTION. The default is "all". list that gives the indices of which hidden-states variances are updated 
        ps : TYPE, optional int
            DESCRIPTION. The default is None. index of the initial distibution
        """
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
        T =len(x)
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        fb = forBack()
        fb.prob_t(x,self.O[aux_len[0]:aux_len[1]],self.M,self.p,self.G,self.P,self.v,self.h,train=False)
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
    
    def log_likelihood(self,y,xunit=False,ps = None):
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
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        fb.prob_t(y,self.O[aux_len[0]:aux_len[1]],self.M,self.p,self.G,self.P,self.v,self.h,train=False)
        fb.forward_backward(self.A,self.pi,y,self.P)
        Clist = fb.ll
        log = np.sum(-np.log(Clist))
        del fb
        if xunit == False:
            return log
        else:
            return log/float(len(y))
        
        
    def sample_state(self,n,i,root=None):
        if root is None:
            root = self.O[:self.P]
        data = np.zeros([n,self.K])
        data = np.concatenate([root,data],axis=0)
        vi = self.v[:,i]
        Gi = self.G[i]
        Li = self.L[i]
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
                Mik = self.M[i][k]
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
        
        
    def plot_densities_k(self,k,leng=500,nombre=""):
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        dat = self.O[aux_len[0]:aux_len[1]]
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
                uk = self.O[aux_len[0]:aux_len[1]][:,k]
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
    
    
    def gaussian_kernel(self,x):
        return  np.exp(-0.5*x**2)/(np.sqrt(2*np.pi)) 
    
    
    def scatter_BN_ik(self,i,k,samples=5000):
        """
        

        Parameters
        ----------
        i : TYPE
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.
        samples : TYPE
            DESCRIPTION.

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
            

    def scatter_pair_kl(self,k,l,samples=5000,name=""):
        """
        

        Parameters
        ----------
        i : TYPE
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.
        samples : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
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
        
    def plot_all_pairs_scatter(self,samples=500):
        
        n_plots = self.K*(self.K-1)
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

    def plot_AR_k_scatter(self,k,samples=500):
        
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
                plt.scatter(data[p:,k],data[:-p,k],label ="state: " +str(i))
                plt.xlabel("$X^t_"+str(k)+"$")
                plt.ylabel("$X^{t-"+str(p)+"}_"+str(k)+"$")
                plt.legend()
                plt.grid("on")
                plt.subplot(n_rows,n_cols,count+1)
                plt.scatter(self.O[p:,k], self.O[:-p,k],color="black")
                plt.xlabel("$X^t_" + str(k)  + "$")
                plt.ylabel("$X^{t-"+str(p)+"}_" + str(k)  + "$")
                plt.grid("on")
                count +=2
        plt.tight_layout()


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
        itemlist = [self.N,self.K,self.P,self.A,self.pi,self.B,self.sigma,self.p
                    ,self.G,self.L,self.mus,self.covs,self.dictionary,
                    self.b,self.bic,self.LogLtrain]
        if self.kappa is not None:
            itemlist.append(self.nu)
            itemlist.append(self.kappa)
        with open(root+"\\"+name+".kdehmm", 'wb') as fp:
            pickle.dump(itemlist, fp)
    
    def load(self,rootr):
        """
        Loads a model, must be an ashmm file 

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
            return "The file is not an ashmm file"
        with open(rootr, "rb") as  fp:
            loaded = pickle.load(fp)
        self.N          = loaded[0]
        self.K          = loaded[1]
        self.P          = loaded[2]
        self.A          = loaded[3]
        self.pi         = loaded[4]
        self.B          = loaded[5]
        self.sigma      = loaded[6]
        self.p          = loaded[7]
        self.G          = loaded[8]
        self.L          = loaded[9]
        self.mus        = loaded[10]
        self.covs       = loaded[11]
        self.dictionary = loaded[12]
        self.b          = loaded[13]
        self.bic        = loaded[14]
        self.LogLtrain  = loaded[15]
        if len(loaded)>16:
            self.nu     = loaded[16]
            self.kappa  = loaded[17]
        
    
class forBack:
    
    def __init__(self):
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
        return  np.exp(-0.5*x**2)/(np.sqrt(2*np.pi)) 
   
  
    def prob_t(self,x,y,M,p,G,P,v,h,train=True):
        """
        Computes the probabililty of the input sequence x for each of the N hidden states,
        given the parameters M, p, G, P, v, h and the training dataset y

        Parameters
        ----------
        x : TYPE 
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        M : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        G : TYPE
            DESCRIPTION.
        P : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.
        h : TYPE
            DESCRIPTION.
        train : TYPE, optional
            DESCRIPTION. The default is True.

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
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        M : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        G : TYPE
            DESCRIPTION.
        P : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.
        h : TYPE
            DESCRIPTION.
        i : TYPE
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.
        train : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

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
        
                
        
    def forward_backward(self,A,pi,O,P,ps=None): 
        """
        Does the scaled forward-backward algorithm using logarithms

        Parameters check AR_ASLG_HMM for more information 
        ----------
        ps : TYPE, optional int
            DESCRIPTION. The default is None. index of the initial distibution
        """
        T = len(O)
        if ps == None:
            alfa = pi*self.probt[0]
        else:
            alfa = A[ps]*self.probt[0]
        Cd = 1./np.sum(alfa)
        Clist = np.array([Cd])
        alfa = alfa*Cd
        ALFA = [alfa]
        for t in range(1,T-P):
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
        for t in range(1,T-P):
            beta = self.backward_step(beta,A,T-t-P)
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
            arg = np.dot(alfa,A)
            return self.probt[t]*arg
        else:
            logaa = np.dot(alfa,A)
            return prob[t]*logaa
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
        arg = np.dot(A,self.probt[t]*beta)
        return arg
    
    def act_aij(self,A,N,O,P): 
        """
        Updates the parameter A

        Parameters check AR_ASLG_HMM for more information 
        ----------
        """
        T = len(O)-P
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
        
        
#%%Functions to generate data

def dag_v(G):
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

def generate_2D(mean,sigma,k,roots,n,ar_coef):
    x1 = []
    lar = len(ar_coef)
    for t in range(n):
        if t <lar:
            x1t = np.random.normal(mean,sigma,1)
        else:
            x1t = np.random.normal(mean+k*np.sin(2*np.pi*0.5*(np.dot(ar_coef[::-1],x1[-lar:]))),sigma,1)
            # x1t = np.random.normal(mean+np.dot(ar_coef[::-1],x1[-lar:]),sigma,1)
            # x1t = np.random.normal(mean,sigma,1)
        x1.append(x1t)
    x1 = np.array(x1)
    x= []
    for i in range(len(x1)):
        mean = k
        for l in roots:
            mean *= (x1[i]-l)
        x.append(np.random.normal(mean,1,1))
    x = np.array(x)
    data = np.concatenate([x,x1],axis=1)
    return data

def generate_2D_seq(means,sigmas,ks,rootss,ns,ar_coef,seq):
    data = np.zeros([0,2])
    for i in seq:
        di = generate_2D(means[i],sigmas[i],ks[i],rootss[i],ns[i],ar_coef[i])
        data = np.concatenate([data,di],axis=0)
    return data

def gen_nl_random_1sample(G,L,P,p,M,means,sigma,f1,f2,k,seed):
    K = len(G)
    xt = np.ones([1,K])
    for i in L:
        if np.sum(M[i]) ==0 :
            xt[0][i] = np.random.normal(means[i],sigma[i],1)
        else:
            mean  = means[i]
            for w in range(K):
                mean += f1(M[i][w]*xt[0][w])
            for l in range(P):
                ki = k[i]
                f2m = M[i][K+l]*f2(seed[-l-1][i])
                mean += ki*f2m
            xt[0][i] = np.random.normal(mean,sigma[i],1)
    return xt
            
def gen_nl_random_l(ncols,nrows,G,L,P,p,M,means,sigma,f1,f2,k):
    K = ncols 
    x = np.ones([P,K])*means[np.newaxis]
    for t in range(nrows):
        xt = gen_nl_random_1sample(G, L, P, p, M, means, sigma, f1,f2,k, x[-P:])
        x = np.concatenate([x,xt],axis=0)
    return x

def gen_nl_random(ns,seq,G,L,P,p,M,means,sigma,k,f1,f2):
    ncols = len(means[0])
    x = np.zeros([0,ncols])
    for l in seq:
        xl = gen_nl_random_l(ncols, ns[l], G[l], L[l], P, p[l], M[l], means[l], sigma[l], f1,f2,k)
        x = np.concatenate([x,xl],axis=0)
    return x


def square(x):
    return np.sign(x)*np.abs(x)**2

def sin(x):
    return np.sin(2*np.pi*0.5*x)

def ide(x):
    return x


def log_gaussian_kernel(x):
    return -0.5*(x**2+np.log(2*np.pi))

def gaussian_kernel(x):
    return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
     

#%% con más variables
K = 7
N=3
means_g = np.array([ [2.3, 4.2, 3.5, 2.5, 1.2, 1, 2],[-2.1, 3.4, -2.4, 3.2, -4.1, 1 , 2], [-3.5 , -1.3, -4.4 , -0.5, -2.2, 1 , 2] ])
sigmas_g = np.array( [[0.7, 0.7, 0.8, 0.6, 1,0.5,0.6],[ 1, 0.8, 1.0 ,0.9 ,0.8,0.5,0.6], [0.8, 0.5, 1.0, 1.0, 0.8,0.5,0.6]])
k = np.array([ 3, 2, 1, 3, 2, 1 ,1])
G          = np.zeros([3,7,7])
G[1][1][0] = 1; G[1][3][2]=1; G[1][4][0]=1 
G[2][1][0] = 1; G[2][1][4]=1; G[2][2][3] =1; G[2][2][4] = 1; G[2][3][4] =1
L          = []
for i in range(3):
    lel = dag_v(G[i])
    print(lel[0])
    L.append(lel[1])
L            = np.array(L)
P            = 2
p            = [[0,0,0,0,0,0,0],[0,1,0,1,0,0,0],[1,2,1,2,1,0,0]]
MT            = np.zeros([N,K,K+P])
MT[1][1][0]   =  1.5  ; MT[1][3][2] = 1.2 ; MT[1][4][0] = -0.6 ; MT[1][1][5] = 0.6 ; MT[1][3][5] = 0.2
MT[2][1][0]   = -1.3  ; MT[2][1][4] = 1.4 ; MT[2][2][3] = -1.8 ; MT[2][2][4] = 2.3 ; MT[2][3][4] = 0.5 
MT[2][0][5]   =  0.2  ; MT[2][1][5] = 0.1 ; MT[2][1][6] =  0.4 ; MT[2][2][5] = 0.5 ; MT[2][3][5] = 0.1;  MT[2][3][6]= 0.89; MT[2][4][5] = 0.5
nses         = [100,200,300]
seqss        = [0,1,2,1,0,2,0]
data_gen     = gen_nl_random(nses,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,sin)
lengths_gen  = np.array([len(data_gen)])


model1 = KDE_AsHMM(data_gen, lengths_gen, 3,P=P)
model1.EM()

model2 = KDE_AsHMM(data_gen, lengths_gen, 3,P=P)
model2.SEM()

model3 = hmm(data_gen, lengths_gen, 3,P=P)
model3.EM()

model4 = hmm(data_gen, lengths_gen, 3)
model4.SEM()

model5 = KDE_AsHMM(data_gen, lengths_gen, 3,p=p,G=G,P=P)
model5.EM()

model6 = KDE_AsHMM(data_gen, lengths_gen, 3,P=P)


data_gen_test = gen_nl_random(nses,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,sin)
ll1 = model1.log_likelihood(data_gen_test )
ll2 = model2.log_likelihood(data_gen_test )
ll3 = model3.log_likelihood(data_gen_test )
ll4 = model4.log_likelihood(data_gen_test )
ll5 = model5.log_likelihood(data_gen_test )
ll6 =  model6.log_likelihood(data_gen_test )
print("Likelihood KDE-HMM:               "+ str(ll1))
print("Likelihood KDE-AsHMM:             "+ str(ll2))
print("Likelihood HMM:                   "+ str(ll3))
print("Likelihood AR-AsLG-HMM:           "+ str(ll4))
print("Likelihood KDE-HMM with known BN: "+ str(ll5))
print("Likelihood KDE-HMM no train:      "+ str(ll6))
# Toco cmabiar el penalty, pero ya se obtienen grafos más razonables, se cambio ln(T) -> T
# Si se le entrega una inicialización de v aleatoria, puede llevar a resultados muy buenos o muy malos
# debe buscarse una forma de iniciar el vector v tal que ayude a descubrir patrones lineales y no lineales, y pueda ser replicable...


#%% generar datos
means =  [0.5,1.5,5.5]
sigmas = [1.0,0.8,0.7]
ks = [5,-5,2.5]
rootss = [[2,-1],[0,3],[5,6]]
ns =  [200,200,200]  #entre menos valores, mas estructura aparece y es más importante, entre más datos, la estrucutra deja de ser relevante
ar_coef = [[0.4,0.5],[-0.4,-0.2],[0.6,0]]
seq = [0,1,2,1,0,2,1,2,2,1,1]
data = generate_2D_seq(means, sigmas, ks, rootss, ns,ar_coef, seq)
lengths = np.array([len(data)])
plt.figure("Data scatter")
plt.clf()
plt.title("Distribution from samples")
plt.scatter(data[:,0],data[:,1])
plt.grid("on")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.tight_layout()

k  = 0 
ar = 1
plt.figure("Data scatter AR")
plt.clf()
plt.scatter(data[ar:,k],data[:-ar,k])
plt.grid("on")
plt.xlabel("$x_{"+str(k)+"}^{t}$")
plt.ylabel("$x_{"+str(k)+"}^{t-"+str(ar)+"}$")
plt.tight_layout()
root = r"C:\Users\fox_e\Dropbox\Doctorado\Software\HMM_develop\C\C++"
np.savetxt(root + r'\prueba.csv',data,delimiter = ",")

model1 = KDE_AsHMM(data, lengths, 3,P=P)
model1.EM()

model2 = KDE_AsHMM(data, lengths, 3,P=P)
model2.SEM()


G2 = []
for i in range(3):
    Gi = np.array([[0,1],[0,0]])
    G2.append(Gi)
p2 = [[2,0],[2,0],[1,0]]

model3 = hmm(data, lengths, 3,P=P)
model3.EM()

model4 = hmm(data, lengths, 3,P=P)
model4.SEM()

model5 = KDE_AsHMM(data, lengths, 3,p=p2,G=G2,P=P)
model5.EM()

#%% Comparación likelihood
ns_test =  [600,400,700]
test_data = generate_2D_seq(means, sigmas, ks, rootss, ns_test, ar_coef,seq)
ll1 = model1.log_likelihood(test_data)
ll2 = model2.log_likelihood(test_data)
ll3 = model3.log_likelihood(test_data)
ll4 = model4.log_likelihood(test_data)
ll5 = model5.log_likelihood(test_data)

print("Likelihood KDE-HMM:         "+ str(ll1))
print("Likelihood KDE-AsHMM:       "+ str(ll2))
print("Likelihood HMM:             "+ str(ll3))
print("Likelihood AR-AsLG-HMM:     "+ str(ll4))
print("Likelihood KDE-HMM BN know: "+ str(ll5))
        
        
        
        
        
        
        
        
        