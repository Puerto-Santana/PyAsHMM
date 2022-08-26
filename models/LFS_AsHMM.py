# -*- coding: utf-8 -*-
"""
@date: Created on Thu Aug 13 08:46:53 2020
@author: Carlos Puerto-Santana at Aingura IIoT, San Sebastian, Spain
@Licence: CC BY-NC-ND 4.0
"""
import pickle
import datetime
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class AsFS_AR_ASLG_HMM:    
    def __init__(self,O,lengths,N,rho= None ,p=None,P=None,A=None, pi=None,G=None,L=None,
                 B=None,sigma=None,epsi = None, tau= None,p_cor=5,struc=True,lags=True,
                 act_inde = True,rho_bar =0.9):
        """
        Creates an object AsFS_AR_ASLG_HMM
        Based on the paper:
            C. Puerto-Santana, P. Larranaga and C. Bielza, "Features Saliencies in Asymmetric Hidden Markov Models"

        Parameters
        ----------
        O : TYPE Numpy array of size TxK
            DESCRIPTION. Observations  to be used in training. T is the number of instances and K is the number of variables
            In case of using two or more datatsets for training, concatenate them such that T=T1+T2+T3+... Ti is the length of dataset i
        lengths : TYPE  Numpy array of size M
            DESCRIPTION. in case of using two or more datasets for training, each component of this array is the length of each dataset
            ex: lengths = np.array([T1,T2,T3,...])
        N : TYPE int
            DESCRIPTION. number of hidden states
        rho : TYPE numpy array of size NxK
            DESCRIPTION relevancies parameters for each hidden state and variable
        p : TYPE,  Numpy array of size NxK 
            DESCRIPTION. The default is None. Matrix of AR orders 
        P : TYPE, optional int 
            DESCRIPTION. The default is None. Maximum permited lag
        A : TYPE, optional Numpy array of size NxN
            DESCRIPTION. The default is None. Transition matrix
        pi : TYPE, optional  Numpy array of size N
            DESCRIPTION. The default is None. Initial distirbution
        G : TYPE, optional list  of size N of Numpy arrays of size KxK
            DESCRIPTION. The default is None. list of graphs
        L : TYPE, optional list of size N Numpy arrays of size K
            DESCRIPTION. The default is None. Represents the toological orders of the graphs in G
        B : TYPE,  list of size N with lists of the parameters given by p and G
            DESCRIPTION. The default is None.
        sigma : TYPE, optional  Numpy arrays of size NxK
            DESCRIPTION. The default is None. Standard deviations
        epsi : TYPE, optional Numpy array of size K
            DESCRIPTION. The default is None. means of irrelevant variables
        tau : TYPE, optional Numpy array of size K
            DESCRIPTION. The default is None. Standard deviations of irrelevant variables
        p_cor : TYPE, optional int
            DESCRIPTION. The default is 5.
        struc : TYPE, optional bool
            DESCRIPTION. The default is True. Performs structural optimization
        lags : TYPE, optional bool
            DESCRIPTION. The default is True. Performs time structure optimization
        act_inde : TYPE, optional
            DESCRIPTION. The default is True.
        rho_bar : TYPE, optional float
            DESCRIPTION. The default is 0.9. relevance treshold between 0 and 1

        Returns
        -------
        None.

        """
        self.rho_bar = rho_bar
        self.act_inde=act_inde
        self.kappa = None     # Valores estandar apra cada variable, se usa en Viterbi
        self.nu = None        # Valores de relevancia o escala de cada varaible 
        self.struc = struc    
        self.lags =lags       
        self.mus = None       # Valor esperado de cada variable dado el estado oculto
        self.covs = None      # Una lista de matrices de convarianzas de cada estado oculto
        self.O = np.array(O)            # Observacion
        self.LogLtrain = None # Verosimilitud de los datos de entrenamiento
        self.bic = None       # Bayesain information criterion de los datos de entrenamiento
        self.b = None         # Numero de parametros 
        self.forBack = []     # Lista de objetos forBack (aqui se jecuto el forward-backward y el paso E del EM. )
        self.dictionary = None# Orden de los estados segun severidad
        self.K = len(O[0])    #Numero de variables
        self.N = N            #Numero de estados
        self.lengths = lengths#vector con longitudes de las observaciones    
        if rho is None:
            self.rho = 0.5*np.ones([N,self.K])
        else:
            self.rho = rho
        if P is None:
            self.P = self.det_lags(p_cor=p_cor)
        else:
            self.P = P
        if p is None:
            self.p = np.zeros([N,self.K]).astype(int)
        else:
            self.p = p  
        if  A is None:
            self.A = np.ones([N,N])/N
        else:
            self.A = A
        if pi is None:
            self.pi = np.ones(N)/N
        else:
            self.pi= pi
        if  epsi is None:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            self.epsi = np.mean(O,axis=0)
        else:
            self.epsi= epsi
        if  tau is None:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            self.tau = np.sqrt(np.var(O,axis=0))
        else:
            self.tau= tau
        if G is None:
            G = self.prior_G(N,self.K)
            L= []
            for i in range(N):
                L.append(np.arange(self.K))
            self.G = G
            self.L = L
        else:
            self.G=G
            self.L = L
        if B  is None:
            kmeans = KMeans(n_clusters=self.N,n_init=100).fit(O)
            B= []
            for i in range(N):
                Bi = []
                bi = kmeans.cluster_centers_[i]
                for k in range(self.K):
                    pak = self.my_fathers(self.G[i],k)
                    # Bik = np.concatenate([[bi[k]],np.zeros(len(pak)+self.p[i][k])])
                    Bik = np.concatenate([[(i+1)*(np.max(O.T[k])-np.min(O.T[k]))/(self.N+1)+np.min(O.T[k])],np.zeros(len(pak)+self.p[i][k])])
                    Bi.append(Bik)
                B.append(Bi) 
            self.B = B
        else :
            self.B=B
            
        if sigma is None:
            # self.sigma = np.tile(np.sqrt(np.var(O,axis=0)),self.N).reshape([self.N,self.K])
            self.sigma = np.ones([self.N,self.K])*(np.max(O,axis=0)-np.min(O,axis=0))
        else:
            self.sigma = sigma
            
        self.pA = self.A
        self.ppi = self.pi
        self.prho = self.rho
        self.pB = self.B
        self.psigma = self.sigma
        if self.act_inde == True:
            self.ptau = self.tau
            self.pepsi = self.epsi
        self.build_matrices()
            
            
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
#            O = np.diff(self.O.T[i])
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
    
      
    def plot_graph(self,route=None,label=None,coef=False):
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
            label = []
            for k in range(self.K):
                label.append("X_{"+str(k+1)+"}")
                
        if self.dictionary is not None:
            dic, g = self.dictionary
        else:
            dic= range(self.K)
            g = range(self.K)
        for index in range(self.N):
            plt.figure("Graph_"+str(round(g[dic[index]],4)))
            plt.clf()
            D = nx.DiGraph()
            G = self.G[index]
            B = self.B[index]
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
            nx.draw_networkx_labels(D, pos, labels=node_labels,font_size=9)
            edge_labels=dict([((u,v,),d['weight']) for u,v,d in D.edges(data=True)])
            nx.draw_networkx_edge_labels(D,pos,edge_labels=edge_labels,font_size=9)
            nx.draw(D,pos, node_size=800,edge_cmap=plt.cm.Reds,arrows=True,node_color="silver",linewidths=2)
            plt.tight_layout()
            if route is not None:
                plt.savefig(route+r"\Graph_g_"+str(round(g[dic[index]],2))+".pdf",dpi=600,bbox_inches="tight")
            plt.show()
    
    def pen(self,G,p,lengths): 
        """
        Calculates the penalty of a list of graphs

        Parameters
        ----------
        G : TYPE list of matrices
            DESCRIPTION.  List of grpahs, each graph ha size KxK, where K is the number of variables
        p : TYPE Numpy array of size NxK
            DESCRIPTION. Matrix of number of AR relationships
        lengths : TYPE list
            DESCRIPTION. lengths of the observed data

        Returns
        -------
        list
            DESCRIPTION. First component is the BIC penalty, second is the number of parameters

        """
        T =np.sum(lengths)
        b=0
        for i in range(self.N):
            b = b+np.sum(G[i])+2*self.K+ np.sum(p[i])+self.K
        b = b + self.N**2+self.N
        b = b + 2*self.K
        self.b=b
        return [-b*0.5*np.log(T),b]
    
    
    def score_complete_info_estimation_local(self,G,L,p,i,k,tb,ts): 
        """
        Estimates the parameters based on the complete information

        Parameters
        ----------
        G : TYPE list
            DESCRIPTION. list with the graph matrices  of size KxK
        L : TYPE list
            DESCRIPTION. list with topological orders  of size K
        p : TYPE numpy array of size NxK
            DESCRIPTION. matrix of number of AR values
        i : TYPE int
            DESCRIPTION. hidden state to compute the complete info component
        k : TYPE int
            DESCRIPTION. variables to compute the complete info component

        Returns
        -------
        list
            DESCRIPTION. first component is local-likelihood, 
            second component is the learned B
            third component is the learned sigma
            fourth compoent is the new number of parameters 
        """
        B = self.act_B2(G,p,tb)
        self.act_mut(G,B,p)
        sigma = self.act_S(ts)
        [ll,b] = self.local_score(G,p,sigma,i,k)
        return [ll,B,sigma,b]

    def local_score(self,G,p,sigma,j,k):
        """
        Computes a local score for a given hidden state j and vvariable k

        Parameters
        ----------
        G : TYPE list 
            DESCRIPTION. list of graph eashc of size KxK
        p : TYPE Numpy array NxK
            DESCRIPTION. matrix with number of AR values per variable
        sigma : TYPE Numpy array NxK
            DESCRIPTION. matrix with the sigma values 
        j : TYPE int
            DESCRIPTION. hidden state to compute local score
        k : TYPE int
            DESCRIPTION. variable to compute local sore

        Returns
        -------
        list
            DESCRIPTION. first component is local BIC, 
            second component is the BIC penalty
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        pena, npa = self.pen(G,p,self.lengths)
        L = 0.
        for i in range(len(self.forBack)):
            for t in range(self.lengths[i]-self.P):
                f = self.forBack[i].psi[j][t][k]*(self.forBack[i].prob_dep_ik((self.O[aux_len[i]:aux_len[i+1]])[t+self.P],self.forBack[i].mut[:,t,:],sigma,j,k))
                L = L+ f
        # return [L+pena+pena2,npa]
        return [L+pena,npa]
    
    def score(self,G,p,sigma): 
        """
        Computes the BIC of the given Bayesian networks and AR orders

        Parameters
        ----------
        G : TYPE list
            DESCRIPTION. list of matrices of the Bayesian networks by graphs of size KxK
        p : TYPE numpy array of size NxK
            DESCRIPTION. matrix of AR orders
        sigma : TYPE numpy array of size NxK
            DESCRIPTION. sigma values

        Returns
        -------
        list
        DESCRIPTION. first component is local BIC, 
        second component is the BIC penalty

        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        pena, npa = self.pen(G,p,self.lengths)
        L = 0.
        for i in range(len(self.forBack)):
            for t in range(self.lengths[i]-self.P):
                f = self.forBack[i].psi[:,t,:]*(self.forBack[i].prob_normal((self.O[aux_len[i]:aux_len[i+1]])[t+self.P],self.forBack[i].mut[:,t,:],sigma))
                L = L+ np.sum(f)
        return [L+pena,npa]
    
    def score_complete_info_estimation(self,G,L,p,tb): 
        """
        Estimates the parameters based on the complete information

        Parameters
        ----------
        G : TYPE list
            DESCRIPTION. list with the graph matrices  of size KxK
        L : TYPE list
            DESCRIPTION. list with topological orders  of size K
        p : TYPE numpy array of size NxK
            DESCRIPTION. matrix of number of AR values
            
        Returns
        -------
        list
            DESCRIPTION. first component is likelihood, 
            second component is the learned B
            third component is the learned sigma
            fourth compoent is the new number of parameters 
        """
        B = self.act_B2(G,p,tb)
        self.act_mut(G,B,p)
        sigma = self.act_S()
        [ll,b] = self.score(G,p,sigma)
        return [ll,B,sigma,b]
    
    def climb_ar(self,tb,ts):
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
        for i in range(self.N):
            accept = np.argwhere(self.rho[i]>self.rho_bar)[:,0]
            for k in accept:
                sm = self.local_score(self.G,self.p,self.sigma,i,k)[0]
                while self.p[i][k] +1 <= self.P :
                    p2 = np.copy(self.p)
                    p2[i][k] = p2[i][k] +1
                    try:
                        [s2,B2,sigma2,b2] = self.score_complete_info_estimation_local(self.G,self.L,p2,i,k,tb,ts)
                    except:
                        # print("AR candidate produced error. It was skipped")
                        s2 = sm-1
                    if s2 > sm:
                        self.p = p2
                        self.b = b2
                        self.sigma =sigma2
                        self.B = B2
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
                if G2[i][j] != 1 :
                    G2[i][j] = 1
                    [fool,L] = self.dag_v(G2)
                    if fool ==True:
                        indexj.append(j)
            indexi.append(indexj)
            index.append(indexi)
        return index
    
    def climb_struc(self,tb,ts): 
        """
        Looks for the best graphical structure, uses an upward greedy algorithm

        Parameters
        ----------
        tb : TYPE boolean list 
            DESCRIPTION. indices to be updated of the parameter B
        ts : TYPE boolean list
            DESCRIPTION. indices to be updated of the parameter sigma
        """
        for i in range(self.N):
            accept = np.argwhere(self.rho[i]>self.rho_bar).T[0]
            # accept = np.array(self.K)
            # for k in range(self.K):
            for k in accept:
                possi = self.pos_ads(self.G[i])
                son = possi[k][0]
                if len(possi[k][1])!=0:
                    sm = self.local_score(self.G,self.p,self.sigma,i,son)[0]
                    for j in possi[k][1]:
                        if j in accept:
                            G2 = np.copy(self.G)
                            G2[i][son][j] =1
                            L2 = []
                            for nn in range(self.N):
                                L2.append(self.dag_v(G2[nn])[1])
                            try:
                                [s2,B2,sigma2,b2] = self.score_complete_info_estimation_local(G2,self.L,self.p,i,son,tb,ts)
                            except:
                                # print("Graph candidate produced error. It was skipped")
                                s2=sm-1
                            if s2>sm:
                                sm= s2
                                self.B = B2
                                self.b = b2
                                self.sigma = sigma2
                                self.G = G2
                                self.L = L2

                        
    def hill_climb(self,struc,lags,tb,ts):
        """
        Executes a greedy forward algorithm to explore the Bayesian networks space.

        Parameters
        ----------
        tb : TYPE boolean list 
            DESCRIPTION. indices to be updated of the parameter B
        ts : TYPE boolean list
            DESCRIPTION. indices to be updated of the parameter sigma
        """
        self.b = self.score(self.G,self.p,self.sigma)[1]
        if struc is True:
            self.climb_struc(tb,ts)
        if lags is True:
            self.climb_ar(tb,ts)
    

#  States management
                        
    def mvn_param(self):
        """
        Obtains the parameters of each multi dimensional normal distribution 
        corresponding to each state
        """
        mus = []
        covs = []
        for i in range(self.N):
            Gi = self.G[i]
            Li = self.L[i]
            Bi = self.B[i]
            sigi = self.sigma[i]
            mui = np.zeros(self.K)
            covi = np.zeros([self.K,self.K])
            for s in Li:
                Pas = list(self.my_fathers(Gi,s))
                if len(Pas) == 0:
                    mui[s] = Bi[s][0]
                    covi[s][s] = sigi[s]
                    if self.p[i][s]>0:
                        mui[s] = mui[s]/(1.-np.sum(Bi[s][-self.p[i][s]:]))
                else:
                    Bis = Bi[s]
                    SIGMA = ((covi[Pas]).T[Pas]).T
                    covi[s][s] = sigi[s] + np.dot(np.dot(Bis[1:len(Pas)+1],SIGMA),Bis[1:len(Pas)+1])
                    cof = np.concatenate([[1.],mui[Pas]])
                    cof2 = np.concatenate([[Bis[0]],Bis[1:len(Pas)+1]])
                    mui[s] = np.dot(cof2,cof)
                    if self.p[i][s]>0:
                        mui[s] = mui[s]/(1.-np.sum(Bi[s][-self.p[i][s]:]))
                    for k in Pas:
                        covi[s][k] = sigi[s] + np.sum(Bis[1:len(Pas)+1]*covi[Pas,s])
                        covi[k][s] = covi[s][k]
            mus.append(mui)
            covs.append(covi)
            self.mus = np.array(mus)
            self.covs = np.array(covs)
            

    def label_parameters(self,kappa,nu):
        """
        Change the parameters for labelling for the Viterbi algorithm

        Parameters
        ----------
        kappa : TYPE numpy array of size K
            DESCRIPTION. base knowledge parameters
        nu : TYPE numpy array of size K
            DESCRIPTION. scale parameters

        Returns
        -------
        str
            DESCRIPTION. if the dimension is not correct

        """
        if len(kappa)!= self.K:
            return "Wrong dimension"
        self.kappa = kappa
        self.nu = nu
        
    def states_order(self,maxg = False,absg = True):
        """
        Order the states depending on the energy of the mvn mean vector taking into account AR coefficients

        Parameters
        ----------
        maxg : TYPE, optional bool
            DESCRIPTION. The default is False.uses max function in the labeling
        absg : TYPE, optional bool
            DESCRIPTION. The default is True. uses absolute value in the labeling
        """
        # mag = {}
        # var = np.argwhere(self.rho>self.rho_bar)[:,0]
        # if self.kappa is not None or self.nu is not None:
        #     if maxg== False and absg == False:
        #         music = np.sum((self.mus[:,var]-self.kappa[var])*self.nu[var],axis=1)
        #     if maxg== True and absg == False:
        #         music = np.max((self.mus[:,var]-self.kappa[var])*self.nu[var],axis=1)
        #     if maxg== False and absg == True:
        #         music = np.max(np.abs(self.mus[:,var]-self.kappa[var])*self.nu[var],axis=1)
        #     if maxg== True and absg == True:
        #         music = np.max(np.abs(self.mus[:,var]-self.kappa[var])*self.nu[var],axis=1)
        # else:
        #     if maxg== False and absg == False:
        #         music = np.sum(self.mus[:,var],axis=1) 
        #     if maxg== True and absg == False:
        #         music = np.max(self.mus[:,var],axis=1)     
        #     if maxg== False and absg == True:
        #         music = np.sum(np.abs(self.mus[:,var]),axis=1) 
        #     if maxg== True and absg == True:
        #         music = np.max(np.abs(self.mus[:,var]),axis=1) 
        mag = {}
        if self.kappa is not None or self.nu is not None:
            if maxg== False and absg == False:
                music = np.sum((self.mus-self.kappa)*self.nu,axis=1)
            if maxg== True and absg == False:
                music = np.max((self.mus-self.kappa)*self.nu,axis=1)
            if maxg== False and absg == True:
                music = np.max(np.abs(self.mus-self.kappa)*self.nu,axis=1)
            if maxg== True and absg == True:
                music = np.max(np.abs(self.mus-self.kappa)*self.nu,axis=1)
        else:
            if maxg== False and absg == False:
                music = np.sum((self.mus),axis=1) 
            if maxg== True and absg == False:
                music = np.max((self.mus),axis=1)     
            if maxg== False and absg == True:
                music = np.sum(np.abs(self.mus),axis=1) 
            if maxg== True and absg == True:
                music = np.max(np.abs(self.mus),axis=1) 
        index = np.argsort(music)
        j=0
        for i in index:
            mag[j] = music[i]
            j=j+1
        self.dictionary= [index,mag]
        

#   HMM
    def log_likelihood(self,O,pena =False,ps=None):
        """
        Computes the log-likelihood of a dataset O.
        it uses the  current learned parameters of the model

        Parameters
        ----------
        O : TYPE Numpy array of size T'xK
            DESCRIPTION. Observations to be tested
        pena : TYPE, optional bool
            DESCRIPTION. The default is False. it applies the BIC penalization
            
        Returns
        -------
        TYPE float
            DESCRIPTION. The score of the observations O

        """
        fb = forback()
        fb.mu_t(O,self.B,self.G,self.N,self.p,self.P,self.K)
        fb.prob_BN_t(O, self.sigma, self.epsi, self.tau, self.rho, self.P)
        fb.forward_backward(self.A,self.pi,self.rho,O,self.N,self.P,ps=ps)
        Clist = fb.ll
        log = np.sum(-Clist)
        del fb
        if pena ==False:
            return log
        else:
            return -2*(log+self.pen(self.G,self.p,np.array([len(O)]))[0])
        
        
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
    
    def viterbi(self,O,plot=True,maxg = False,absg = True,indexes=False,xlabel="Time units",ylabel="Values",ps=None): 
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

        Returns 
        -------
        TYPE numpy array of size T 
            DESCRIPTION. Most likely sequence of hidden states

        """
        if indexes==True:
            self.states_order(maxg=maxg,absg=absg)
        T =len(O)
        fb = forback()
        fb.mu_t(O,self.B,self.G,self.N,self.p,self.P,self.K)
        fb.prob_BN_t(O,self.sigma,self.epsi,self.tau,self.rho,self.P)
        if ps == None:
            delta = np.log(self.pi)+fb.probt[0]
        else:
            delta = np.log(self.A[ps])+fb.probt[0]
        psi = [np.zeros(len(self.pi))]
        for t in range(1,T-self.P):
            delta = self.viterbi_step(delta,fb,t)
            psi.append(np.argmax(delta+np.log(self.A.T),axis=1))
            
        psi = np.flipud(np.array(psi)).astype(int)
        Q = [np.argmax(delta)]
        for t in np.arange(0,T-1-self.P):
            Q.append(psi[t][Q[t]])
        Q=np.array(Q)
        if indexes==False:
            [index,mag] = self.dictionary
            states = []
            for i in range(len(Q)):
                states.append(mag[index[Q[i]]])
            Q= np.array(states)
            
        
        if plot==True:
            plt.figure("Sequence of labeled hidden states")
            plt.clf()
            plt.title("Sequence of labeled hidden state")
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.plot(Q[::-1])
            plt.grid("on")
            plt.show()
            plt.tight_layout()
            plt.pause(0.2)
        return Q[::-1]
    
    def relevancy_evolution(self,O,plot=True,ps=None):
        Q = self.viterbi(O,plot=False,indexes=True,ps=ps)
        rel = self.rho[Q]
        if plot ==True:
            plt.figure("Relevancy evolution")
            plt.clf()
            for i in range(self.K):
                plt.plot(rel[:,i],label="Variable:" +str(i))
            plt.xlabel("Time units")
            plt.ylabel(r"$\rho$")
            plt.grid("on")
            plt.tight_layout()
            plt.legend()
        return rel
        
        
    
    
    def sample(self,leng,pseed,ini=None):
        """
        Warning: not implemented
        Generates a sample from the current parameters 
        A pseed of length P*xK must be provided
        returns the generated data and the sequence of hidden states

        Parameters
        ----------
        leng : TYPE int
            DESCRIPTION. length of the desired sample
        pseed : TYPE numpy array of size PxK
            DESCRIPTION. seed of the sample, if P is 0, provide an empty numpy array

        Returns
        -------
        list
            DESCRIPTION. first is the observations
            second is the sequence of hidden states
        """

        return None
     
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
    
    def act_mut(self,G,B,p):
        """
        Updates the mut ( i.e., the mean for each time instance, hidden state and variable,) parameter of each forBack object

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
            self.forBack[i].mu_t(self.O[aux_len[i]:aux_len[i+1]],B,G,self.N,p,self.P,self.K)
    

    def act_gamma(self,ps=None):
        """
        Updates for each forBack object its gamma parameter or latent probabilities
        
        Parameters
        ----------
        ps : TYPE, optional int
            DESCRIPTION. The default is None. index of the initial distibution
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].forward_backward(self.A,self.pi,self.rho,self.O[aux_len[i]:aux_len[i+1]],self.N,self.P,ps=ps)


    def act_A(self,ta):
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
        self.forBack[0].act_aij(self.A,self.N,self.O[aux_len[0]:aux_len[1]],self.P)
        numa = self.forBack[0].numa
        dena = self.forBack[0].dena
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_aij(self.A,self.N,self.O[aux_len[i]:aux_len[i+1]],self.P)
            numa = numa +self.forBack[i].numa 
            dena = dena +self.forBack[i].dena 
        A = np.ones([self.N,self.N])
        for j in range(self.N):
            if ta[j] != 1:
                A[j] = numa[j]/dena[j]
            else:
                A[j] = numa[j]/dena[j]
                for k in range(self.N):
                    A[k][j] = numa[k][j]/dena[k]
        for n in range(self.N):    
            A[n] = A[n]/np.sum(A[n]) # Normalizo para asegurar que cada fila de A sea una distribucion
        return A
        
    def act_pi(self):
        """
        Updates the initial distribution parameters
        """
        api = self.forBack[0].gamma[0]
        for i in range(1,len(self.lengths)):
            api = api+self.forBack[i].gamma[0]
        npi = api/len(self.lengths)
        npi = self.checkzero(npi)
        return npi

    ########### Prueba, actualizacion alternativa de parametros ###############  
    def act_B2(self,G,p,tb):
        gamma = np.zeros([self.N,0,self.K])
        for l in range(len(self.lengths)):
            gamma = np.concatenate([gamma,self.forBack[l].psi],axis=1)
        
        B = []
        for i in range(self.N):
            if tb[i] ==1:
                Gi = G[i]
                Bi=[]
                for k in range(self.K):
                    gamma_ik = gamma[i,:,k]
                    gamma_ik = gamma_ik.reshape([len(gamma_ik),1])
                    Gik =Gi[k]
                    pa_ind = np.argwhere(Gik==1).T[0]
                    Xm = self.Og[:,k]
                    pa = self.Og[:,pa_ind]
                    ones = np.ones([len(Xm),1])
                    pa = np.concatenate([ones,pa],axis=1)
                    d =  self.d[k][:,:p[i][k]]
                    U = np.concatenate([pa,d],axis=1)
                    gammau = gamma_ik*U
                    ugu = np.dot(gammau.T,U)
                    # print("Determinante de matriz Upsi: " +str(np.linalg.det(ugu)))
                    Bik = np.dot(np.dot(np.linalg.inv(ugu),gammau.T),Xm)
                    Bi.append(Bik)
                B.append(np.array(Bi))
            else:
                B.append(self.B[i])
        return B
    
    def build_matrices(self):
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        Og = np.zeros([0,self.K])
        for l in range(len(self.lengths)):
            Ol = (self.O[aux_len[l]:aux_len[l+1]])[self.P:]
            Og = np.concatenate([Og,Ol],axis=0)
    
        d = []
        for m in range(self.K):
            dm = np.zeros([0,self.P])
            for l in range(len(self.lengths)):
                Ol = (self.O[aux_len[l]:aux_len[l+1]])[:,m]
                dml = np.zeros([len(Ol)-self.P,self.P])
                for j in range(1,self.P+1):
                    z = np.roll(Ol,j)[self.P:]
                    dml[:,j-1]= z
                dm = np.concatenate([dm,dml],axis=0)
            d.append(dm)

        self.Og = Og
        self.d = d
    ############################ Aquí termina la prueba #######################    
    def act_B(self,G,p,tb):
        """
        Updated the emission probabilities parameters

        Parameters
        ----------
        G : TYPE list
            DESCRIPTION. list of graphs with numpy arrays of size KxK
        p : TYPE numpy array of size NxK
            DESCRIPTION. matrix of number of AR values
        tb : TYPE boolean list 
            DESCRIPTION. indices to be updated of the parameter B

        Returns
        -------
        B : TYPE list
            DESCRIPTION. list with lits with the parameters of the emission probabilities

        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_coef(G,self.O[aux_len[0]:aux_len[1]],p,self.P)
        ac = self.forBack[0].coefvec
        bc = self.forBack[0].coefmat    
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_coef(G,self.O[aux_len[i]:aux_len[i+1]],p,self.P)
            bc = bc+ self.forBack[i].coefmat
            ac = ac+ self.forBack[i].coefvec
        B= []
        for i in range(self.N):
            if tb[i] ==1:
                Bi = []
                for k in range(self.K):
                    if np.prod(bc[i][k].shape)> 1:
                        Bik = np.linalg.solve(bc[i][k],ac[i][k])
                    else:
                        Bik = (ac[i][k]/bc[i][k])[0]
                    Bi.append(Bik)
                B.append(Bi)
            else:
                B.append(self.B[i])
        return B
    
    def act_S(self,ts):
        """
        It updates the sigma parameter

        Parameters
        ----------
        ts : TYPE boolean list
            DESCRIPTION. list that gives the indices of which hidden-states variances are updated 

        Returns
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_sigma(self.O[aux_len[0]:aux_len[1]],self.N,self.P,ts)
        nums = self.forBack[0].numsig
        dens = self.forBack[0].densig
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_sigma(self.O[aux_len[i]:aux_len[i+1]],self.N,self.P,ts)
            nums = nums+ self.forBack[i].numsig
            dens = dens+self.forBack[i].densig
        cov = []
        for i in range(self.N):
            if ts[i] ==1:
                covi = (nums[i]/dens[i])**0.5
                for k in range(self.K):
                    if covi[k] <= 1e-10:
                        covi[k] = np.sqrt(np.var(self.O[:,k]))
                cov.append(covi)
            else:
                cov.append(self.sigma[i])
        # cov = np.array(cov)
        # inds = np.argwhere(cov<1e-10).T[0]
        # cov[inds] = 1e-10
        
        return np.array(cov)
    
    def act_rho(self,tb):
        """
        It updates the relevancy vector

        Returns
        -------
        rho2 : TYPE numpy array of size NxK
            DESCRIPTION. Updated 

        """
        self.forBack[0].act_rho()
        nums = self.forBack[0].numrho
        dens = self.forBack[0].denrho
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_rho()
            nums = nums+ self.forBack[i].numrho
            dens = dens+self.forBack[i].denrho
        rhoo = np.array(nums/dens)
        rho2 = self.checkrho(rhoo)
        for j in range(self.N):
            if tb[j]==0:
                rho2[j] = self.rho[j]
        
        for i in range(self.N):
            for k in range(self.K):
                if len(self.B[i][k]) == 1 and (np.abs(self.epsi[k]-self.B[i][k])<1e-1):
                        rho2[i][k]  = 0.01
        return rho2
    
    def checkzero(self,z):
        """
        Returns a modified vector z with no zero instances
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)
        y[inds] = np.exp(-740)
        return y
    
    def checkrho(self,rho):
        """
        It assures that the relevancy vector rho is in the set (0,1)
        """
        N,M = rho.shape
        y = np.copy(rho)
        for m in range(M):
            for n in range(N):
                if rho[n][m] <= 0:
                    y[n][m] = np.exp(-744)
                if rho[n][m] >= 1.:
                    y[n][m] = 1-0.00001
        return y
    
    def act_epsi(self):
        """
        It updates the noise mean vector parameter

        Returns
        -------
        TYPE numpy array of size K
            DESCRIPTION. the updated mean of irrelevant variables

        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_epsi(self.O[aux_len[0]:aux_len[1]],self.P)
        nums = self.forBack[0].numepsi
        dens = self.forBack[0].denepsi
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_epsi(self.O[aux_len[i]:aux_len[i+1]],self.P)
            nums = nums+ self.forBack[i].numepsi
            dens = dens+self.forBack[i].denepsi
        
        return np.array(nums/dens)
    
    def act_tau(self):
        """
        It updates the noise standard deviation vector parameter

        Returns
        -------
        tau : TYPE numpy array of size K
            DESCRIPTION. standad deviations of irrelevant variables

        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_tau(self.O[aux_len[0]:aux_len[1]],self.epsi,self.P)
        nums = self.forBack[0].numtau
        dens = self.forBack[0].dentau
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_tau(self.O[aux_len[i]:aux_len[i+1]],self.epsi,self.P)
            nums = nums+ self.forBack[i].numtau
            dens = dens+self.forBack[i].dentau
        tau =np.array(nums/dens)**(0.5)
        
        inds = np.argwhere(tau<1e-10).T[0]
        tau[inds] = 1e-10
        return tau
    
    def EM(self,its=300,err=1e-4,ta="all", tpi=True, tb="all",ts = "all",ps=None): #Puede que tenga errores, toca revisar con calma
        """
        Computes the EM algorithm for parameter learning

        Parameters
        ----------
        its : TYPE, optional int
            DESCRIPTION. The default is 200. Number of iterations
        err : TYPE, optional float
            DESCRIPTION. The default is 1e-10. maximum error allowed in the EM iteration

        """
        if ta == "all":
            ta = np.ones(self.N)
        if tb == "all":
            tb = np.ones(self.N)
        if ts == "all":
            ts = np.ones(self.N)
        eps = np.inf
        it = 0
        del self.forBack[:]
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            fb = forback()
            fb.mu_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.G,self.N,self.p,self.P,self.K)
            fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.sigma,self.epsi,self.tau,self.rho,self.P)
            self.forBack.append(fb)
        likeli= -np.inf
        while eps >err and it<its:
            self.act_gamma(ps=ps)
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-self.forBack[i].ll)
            eps = likelinew-likeli
            if np.mod(it,100) == 0:
                print("it: " +str(it+1)+". EM error: " + str(eps) +" with log-likelihood: " + str(likelinew))
            if eps<0 or np.isnan(likeli):
                self.A = self.pA
                self.pi = self.ppi
                self.rho = self.prho
                self.B = self.pB
                self.sigma = self.psigma
                if self.act_inde == True:
                    self.tau = self.ptau
                    self.epsi = self.pepsi
                self.act_mut(self.G,self.B,self.p)
                self.act_gamma(ps=ps)
                likelinew = 0
                for i in range(len(self.lengths)):
                    likelinew = likelinew + np.sum(-self.forBack[i].ll)
                self.bic = likelinew + self.pen(self.G,self.p,self.lengths)[0]
                self.LogLtrain = self.log_likelihood(self.O)
                print("it: " +str(it+1)+". EM error: " + str(eps) +" with log-likelihood: " + str(likelinew))
                break
            self.prho = self.rho
            self.rho = self.act_rho(tb)
            if self.act_inde==True:
                self.pepsi = self.epsi
                self.epsi = self.act_epsi()
                self.ptau = self.tau
                self.tau = self.act_tau()
            self.pA = self.A
            self.A = self.act_A(ta)
            self.ppi = self.pi 
            if tpi == True:
                self.pi = self.act_pi()
            self.pB = self.B
            self.B = self.act_B2(self.G,self.p,tb)
            self.act_mut(self.G,self.B,self.p)
            self.psigma = self.sigma
            self.sigma = self.act_S(ts)
            self.LogLtrain = likelinew
            self.bic = likelinew + self.pen(self.G,self.p,self.lengths)[0]
            self.b = self.pen(self.G,self.p,self.lengths)[1]
            for i,fb in enumerate(self.forBack):
                fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.sigma,self.epsi,self.tau,self.rho,self.P)
            likeli = likelinew
            # print(self.LogLtrain)
            it = it+1  
            if eps < err:
                print("it: " +str(it+1)+". EM error: " + str(eps) +" with log-likelihood: " + str(likelinew))
            if np.isnan(eps):
                self.A = self.pA
                self.pi = self.ppi
                self.rho = self.prho
                self.B = self.pB
                self.sigma = self.psigma
                if self.act_inde == True:
                    self.tau = self.ptau
                    self.epsi = self.pepsi
                self.act_mut(self.G,self.B,self.p)
                self.act_gamma(ps=ps)
                likelinew = 0
                for i in range(len(self.lengths)):
                    likelinew = likelinew + np.sum(-self.forBack[i].ll)
                self.bic = likelinew + self.pen(self.G,self.p,self.lengths)[0]
                self.LogLtrain = self.log_likelihood(self.O)
                print("it: " +str(it+1)+". EM error: " + str(eps) +" with log-likelihood: " + str(likelinew))
                break
                
        self.mvn_param()
        self.states_order()
        self.bic = -2*self.bic
        self.b = self.pen(self.G,self.p,self.lengths)[1]
                
    def SEM(self,err1=1,err2=1e-4,its1=10,its2=300,ta="all",tpi=True, tb="all",ts="all",ps=None): 
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
        """
        eps = 100
        it = 0
        likelihood = [ ]
        if tb =="all":
            tb = np.ones(self.N)
        if ts =="all":
            ts = np.ones(self.N)
        if ta =="all":
            ta = np.ones(self.N)
        it = 0
        likelihood = [ ]
        self.EM(its2,err2,ta=ta,tb=tb,tpi=tpi,ts=ts,ps=ps)
        likelihood.append(self.bic)
        print ("SEM-Epsilon: " + str(eps)) 
        while eps > err1 and it < its1:
            self.hill_climb(self.struc,self.lags,tb,ts)
            self.EM(its2,err2,ta=ta,tb=tb,tpi=tpi,ts=ts,ps=ps)
            eps = np.abs((self.bic-likelihood[-1]))
            likelihood.append(self.bic)
            it = it+1
            print ("SEM-Epsilon: " + str(eps)) 
            
# Forecasting

    def expected_value(self,epochs,state):
        """
        Gives an forecastingfor the given time and assumes that the hidden state is fixed

        Parameters
        ----------
        epochs : TYPE int
            DESCRIPTION. time for forecast
        state : TYPE int
            DESCRIPTION. index of the hidden state

        Returns
        -------
        list a list with three components
            DESCRIPTION. The first component is the expected value of the prediction
                         The second component is the standard deviation of the prediction
                         The third component is the transition probabilities 

        """
        if len(self.dictionary[0])>0:
            epochs = int(epochs)
            An= np.linalg.matrix_power(self.A, epochs+1)
            Ani = An[state]
            expected = 0
            for i in range(self.N):
                expected+= Ani[i]*self.dictionary[1][i]
            var = 0
            for i in range(self.N):
                var += Ani[i]*(self.dictionary[1][i]-expected)**2
            return [expected,var**0.5,Ani]


    def predict_s(self,t_final,state,plot=True,xlabel="Time units",ylabel="Expected value"):
        """
        Computes the evolution of the prediction for a given hidden state index 

        Parameters
        ----------
        t_final : TYPE int
            DESCRIPTION.time for forecast
        state : TYPE int
            DESCRIPTION. index of the hidden state
        plot : TYPE, optional bool
            DESCRIPTION. The default is True.
        xlabel : TYPE, optional str
            DESCRIPTION. The default is "Time units".
        ylabel : TYPE, optional str
            DESCRIPTION. The default is "Expected value".

        Returns
        -------
        list od two components
            DESCRIPTION. first component has the evolution of the expected values
                         second component has the evolution of the standard deviation values

        """
        t_final = int(t_final)
        exs = []
        var = []
        for t in range(t_final):
            [et,vt,at] = self.expected_value(t, state)
            exs.append(et)
            var.append(vt)
        exs = np.array(exs)
        var =np.array(var)
        if plot==True:
            plt.figure("Prediction after " + str(t_final)+ " epochs at hidden state " + str(state) +" by DHMM")
            plt.clf()
            plt.title("Prediction after " + str(t_final)+ " epochs at hidden state " + str(state) +" by DHMM")
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.plot(exs+2*var,color="red", label = "Prediction Deviation")
            plt.plot(exs-2*var,color="red")
            plt.fill_between(range(t_final),exs+1.96*var,exs-1.96*var,color="red",alpha=0.2)
            plt.plot(exs,color="blue",label = ylabel)
            plt.ylim(ymin=0)
            plt.grid("on")
            plt.legend(loc=1)
            plt.tight_layout()
        return [exs,var] 
            
    def predict(self,t_final,plot=True,xlabel="Time units",ylabel="E value"):
        """
        Computes the  evolution of expected values for all the hidden states

        Parameters
        ----------
        t_final : TYPE int
            DESCRIPTION.
        plot : TYPE, optional bool
            DESCRIPTION. The default is True.
        xlabel : TYPE, optional str
            DESCRIPTION. The default is "Time units".
        ylabel : TYPE, optional str
            DESCRIPTION. The default is "E value".

        Returns
        -------
        list 2 components
            DESCRIPTION. first component has the evolution of the expected values for each hidden state
                         second component has the evolution of the standard deviation values for each hidden state
        """
        exs = []
        std = []
        for i in range(self.N):
            [ex,va] = self.predict_s(t_final, i,plot=False)
            exs.append(ex)
            std.append(va)
        if plot==True:
            plt.figure("Prediction after " + str(t_final)+ " epochs by DHMM")
            plt.clf()
            n_rows = int(np.ceil(self.N**0.5))
            n_columns = int(np.round(self.N/n_rows))
            for i in range(self.N):
                plt.subplot(n_rows,n_columns,i+1)
                plt.title("Prediction State: "+str(i)+" label: "+str(round(self.dictionary[1][i],2)),fontsize=10)
                plt.ylabel(ylabel,fontsize=9)
                plt.xlabel(xlabel,fontsize=9)
                plt.plot(exs[i]+1.96*std[i],color="red", label = "Prediction deviation")
                plt.plot(exs[i]-1.96*std[i],color="red")
                plt.fill_between(range(t_final),exs[i]+2*std[i],exs[i]-2*std[i],color="red",alpha=0.2)
                plt.plot(exs[i],color="blue",label = ylabel)
                plt.ylim(ymin=0)
                plt.grid("on")
            plt.tight_layout()
        return [exs,std] 
            
        
    def prob_ev(self,T,j,plot=True,xlabel ="Time units"):
        """
        Evolution of transition probabilities for a fixed hidden state

        Parameters
        ----------
        T : TYPE int
            DESCRIPTION. time horizon
        j : TYPE
            DESCRIPTION. int
        plot : TYPE, optional index hidden state
            DESCRIPTION. The default is True.
        xlabel : TYPE, optional str
            DESCRIPTION. The default is "Time units".

        Returns
        -------
        None.

        """
        valores = np.zeros([T,self.N])
        for t in range(T):
            valores[t] = np.linalg.matrix_power(self.A,t)[j]
        if plot== True:
            plt.figure("DHMM Evolution of probabilities of state " + str(j)+" label "+str(round(self.dictionary[1][j],2)))
            plt.clf()
            plt.title( "Evolution of probabilities of state " + str(j)+" label:"+str(round(self.dictionary[1][j],2)))
            for i in range(self.N):
                if i!=j:
                    plt.plot(valores[:,i],label="Transit to " +str(round(self.dictionary[1][i],2)))
                if i==j:
                    plt.plot(valores[:,i],label="Remain")
            plt.grid("on")
            plt.ylabel("Probability")
            plt.xlabel(xlabel)
            plt.legend(loc=1)
            plt.tight_layout()
            

    def save(self,root=None, name = "hmm_"+str(datetime.datetime.now())[:10]):
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
        itemlist = [self.N,self.K,self.P,self.rho,self.A,self.pi,self.B,self.sigma,
                    self.epsi,self.tau,self.p,self.G,self.L,self.mus,self.covs,
                    self.dictionary,self.b,self.bic,self.LogLtrain,
                    self.rho_bar]
        if self.kappa is not None:
            itemlist.append(self.nu)
            itemlist.append(self.kappa)
        with open(root+"\\"+name+".fs2ashmm", 'wb') as fp:
            pickle.dump(itemlist, fp)
    
    def load(self,rootr):
        """
        Loads a model, must be an fs2ashmmfile 

        Parameters
        ----------
        rootr : TYPE str
            DESCRIPTION. direction of the saved model

        Returns
        -------
        str
            DESCRIPTION. in case of  wrong filename, returns error

        """
        if rootr[-9:] != ".fs2ashmm":
            return "The file is not an fs2ashmm file"
        with open(rootr, "rb") as  fp:
            loaded = pickle.load(fp)
        self.N          = loaded[0]
        self.K          = loaded[1]
        self.P          = loaded[2]
        self.rho        = loaded[3]
        self.A          = loaded[4]
        self.pi         = loaded[5]
        self.B          = loaded[6]
        self.sigma      = loaded[7]
        self.epsi       = loaded[8]
        self.tau        = loaded[9]
        self.p          = loaded[10]
        self.G          = loaded[11]
        self.L          = loaded[12]
        self.mus        = loaded[13]
        self.covs       = loaded[14]
        self.dictionary = loaded[15]
        self.b          = loaded[16]
        self.bic        = loaded[17]
        self.LogLtrain  = loaded[18]
        self.rho_bar    = loaded[19]
        if len(loaded)>20:
            self.nu     = loaded[20]
            self.kappa  = loaded[21]
            
    def latex_B_parameters_report(self,root, name):
        """
        Prints a txt document with the latex table of the models parameters

        Parameters
        ----------
        root : TYPE str
            DESCRIPTION. The directory where the file is saves
        name : TYPE
            DESCRIPTION. Name of the file

        Returns
        -------
        None.

        """
        file = open(root+"\\"+name+".txt","w")
        file.write(r"\begin{table}"+"\n")
        file.write(r"\begin{tabular}{ccc}"+"\n")
        file.write(r"Q & M & $f_{im}$ \\" + "\n")
        file.write(r"\hline"+"\n")
        for i in range(self.N):
            for k in range(self.K):
                file.write(str(i+1)+"&" +str(k+1) +"& $" + str(round(self.B[i][k][0],3)))
                pa_ik = []
                if len(self.G[i][k]) != 0:
                    pa_ik = self.my_fathers(self.G[i], k)
                    for j, pa in enumerate(pa_ik):
                        file.write("+"+str(round(self.B[i][k][j+1],3))+r"X^t_{"+ str(pa+1) +r"}" )
                if self.p[i][k] != 0:
                    for d in range(1,self.p[i][k]+1):
                        file.write("+"+str(round(self.B[i][k][len(pa_ik)+d],3))+r"X_{"+ str(k+1) +r"}^{t-"+str(d)+r"}" )
                file.write(r"$\\"+"\n")
            file.write(r"\hline"+"\n")
        file.write(r"\end{tabular}" +"\n")
        file.write(r"\end{table}" +"\n")
        file.close()
        
    
    def latex_rho_parameters(self,root,name,labs= None,cols=5):
        """
        """
        file = open(root+"\\"+name+".txt","w")
        file.write(r"\begin{table}"+"\n")
        file.write(r"\begin{tabular}{c"+"".join(list(np.repeat("c",cols)))+"}"+"\n")
        rows = int(np.ceil(self.K/cols))
        for r in range(rows):
            file.write(r"$Q \backslash$ M & ")
            for k in range(cols):
                if k+cols*r< self.K:
                    if k!= cols-1:
                        if labs is None:
                            file.write("$X_{"+str(k+1+cols*r)+"}$ &")
                        else:
                            file.write("$"+labs[k+cols*r]+"$ &")
                    else:
                        if labs is None:
                            file.write("$X_{"+str(k+1+cols*r)+"}$"+ r"\\"+"\n")
                        else:
                            file.write("$"+labs[k+cols*r]+"$" + r"\\"+"\n")
                else:
                    if k!= cols-1:
                        file.write("&")
                    else:
                        file.write(r"\\"+"\n")
            file.write(r"\hline"+"\n")
            for i in range(self.N):
                file.write(str(i+1)+"&")
                for k in range(cols):
                    if k+cols*r< self.K:
                        if k !=cols-1:
                            file.write(str(round(self.rho[i][k+cols*r],2))+"&")
                        else:
                            file.write(str(round(self.rho[i][k+cols*r],2))+r"\\""\n")
                    else:
                        if k !=cols-1:
                            file.write("&")
                        else:
                            file.write(r"\\"+"\n")
            file.write(r"\hline"+"\n")
        file.write(r"\end{tabular}" +"\n")
        file.write(r"\end{table}" +"\n")
        file.close()
        
        
class forback:
    def __init__(self):
        """
        forBack objects are useful for to compute for each dataset the latent probabilities and the forward-backward algorithm

        Returns
        -------
        None.

        """
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.psi = None
        self.phi = None
        self.mut = None
        self.probt = None
        self.pfit = None
        self.pgt = None
        self.ll = None
        self.numa = None
        self.dena = None
        self.matB = None
        self.coefmat = None
        self.coefvec =None
        self.numsig = None
        self.densig = None
        self.numrho = None
        self.denrho = None
        self.numepsi = None
        self.denepsi = None
        self.numtau = None
        self.dentau = None     
        
    def mu_t(self,O,B,G,N,p,P,K): 
        """
        Computes the temporal mean for each variable for each hidden state

        Parameters: see AsFS_AR_ASLG_HMM class for more information
        ----------

        Returns
        -------
        TYPE numpy array of size Nx(T-P)xK 
            DESCRIPTION.

        """
        T =len(O.T[0])
        mut = []
        for  i in range(N):
            Gi = G[i]
            mui = np.zeros([T-P,K])
            for k in range(K):
                pa_k = list(self.my_fathers(Gi,k).astype(int))
                acum = np.concatenate([np.ones([T-P,1]),((O.T[pa_k]).T)[P:]],axis=1)
                for j in range(1,p[i][k]+1):
                    x = np.roll(O.T[k],j)[P:]
                    x = x.reshape((len(x),1))
                    acum = np.concatenate([acum,x],axis=1)
                # mui.T[k] = np.sum(acum*(B[i][k].T),axis=1)
                mui.T[k] = np.dot(acum,B[i][k].T)
            mut.append(mui)
        self.mut = np.array(mut)
        
    def prob_BN_t(self,O,sigma,epsi,tau,rho,P):
        """
        Computes a set of temporal probabilities
            - full_p wich is the temporal emission probability
                full_p[t][i]   = log(P(x^t|Q^t=i))
            - dep_p which is the temporal probabilities of the relevant distribution for each variable
                dep_p[t][m] = log(P(x_m^t|Q^t=i,Z_m^t=1))
            - ind_p which is the temporal probabilities of the irrelevant distribution for each variable
                ind_p[t][i][m] = log(P(x_m^t|Q^t=i,Z_m^t=0))
            - mix_p which is the temporal mixture of relevancy probabilities for each variable
                mix_p[t][i][m] = log(P(x_m^t|Q^t=i))
                
        Parameters: see AsFS_AR_ASLG_HMM class for more information
        """
        T = len(O)
        full_p = []
        dep_p = []
        ind_p = []
        mix_p = []
        for t in range(T-P):
            [p,ft,gt,qt] = self.prob_BN(O[t+P],self.mut[:,t,:],sigma,epsi,tau,rho)
            full_p.append(p)
            dep_p.append(ft)
            ind_p.append(gt)
            mix_p.append(qt)
        self.probt = np.array(full_p)
        self.pfit = np.array(dep_p)
        self.pgt = np.array(ind_p)
        self.probtk =np.array(mix_p)
    
    def prob_BN(self,o,mu,sigma,epsi,tau,rho):  #Este es un punto critico
        """
        Computes a set of probabilities
            - p[i]    = log(P(x^t|Q^t=i))
            - f[i][m] = log(P(x_m^t|Q^t=i,Z_m^t=1))
            - g[m]    = log(P(x_m^t|Q^t=i,Z_m^t=0)) 
            - q[i][m] = log(P(x_m^t|Q^t=i))
            
        Parameters
        ----------
        o : TYPE numpy array of size K
            DESCRIPTION. Observation instance
        mu : TYPE numpy array of size NxK
            DESCRIPTION. mean of the relevant variables 
        sigma : TYPE numpy array of size NxK
            DESCRIPTION. standard deviations of the relevant variables
        epsi : TYPE numpy array of size K
            DESCRIPTION. mean of the irrelevant variables
        tau : TYPE numpy array of size K
            DESCRIPTION. standard deviations of the irrelevant variables

        Returns
        -------
        p : TYPE list of size 4
            DESCRIPTION. [p,f,g,q]
        """
        g  = self.prob_normal(o,epsi,tau)
        f  = self.prob_normal(o,mu,sigma)
        
        w_f  = (np.log(rho)+f)
        w_g  = (np.log(1-rho)+ g[np.newaxis])
        k = np.min(np.concatenate([w_f[np.newaxis],w_g[np.newaxis]],axis=0),axis=0)
        q1 = k +np.log(np.exp(w_f-k)+np.exp(w_g-k))     
        if np.sum(q1) == np.inf:            
            arg = rho*np.exp(f)+(1.-rho)*np.exp(g[np.newaxis]) 
            q2 = np.log(arg)
            q = self.mix_q(q1,q2)
        else:
            q=q1
              
        p = np.sum(q,axis=1)
        return [p,f,g,q]
    
    
    def mix_q(self,q1,q2):
        """
        

        Parameters
        ----------
        q1 : TYPE
            DESCRIPTION.
        q2 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        q = q1
        for i in range(len(q1)):
            for j in range(len(q1[0])):
                if q1[i][j] == np.inf:    
                    q[i][j] = q2[i][j]
        return q
                
                
    
    def checkzero(self,z):
        """
        Deletes zero instances in an array z
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)[:,0]
        y[inds] = np.exp(-740)
        return y
    
    
        
    def prob_normal(self,o,epsi,tau):
        """
        Computes thje log-probability from a noraml distribution

        Parameters
        ----------
        o : TYPE numpy array of size K
            DESCRIPTION. Observation instance
        epsi : TYPE numpy array 
            DESCRIPTION. mean of the variables 
        tau : TYPE numpy array 
            DESCRIPTION. standard deviations of the variables

        Returns
        -------
        p : TYPE numpy array of the same size of epsi
            DESCRIPTION. log probabilities 

        """
        g = -0.5*(np.log(2.*np.pi)+2*np.log(tau)+((o-epsi)/tau)**2)
        return g
    
    def prob_dep_ik(self,o,mu,sigma,i,k):
        """
        Computes the log of the emission probabilities of the relevant variables

        Parameters
        ----------
        o : TYPE numpy array of size K
            DESCRIPTION. Observation instance
        mu : TYPE numpy array of size NxK
            DESCRIPTION. mean of the variables 
        sigma : TYPE numpy array of size NxK
            DESCRIPTION. standard deviations of the variables
        i : TYPE int
            DESCRIPTION. Hidden state to compute its probability
        k : TYPE int
            DESCRIPTION. Variable to compute its probability

        Returns
        -------
        p : TYPE float
            DESCRIPTION. log probability of variable k at hidden state i

        """
        f = -0.5*(np.log(2.*np.pi)+2*np.log(sigma[i][k])+((o[k]-mu[i][k])/sigma[i][k])**2)
        return f
    
        
    def  forward_backward(self,A,pi,rho,O,N,P,ps=None): 
        """
        Does the scaled forward-backward algorithm using logarithms

        Parameters check AsFS_AR_ASLG_HMM for more information 
        ----------
        ps : TYPE, optional int
            DESCRIPTION. The default is None. index of the initial distibution
        """
        T = len(O)
        if ps==None:
            pi2 = self.checkzero(pi)
            alfa = np.log(pi2)+ self.probt[0]
        else:
            alfa = np.log(A[ps]) + self.probt[0]
        
        Cd = -np.max(alfa)-np.log(np.sum(np.exp(alfa-np.max(alfa))))
        Clist = np.array([Cd])
        Clist = np.array([Cd])
        alfa = alfa+Cd
        ALFA = [alfa]
        for t in range(1,T-P):
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
        for t in range(1,T-P):
            beta = self.backward_step(beta,A,T-t-P)
            beta = beta + Clist[t]
            BETA.append(beta)
        BETA= np.flipud(np.array(BETA))
        self.beta = BETA
        self.ll = Clist
        self.compute_gamma()
        self.compute_phi_psi(N,rho)
        

        
    def compute_phi_psi(self,N,rho):
        """
        Computes the psi and phi temporal probabilities

        Parameters
        ----------
        N : TYPE
            DESCRIPTION.
        rho : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.phi = []
        self.psi = []
        for i in range(N):
            psii =       rho[i] * np.exp(-self.probtk[:,i,:]+ self.pfit[:,i])*(self.gamma.T[i])[np.newaxis].T
            phii =  (1.-rho[i]) * np.exp(-self.probtk[:,i,:]+ self.pgt      )*(self.gamma.T[i])[np.newaxis].T
            psii = self.checkzero(psii)
            phii = self.checkzero(phii)
            self.phi.append(phii)
            self.psi.append(psii)         
        self.phi = np.array(self.phi)
        self.psi = np.array(self.psi)
        
        
    def compute_gamma(self):
        """
        Compute Gamma or the latent probabilities
        """
        num = np.exp(self.alpha +self.beta)
        den = np.sum(np.exp(self.alpha+self.beta),axis=1)[np.newaxis].T
        self.gamma = num/den
        # self.gamma_discrete()
        

        #####
    def gamma_discrete(self):
        T = len(self.gamma)
        N = len(self.gamma[0])
        for t in range(T):
            for i in range(N):
                if self.gamma[t][i] < 0.6:
                    self.gamma[t][i] = 0
                else:
                    self.gamma[t][i] = 1

        
        
        #####
    
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
    
    def forward_step(self,alfa,A,t):
        """
        Does an inductive step in the alfa variable
        alfa_t(i) = P(O1,..., Ot, q_t=i|lambda)

        Parameters check AsFS_AR_ASLG_HMM for more information 
        ----------
        alfa : TYPE numpy array of size N
            DESCRIPTION. alfa_t(i)
        t : TYPE int
            DESCRIPTION. time isntance

        Returns
        -------
        TYPE numpy array  of size N
            DESCRIPTION. [alfa_{t+1}(i)]_i

        """
        arg = np.dot(np.exp(alfa),A)
        arg = self.checkzero(arg)
        return self.probt[t]+ np.log(arg)  
    
    def backward_step(self,beta,A,t):
        """
        Does an inductive step in the beta variable
        beta_t(i) = P(Ot+1,..., OT| q_t=i,lambda)

        Parameters check AsFS_AR_ASLG_HMM for more information 
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
        arg =np.dot(A,np.exp(self.probt[t]+beta))
        arg = self.checkzero(arg)
        return np.log(arg)
    
    def act_aij(self,A,N,O,P): 
        """
        Updates the parameter A

        Parameters check AsFS_AR_ASLG_HMM for more information 
        ----------
        """
        T = len(O)-P
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
        
    def act_coef(self,G,O,p,P): 
        """
        Updates parameters from B

        Parameters check AsFS_AR_ASLG_HMM for more information 
        ----------
        """
        w = self.psi
        N = len(G)
        K = len(O[0])
        T = len(O)
        bc = []
        ac = []
        for i in range(N):
            Gi = G[i]
            wi = w[i]
            ack = []
            bck = []
            for k in range(K):
                pak = self.my_fathers(Gi,k)
                opa = (O[P:].T[pak])
                y = np.concatenate([np.ones([1,T-P]),opa],axis=0)
                for j in range(1,p[i][k]+1):
                    z = np.roll(O.T[k],j)[P:]
                    z = z.reshape((1,len(z)))
                    y = np.concatenate([y,z],axis=0)
                a = np.sum(wi.T[k]*(O.T[k])[P:]*y,axis=1)
                b = [np.sum(wi.T[k]*y,axis=1)]
                for pa in pak:
                    b.append(np.sum(wi.T[k]*(O[P:].T[pa])*y,axis=1))
                for j in range(1,p[i][k]+1):
                    zi = np.roll(O.T[k],j)[P:]
                    b.append(np.sum(wi.T[k]*zi*y,axis=1))
                b= np.array(b).T
                bck.append(b)
                ack.append(a)
            bc.append(bck)
            ac.append(ack)
        self.coefmat = bc
        self.coefvec = ac
    

    def act_sigma(self,O,N,p,ts):
        """
        Updates sigma parameter

        Parameters check AsFS_AR_ASLG_HMM for more information 
        """
        w= self.psi
        nums = []
        dens = []
        for i in range(N):
            if ts[i]==1:
                wi= w[i].T
                num = np.sum(wi*((O[p:]-self.mut[i]).T)**2,axis=1)
                den = np.sum(wi,axis=1)
                if np.sum(den <1e-10) > 1:
                    print("########## Problems with irrelevant hidden states ##############")
                nums.append(num)
                dens.append(den)
            else:
                nums.append(1)
                dens.append(1)
        self.numsig = np.array(nums)
        self.densig = np.array(dens)
        
    def act_rho(self): #Revisar con cuidado
        """
        Updates feature relevancies 

        Parameters check AsFS_AR_ASLG_HMM for more information 
        """
        self.numrho = np.sum(self.psi,axis=1)
        self.denrho = np.sum(self.gamma,axis=0)[np.newaxis].T

    def act_epsi(self,O,P):
        """
        Updates mean of irrelevant parameters

        Parameters check AsFS_AR_ASLG_HMM for more information 
        """
        self.numepsi = np.sum(np.sum(self.phi,axis=0)*O[P:],axis=0)
        self.denepsi = np.sum(np.sum(self.phi,axis=1),axis=0)
        
    def act_tau(self,O,epsi,P):
        """
        Updates variance of irrelevant variables

        Parameters check AsFS_AR_ASLG_HMM for more information 
        """
        self.numtau = np.sum(np.sum(self.phi*(O[P:]-epsi)**2,axis=1),axis=0)
        self.dentau = np.sum(np.sum(self.phi,axis=1),axis=0)

    