# -*- coding: utf-8 -*-
"""
@date: Created on Thu Jul 19 14:07:03 2018
@author: Carlos Puerto-Santana at Aingura IIoT, San Sebastian, Spain
@Licence: CC BY-NC-ND 4.0
"""
import pickle
import datetime
import os
import numpy as np
import networkx as nx
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
class AR_ASLG_HMM:    
    def __init__(self,O,lengths,N,p=None,P=None,A=None,pi=None,G=None,
                 B=None,sigma=None,p_cor=5,struc=True,lags=True):
        """
        Creates an object AR_ASLG_HMM
        Based on the paper:
            C. Puerto-Santana, P. Larranaga and C. Bielza, "Autoregressive Asymmetric 
            Linear Gaussian Hidden Markov Models," in IEEE Transactions on Pattern 
            Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2021.3068799.
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
        p_cor : TYPE, optional int
            DESCRIPTION. The default is 5.
        struc : TYPE, optional bool
            DESCRIPTION. The default is True. Performs structural optimization
        lags : TYPE, optional bool
            DESCRIPTION. The default is True. Performs time structure optimization
        """
        self.kappa   = None          
        self.nu      = None          
        self.struc   = struc    
        self.lags    = lags       
        self.mus     = None          # Exepcted value of the variables for each hidden state
        self.covs    = None          # Covariance matrices for each hidden state
        self.O       = np.array(O)   
        self.LogLtrain = None        # training data likelihood
        self.bic     = None          # Bayesian information criterion of training data
        self.b       = None          # Number of parameters
        self.forBack = []            # List of forBack objects (forward-backward and E-step are computed there )
        self.dictionary = None       # States order labelling
        self.K       = len(O[0])     # Number of variables
        self.N       = N             
        self.lengths = lengths            
        if P is None:
            self.P = self.det_lags(p_cor=p_cor)
        else:
            self.P = P
        if p is None:
            if lags == True:
                self.p = np.zeros([N,self.K]).astype(int)
            if lags == False:
                self.p = (self.P*np.ones([N,self.K])).astype(int)
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
                return "Invalid G"
            else:
                for i in range(self.N):
                    if np.prod(G[i].shape) != self.K**2 or len(G[i]) != self.K or  len(G[i].T) != self.K  :
                        return "Invalid G"
            for i in range(self.N):
                booli, Li = self.dag_v(G[i])
                if booli == True:
                    self.L.append(Li)
                else:
                    return "G at state "+ str(i) + " is not a DAG"  
               
        
        if B  is None:
            # kmeans = KMeans(n_clusters=self.N,n_init=100).fit(O)
            B= []
            for i in range(N):
                Bi = []
                # bi = kmeans.cluster_centers_[i]
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
            self.sigma = np.tile(np.sqrt(np.var(O,axis=0)),self.N).reshape([self.N,self.K])

        else:
            self.sigma = sigma
            
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
            nx.draw_networkx_labels(D, pos, labels=node_labels,font_size=lab_font_siz)
            edge_labels=dict([((u,v,),d['weight']) for u,v,d in D.edges(data=True)])
            nx.draw_networkx_edge_labels(D,pos,edge_labels=edge_labels,font_size=edge_font_siz)
            nx.draw(D,pos, node_size=node_siz,edge_cmap=plt.cm.Reds,arrows=True,node_color="silver",linewidths=2)
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
            b = b+np.sum(G[i])+2*self.K+ np.sum(p[i])
        b = b+self.N**2+self.N
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
        tb : TYPE boolean list 
            DESCRIPTION. indices to be updated of the parameter B
        ts : TYPE boolean list
            DESCRIPTION. indices to be updated of the parameter sigma

        Returns
        -------
        list
            DESCRIPTION. first component is local-likelihood, 
            second component is the learned B
            third component is the learned sigma
            fourth compoent is the new number of parameters 
        """
        B = self.act_B(G,p,tb)
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
                L = L+ self.forBack[i].gamma[t][j]*(self.forBack[i].prob_k_i((self.O[aux_len[i]:aux_len[i+1]])[t+self.P],self.forBack[i].mut[:,t,:],sigma,j,k))
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
                L = L+ np.sum((self.forBack[i].gamma[t]*self.forBack[i].prob_BN((self.O[aux_len[i]:aux_len[i+1]])[t+self.P],self.forBack[i].mut[:,t,:],sigma)))
        return [L+pena,npa]
    
    def score_complete_info_estimation(self,G,L,p,tb,ts): 
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
        tb : TYPE boolean list 
            DESCRIPTION. indices to be updated of the parameter B
        ts : TYPE boolean list
            DESCRIPTION. indices to be updated of the parameter sigma

        Returns
        -------
        list
            DESCRIPTION. first component is likelihood, 
            second component is the learned B
            third component is the learned sigma
            fourth compoent is the new number of parameters 
        """
        B = self.act_B(G,p,tb)
        self.act_mut(G,B,p)
        sigma = self.act_S(ts)
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
            if 1 == tb[i] or 1== ts[i]:
                for k in range(self.K):
                    sm = self.local_score(self.G,self.p,self.sigma,i,k)[0]
                    while self.p[i][k] +1 <= self.P :
                        p2 = np.copy(self.p)
                        p2[i][k] = p2[i][k] +1
                        [s2,B2,sigma2,b2] = self.score_complete_info_estimation_local(self.G,self.L,p2,i,k,tb,ts)
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
                if G2[i][j] != 1:
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
            if tb[i] ==1 or ts[i]==1 :
                for k in range(self.K):
                    possi = self.pos_ads(self.G[i])
                    son = possi[k][0]
                    if len(possi[k][1])!=0:
                        sm = self.local_score(self.G,self.p,self.sigma,i,son)[0]
                        for j in possi[k][1]:
                            G2 = np.copy(self.G)
                            G2[i][son][j] =1
                            L2 = []
                            for nn in range(self.N):
                                L2.append(self.dag_v(G2[nn])[1])
                            [s2,B2,sigma2,b2] = self.score_complete_info_estimation_local(G2,self.L,self.p,i,son,tb,ts)
                            if s2>sm: # Teorema: Irresoluble
                                sm= s2
                                self.B = B2
                                self.b = b2
                                self.sigma = sigma2
                                self.G = G2
                                self.L = L2

                                
        self.act_mut(self.G,self.B,self.p)
               
    def hill_climb(self,tb,ts):
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
        if self.lags is True:
            self.climb_ar(tb,ts)
        if self.struc is True:
            self.climb_struc(tb,ts)



            
                
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
            sigi = self.sigma[i]**2
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
                        # covi[s][k] = sigi[s] + np.sum(Bis[1:len(Pas)+1]*covi[Pas,s])
                        covi[k][s] = np.sum(Bis[1:len(Pas)+1]*covi[k,Pas])
                        covi[s][k] = covi[k][s]
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
        self.dictionary= [index,mag,]
        
    
    
#   HMM
    def log_likelihood(self,O,pena =False,xunit=False,ps = None):
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
        fb = forback()
        fb.mu_t(O,self.B,self.G,self.N,self.p,self.P,self.K)
        fb.prob_BN_t(O, self.sigma, self.P)
        fb.forward_backward(self.A,self.pi,O,self.P,ps=ps)
        Clist = fb.ll
        log = np.sum(-Clist)
        del fb
        if xunit == False:
            if pena == False:
                return log
            else:
                return -2*(log+self.pen(self.G,self.p,np.array([len(O)]))[0])
        else:
            if pena == False:
                return log/float(len(O))
            else:
                return (-2*(log+self.pen(self.G,self.p,np.array([len(O)]))[0]))/float(len(O))
        
        
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
    
    def viterbi(self,O,plot=True,maxg = False,absg = True,indexes=False,ps =None,xlabel="Time units",ylabel="Values"): 
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
        self.states_order(maxg=maxg,absg=absg)
        T =len(O)
        fb = forback()
        fb.mu_t(O,self.B,self.G,self.N,self.p,self.P,self.K)
        fb.prob_BN_t(O, self.sigma, self.P)
        if ps == None:
            delta = np.log(self.pi)+fb.probt[0]
        else:
            delta = np.log(self.A[ps])+fb.probt[0]
        psi = [np.zeros(len(self.pi))]
        for i in range(1,T-self.P):
            delta = self.viterbi_step(delta,fb,i)
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
            plt.figure("Sequence of labeled hidden states of AR-AsLG-HMM P:"+str(self.P),figsize=(4.5,3))
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
    
    
    def sample(self,leng,pseed):
        """
        Warning: not fully tested function
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
        Q = []
        data = pseed
        for t in range(leng):
            data_t = np.zeros(self.K)
            if t == 0:
                qt = np.argmax(np.random.multinomial(1,self.pi))
            else:
                qt = np.argmax(np.random.multinomial(1,self.A[qt]))
            Q.append(qt)
            for k in self.L[qt]:
                pak = self.my_fathers(self.G[qt],k)
                mean  = self.B[qt][k][0]
                if (len(pak) != 0):
                    mean = mean+ np.sum(data_t[pak]*self.B[qt][k][1:len(pak)+1])  
                if self.p[qt][k] != 0:
                    mean = mean+np.sum((data.T[k])[-self.p[qt][k]:]*self.B[qt][k][-self.p[qt][k]:])
                data_t[k] = np.random.normal(mean,self.sigma[qt][k])
            data_t = data_t[np.newaxis]
            data = np.concatenate([data,data_t],axis=0)
        return [data,Q]
    
    
    def sample_state(self,n,i,pseed=None):
        """
        

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.
        i : TYPE
            DESCRIPTION.
        pseed : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if pseed is None:
            pseed = self.O[:self.P]
        data = pseed
        for t in range(n):
            data_t = np.zeros(self.K)
            for k in self.L[i]:
                pak = self.my_fathers(self.G[i],k)
                mean  = self.B[i][k][0]
                if (len(pak) != 0):
                    mean = mean+ np.sum(data_t[pak]*self.B[i][k][1:len(pak)+1])  
                if self.p[i][k] != 0:
                    mean = mean+np.sum((data.T[k])[-self.p[i][k]:]*self.B[i][k][-self.p[i][k]:])
                data_t[k] = np.random.normal(mean,self.sigma[i][k])
            data_t = data_t[np.newaxis]
            data = np.concatenate([data,data_t],axis=0)
        return data
            
     
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
            self.forBack[i].forward_backward(self.A,self.pi,self.O[aux_len[i]:aux_len[i+1]],self.P,ps=ps)
    
    def act_pbnt(self,sigma,tb):
        """
        Updates the tmeporal probabilities of each forBack object
        
        Parameters
        ----------
        sigma : TYPE numpy array of size NxK
            DESCRIPTION. sigma values
        tb : TYPE boolean list 
            DESCRIPTION. indices to be updated of the parameter B
        """        
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],sigma,self.P)
        
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
                A[j] =self.A[j]
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

        Returns
        -------
        npi : TYPE numpy array of size N
            DESCRIPTION. The updated  initial distribution

        """
        api = self.forBack[0].gamma[0]
        for i in range(1,len(self.lengths)):
            api = api+self.forBack[i].gamma[0]
        npi =  api/len(self.lengths)
        ind = np.argwhere(npi==0).T[0]
        npi[ind] = np.exp(-740)
        return npi
    
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
        self.forBack[0].act_coef(G,self.O[aux_len[0]:aux_len[1]],p,self.P,tb)
        ac = self.forBack[0].coefvec
        bc = self.forBack[0].coefmat    
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_coef(G,self.O[aux_len[i]:aux_len[i+1]],p,self.P,tb)
            bc = bc+ self.forBack[i].coefmat
            ac = ac+ self.forBack[i].coefvec
        B= []
        for i in range(self.N):
            if tb[i] == 1:
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
        -------
        TYPE numpy array of size NxK
            DESCRIPTION. updated standard variances

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
                cov.append(covi)
            else:
                cov.append(self.sigma[i])
        cov = np.array(cov)
        inds = np.argwhere(cov<1e-10).T[0]
        cov[inds] =1e-10
        
        return np.array(cov)
    
    def EM(self,its=25,err=1e-3,ta="all", tpi=True, tb="all",ts = "all",ps=None,plot=False):
        """
        Computes the EM algorithm for parameter learning

        Parameters
        ----------
        its : TYPE, optional int
            DESCRIPTION. The default is 200. Number of iterations
        err : TYPE, optional float
            DESCRIPTION. The default is 1e-10. maximum error allowed in the EM iteration
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
        if ta == "all":
            ta = np.ones(self.N)
        if tb == "all":
            tb = np.ones(self.N)
        if ts == "all":
            ts = np.ones(self.N)
        likeli= -1e10
        eps =1
        it = 0
        del self.forBack[:]
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            fb = forback()
            fb.mu_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.G,self.N,self.p,self.P,self.K)
            fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.sigma,self.P)
            self.forBack.append(fb)
        while eps >err and it<its:
            self.act_gamma(ps=ps)
            self.A = self.act_A(ta)
            if tpi==True:
                self.pi = self.act_pi()
            self.B = self.act_B(self.G,self.p,tb)
            self.act_mut(self.G,self.B,self.p)
            self.sigma = self.act_S(ts)
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-self.forBack[i].ll)
            self.LogLtrain = likelinew
            for i in range(len(self.lengths)):
                self.forBack[i].prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.sigma,self.P)
            self.bic = -2*(likelinew + self.pen(self.G,self.p,self.lengths)[0])
            self.b = self.pen(self.G,self.p,self.lengths)[1]
            eps = np.abs(likelinew-likeli)
            likeli = likelinew
#            print likel
            if plot == True:
                for gg in range(self.K):
                    for ff in range(gg+1,self.K):
                        self.scatter_pair_kl(gg,ff)
            it = it+1   
        self.mvn_param()
        self.states_order()
                
    def SEM(self,err1=1e-2,err2=1e-2,its1=100,its2=100,ta="all",tpi=True, tb="all",ts="all",ps=None,plot=False): 
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
        eps = 1
        it = 0
        likelihood = [ ]
        if tb =="all":
            tb = np.ones(self.N)
        if ts =="all":
            ts = np.ones(self.N)
        if ta =="all":
            ta = np.ones(self.N)
        self.EM(its2,err2,ta=ta,tb=tb,tpi=tpi,ts=ts,ps=ps,plot=plot)
        likelihood.append(self.bic)
        while eps > err1 and it < its1:
            self.hill_climb(tb,ts)
            self.EM(its2,err2,ta=ta,tb=tb,tpi=tpi,ts=ts,ps=ps,plot=plot)
            eps = np.abs((self.bic-likelihood[-1]))
            likelihood.append(self.bic)
            it = it+1
            
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
            

    def scatter_pair_kl(self,k,l,samples=500):
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
        plt.figure("Scatter structure variable "+ "$X_"+str(k)+"$ and  variable "+ "$X_"+str(l))
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
        plt.pause(1)
        
        
    def plot_all_pairs_scatter(self,samples=500,name="",root=None):
        
        n_plots = self.K*(self.K-1)
        if n_plots<6:
            n_cols = n_plots
            n_rows = 1
        else:
            n_cols = 6
            n_rows = int(np.ceil(n_plots/6))
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
        

    def plot_AR_k_scatter(self,k,samples=500,name="",root=None):
        
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
        with open(root+"\\"+name+".ashmm", 'wb') as fp:
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
        if rootr[-6:] != ".ashmm":
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
        self.mut = None
        self.ll = None
        self.probt = None
        self.numa = None
        self.dena = None
        self.matB = None
        self.coefmat = None
        self.coefvec =None
        self.numsig = None
        self.densig = None
        
    def mu_t(self,O,B,G,N,p,P,K,eval_k =False): 
        """
        Computes the temporal mean for each variable for each hidden state

        Parameters: see AR_ASLG_HMM class for more information
        ----------
        eval_k : TYPE, optional bool
            DESCRIPTION. The default is False. determine if the means are returned or not

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
                mui.T[k] = np.sum(acum*(B[i][k].T),axis=1)
            mut.append(mui)
        if eval_k == False:
            self.mut = np.array(mut)
        else:
            return np.array(mut)
            
    
    def prob_BN(self,o,mu,sigma):  
        """
        Computes the log of the emission probabilities [P(O_t|q_t=i,lambda)]_i

        Parameters
        ----------
        o : TYPE numpy array of size K
            DESCRIPTION. Observation instance
        mu : TYPE numpy array of size NxK
            DESCRIPTION. mean of the variables 
        sigma : TYPE numpy array of size NxK
            DESCRIPTION. standard deviations of the variables

        Returns
        -------
        p : TYPE numpy array of size N
            DESCRIPTION. log probabilities for each hidden state 

        """
        p = np.sum(-0.5*(np.log(2*np.pi*sigma**2)+((o-mu)/sigma)**2),axis=1)
        return p
    
    def prob_k_i(self,o,mu,sigma,i,k):
        """
        Computes the log of the emission probabilities P(O_t=k|q_t=i,lambda)

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
        return -0.5*(np.log(2*np.pi*sigma[i][k]**2)+((o[k]-mu[i][k])/sigma[i][k])**2)
    
    def prob_BN_t(self,O,sigma,P):
        """
        Computes the temporal probabilities of the dataset O

        Parameters check AR_ASLG_HMM for more information 
        """
        T = len(O)
        B = []
        for t in range(T-P):
            B.append(self.prob_BN(O[t+P],self.mut[:,t,:],sigma))
        self.probt = np.array(B)
        
    def prob_BN_t_k(self,O,k,B,G,N,p,P,K,sigma):
        """
        Computes the temporal probabilities of variable k 

         Parameters check AR_ASLG_HMM for more information 
        ----------

        k : TYPE int
            DESCRIPTION. index of the desired variable

        Returns
        -------
        probas : TYPE numpy array of size TxN
            DESCRIPTION. Temporal probabilities of variable k

        """
        T = len(O)
        mus = self.mu_t(O,B,G,N,p,P,K,eval_k = True)
        probas =[]
        for t in range(T):
            p = self.prob_BN(O[t][k],(mus[:,k])[np.newaxis].T,(sigma[:,k])[np.newaxis].T)
            probas.append(p)
        return probas
    
    def log_likelihood_k(self,O,k,A,pi,B,G,N,p,P,K,sigma,ps=None):
        """
        Computes the likelihood of the model restricted to variable k 

         Parameters check AR_ASLG_HMM for more information 
        ----------

        k : TYPE int
            DESCRIPTION. index of the desired variable

        Returns
        -------
        probas : TYPE float
            DESCRIPTION. Likelihood  of variable k

        """
        T = len(O)
        probt = np.array(self.prob_BN_t_k(O,k,B,G,N,p,P,K,sigma))
        if ps == None:
            alfa = np.log(pi)    + probt[0]
        else:
            alfa = np.log(A[ps]) + probt[0]
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
        return -np.sum(np.array(Clist))
        
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
            alfa = np.log(pi) + self.probt[0]
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
        self.gamma = np.exp(self.alpha+self.beta)/np.sum(np.exp(self.alpha+self.beta),axis=1)[np.newaxis].T
        
        
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
                
            
        
    def act_coef(self,G,O,p,P,tb): 
        """
        Updates parameters from B

        Parameters check AR_ASLG_HMM for more information 
        ----------
        tb : TYPE, optional boolean list
            DESCRIPTION. The default is "all". indices to be updated of the parameter B
        """
        w = self.gamma
        N = len(G)
        K = len(O[0])
        T = len(O)
        bc = []
        ac = []
        for i in range(N):
            if tb[i] == 1:
                Gi = G[i]
                wi = w.T[i]
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
                    a = np.sum(wi*(O.T[k])[P:]*y,axis=1)
                    b = [np.sum(wi*y,axis=1)]
                    for pa in pak:
                        b.append(np.sum(wi*(O[P:].T[pa])*y,axis=1))
                    for j in range(1,p[i][k]+1):
                        zi = np.roll(O.T[k],j)[P:]
                        b.append(np.sum(wi*zi*y,axis=1))
                    b= np.array(b).T
                    bck.append(b)
                    ack.append(a)
                bc.append(bck)
                ac.append(ack)
            else:
                bc.append(1.)
                ac.append(1.)
        self.coefmat = bc
        self.coefvec = ac
    

    def act_sigma(self,O,N,p,ts):
        """
        Updates sigma parameter

        Parameters check AR_ASLG_HMM for more information 
        ----------

        ts : TYPE, optional boolean list
            DESCRIPTION. The default is "all". list that gives the indices of which hidden-states variances are updated 
        """
        w= self.gamma
        nums = []
        dens = []
        for i in range(N):
            if ts[i] == 1:
                wi= w.T[i]
                num = np.sum(wi*((O[p:]-self.mut[i]).T)**2,axis=1)
                den = np.sum(wi)
                nums.append(num)
                dens.append(den)
            else:
                nums.append(1)
                dens.append(1)
        self.numsig = np.array(nums)
        self.densig = np.array(dens)