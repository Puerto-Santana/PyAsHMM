# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:09:33 2021
@author: Carlos Puerto-Santana at Aingura IIoT, San Sebastian, Spain
@Licence: CC BY 4.0
"""
import pickle
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
class FS_HMM:    
    def __init__(self,O,lengths,N,rho= None,A=None, pi=None, mu=None,sigma=None,epsi = None, tau= None,act_inde = True,rho_var=0.8):
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
        rho : TYPE numpy array of size K
            DESCRIPTION relevancies parameters for each variable
        A : TYPE, optional Numpy array of size NxN
            DESCRIPTION. The default is None. Transition matrix
        pi : TYPE, optional  Numpy array of size N
            DESCRIPTION. The default is None. Initial distirbution
        mu : TYPE,  optional Numpy array of size NxK
            DESCRIPTION. The default is None.
        sigma : TYPE, optional  Numpy arrays of size NxK
            DESCRIPTION. The default is None. Standard deviations
        epsi : TYPE, optional Numpy array of size K
            DESCRIPTION. The default is None. means of irrelevant variables
        tau : TYPE, optional Numpy array of size K
            DESCRIPTION. The default is None. Standard deviations of irrelevant variables
        act_inde : TYPE, optional
            DESCRIPTION. The default is True.
        rho_bar : TYPE, optional float
            DESCRIPTION. The default is 0.9. relevance treshold between 0 and 1
            
        Returns
        -------
        None.

        """
        self.rho_bar=0.8
        self.act_inde = act_inde 
        self.kappa = None     # Valores estandar apra cada variable, se usa en Viterbi
        self.nu = None        # Valores de relevancia o escala de cada varaible   
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
            self.rho = 0.1*np.ones(self.K)
        else:
            self.rho = rho
        
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
        if mu is None:
            mu= []
            kmeans = KMeans(n_clusters=self.N,n_init=20).fit(O)
            for i in range(N):
                mui = kmeans.cluster_centers_[i]
                # mui = []
                # for k in range(self.K):
                #     muik = (i+1)*(np.max(O.T[k])-np.min(O.T[k]))/(self.N+1)+np.min(O.T[k])
                #     mui.append(muik)
                mu.append(mui)
            self.mu = np.array(mu)
        else :
            self.mu=mu
        if sigma is None:
            kmeans = KMeans(n_clusters=self.N,n_init=20).fit(O)
            ppl = kmeans.predict(O)
            sigma =[]
            for i in range(N):
                sigmai =[]
                Oi = O[np.argwhere(ppl==i).T[0]]
                for k in range(self.K):
                    sigmaik = np.std(Oi.T[k])
                    # sigmaik = np.abs(np.min(O.T[k])-np.max(O.T[k]))*2.0
                    sigmai.append(sigmaik)
                sigma.append(np.array(sigmai))
            sigma = np.array(sigma)
            self.sigma = sigma
        else:
            self.sigma = sigma
            
            
    def pen(self,lengths):
        """
        Calculates the penalty of a list of graphs

        Parameters
        ----------
        lengths : TYPE list
            DESCRIPTION. lengths of the observed data

        Returns
        -------
        list
            DESCRIPTION. First component is the BIC penalty, second is the number of parameters

        """
        T =np.sum(lengths)
        b = self.b
        return [-b*0.5*np.log(T),b]
    
    def mvn_param(self): 
        """
        Obtains the parameters of each multi dimensional normal distribution 
        corresponding to each state
        """
        mus = []
        covs = []
        for i in range(self.N):
            mui = self.mu[i]
            covi = np.diag(self.sigma[i])
            mus.append(mui)
            covs.append(covi)
        self.mus = np.array(mus)
        self.covs = np.array(covs)
            
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
        var = np.argwhere(self.rho>self.rho_bar)[:,0]
        if self.kappa is not None or self.nu is not None:
            if maxg== False and absg == False:
                music = np.sum((self.mus[:,var]-self.kappa[var])*self.nu[var],axis=1)
            if maxg== True and absg == False:
                music = np.max((self.mus[:,var]-self.kappa[var])*self.nu[var],axis=1)
            if maxg== False and absg == True:
                music = np.max(np.abs(self.mus[:,var]-self.kappa[var])*self.nu[var],axis=1)
            if maxg== True and absg == True:
                music = np.max(np.abs(self.mus[:,var]-self.kappa[var])*self.nu[var],axis=1)
        else:
            if maxg== False and absg == False:
                music = np.sum(self.mus[:,var],axis=1) 
            if maxg== True and absg == False:
                music = np.max(self.mus[:,var],axis=1)     
            if maxg== False and absg == True:
                music = np.sum(np.abs(self.mus[:,var]),axis=1) 
            if maxg== True and absg == True:
                music = np.max(np.abs(self.mus[:,var]),axis=1) 
        index = np.argsort(music)
        j=0
        for i in index:
            mag[j] = music[i]
            j=j+1
        self.dictionary= [index,mag]
        
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
        
    def log_likelihood(self,O,pena =False):
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
        fb.prob_BN_t(O,self.mu,self.sigma,self.epsi,self.tau,self.rho)
        fb.forward_backward(self.A,self.pi,self.rho,O,self.N)
        Clist = fb.ll
        log = np.sum(-Clist)
        del fb
        if pena ==False:
            return log
        else:
            return -2*(log+self.pen(np.array([len(O)]))[0])
        
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
    
    def viterbi(self,O,plot=True,maxg = False,absg = True,indexes=False,xlabel="Time units",ylabel="Values"): 
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
        self.states_order(maxg=maxg,absg=absg)
        T =len(O)
        fb = forback()
        fb.prob_BN_t(O, self.mu, self.sigma, self.epsi, self.tau, self.rho)
        delta = np.log(self.pi)+fb.probt[0]
        psi = [np.zeros(len(self.pi))]
        for t in range(1,T):
            delta = self.viterbi_step(delta,fb,t)
            psi.append(np.argmax(delta+np.log(self.A.T),axis=1))
            
        psi = np.flipud(np.array(psi)).astype(int)
        Q = [np.argmax(delta)]
        for t in np.arange(0,T-1):
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
    
    def act_gamma(self,A,mu,sigma,pi,epsi,tau,rho):
        """
        Updates for each forBack object its gamma parameter or latent probabilities
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].forward_backward(A,pi,rho,self.O[aux_len[i]:aux_len[i+1]],self.N)
    
    def act_A(self,A):
        """
        Updates for each forBack object its gamma parameter or latent probabilities
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_aij(A,self.N,self.O[aux_len[0]:aux_len[1]])
        numa = self.forBack[0].numa
        dena = self.forBack[0].dena
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_aij(A,self.N,self.O[aux_len[i]:aux_len[i+1]])
            numa = numa +self.forBack[i].numa 
            dena = dena +self.forBack[i].dena 
        A = np.ones([self.N,self.N])
        for j in range(self.N):
            A[j] = numa[j]/dena[j]
        return A
        
    def act_pi(self):
        """
        Updates the initial distribution parameters
        """
        api = self.forBack[0].gamma[0]
        for i in range(1,len(self.lengths)):
            api = api+self.forBack[i].gamma[0]
        return api/len(self.lengths)
    
    def act_B(self):
        """
        Updates the parameter mu
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_coef(self.O[aux_len[0]:aux_len[1]])
        ac = self.forBack[0].coefvec
        bc = self.forBack[0].coefmat    
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_coef(self.O[aux_len[i]:aux_len[i+1]])
            bc = bc+ self.forBack[i].coefmat
            ac = ac+ self.forBack[i].coefvec
        B= []
        for i in range(self.N):
            Bi = []
            for k in range(self.K):
                # print("ac:"+str(ac[i][k])+" bc:"+str(bc[i][k])+" i:"+str(i)+" k:"+ str(k))
                Bik = (ac[i][k]/bc[i][k])
                Bi.append(Bik)
            B.append(Bi)
        return np.array(B)
        
    def act_S(self):
        """
        It updates the sigma parameter

        Returns
        -------
        TYPE numpy array of size NxK
            DESCRIPTION. updated variances

        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_sigma(self.O[aux_len[0]:aux_len[1]],self.mu,self.N)
        nums = self.forBack[0].numsig
        dens = self.forBack[0].densig
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_sigma(self.O[aux_len[i]:aux_len[i+1]],self.mu,self.N)
            nums = nums+ self.forBack[i].numsig
            dens = dens+self.forBack[i].densig
        cov = []
        for i in range(self.N):
            covi = (nums[i]/dens[i])**0.5
            cov.append(covi)
        cov = np.array(cov)
        inds = self.reportzero(cov)
        if len(inds)>0:
            print("Warning: zero variances found in sigma")
            
        inds = np.argwhere(cov<1e-10).T[0]
        cov[inds] = 1e-10
            
        
        return cov
    
    def act_rho(self):
        """
        It updates the relevancy vector

        Returns
        -------
        rho2 : TYPE numpy array of size K
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
        rho2 = self.checkzero(rhoo)
        rho3 = np.copy(rho2)
        ind = np.argwhere(rho3==1)
        rho2[ind] = 1-0.00001
        return rho2
    
    
    def checkzero(self,z):
        """
        Check rho
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)
        y[inds] = np.exp(-740)
        return y
    
    def reportzero(self,z):
        """
        Check rho
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)
        return inds
    
    def act_epsi(self):
        """
        It updates the noise mean vector parameter

        Returns
        -------
        TYPE numpy array of size K
            DESCRIPTION. the updated mean of irrelevant variables

        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_epsi(self.O[aux_len[0]:aux_len[1]])
        nums = self.forBack[0].numepsi
        dens = self.forBack[0].denepsi
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_epsi(self.O[aux_len[i]:aux_len[i+1]])
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
        self.forBack[0].act_tau(self.O[aux_len[0]:aux_len[1]],self.epsi)
        nums = self.forBack[0].numtau
        dens = self.forBack[0].dentau
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_tau(self.O[aux_len[i]:aux_len[i+1]],self.epsi)
            nums = nums+ self.forBack[i].numtau
            dens = dens+self.forBack[i].dentau
        tau  = np.array(nums/dens)**(0.5)
        
        inds = np.argwhere(tau<1e-10).T[0]
        tau[inds] = 1e-10
        
        
        inds = self.reportzero(tau)
        if len(inds)>0:
            print("Warning: zero variances found in tau")
        return tau
    
    
    def EM(self,its=500,err=1e-2): #Puede que tenga errores, toca revisar con calma
        """
        Computes the EM algorithm for parameter learning

        Parameters
        ----------
        its : TYPE, optional int
            DESCRIPTION. The default is 200. Number of iterations
        err : TYPE, optional float
            DESCRIPTION. The default is 1e-10. maximum error allowed in the EM iteration

        """
        self.b = self.N**2+self.N+self.K+self.N*self.K*2+2*self.K
        likeli= -1e6
        eps =100
        self.it = 0
        del self.forBack[:]
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            fb = forback()
            fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.mu,self.sigma,self.epsi,self.tau,self.rho)
            self.forBack.append(fb)
        while eps >err and self.it<its:
            self.act_gamma(self.A,self.mu,self.sigma,self.pi,self.epsi,self.tau,self.rho)
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-self.forBack[i].ll)
            eps = likelinew-likeli
            if np.mod(self.it,100) == 0:
                print("it: " +str(self.it)+". EM error: " + str(eps) +" with log-likelihood: " + str(likelinew))
            if eps<0:
                self.bic   = likelinew + self.pen(self.lengths)[0]
                break
            self.rho   = self.act_rho()
            if self.act_inde == True:
                self.epsi  = self.act_epsi()
                self.tau   = self.act_tau()
            self.A     = self.act_A(self.A)
            self.pi    = self.act_pi()
            # print("it: "+str(self.it))
            self.mu    = self.act_B()
            self.sigma = self.act_S()
            self.LogLtrain = likelinew
            self.bic   = likelinew + self.pen(self.lengths)[0]
            self.b     = self.pen(self.lengths)[1]
            for i,fb in enumerate(self.forBack):
                fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.mu,self.sigma,self.epsi,self.tau,self.rho)
            likeli = likelinew
            self.it = self.it+1   
        self.mvn_param()
        self.states_order()
        self.bic = -2*self.bic
        
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
        itemlist = [self.N,self.K,self.rho,self.A,self.pi,self.mu,self.sigma,
                    self.epsi,self.tau,self.mus,self.covs,self.dictionary,
                    self.b,self.bic,self.LogLtrain,self.rho_bar]
        if self.kappa is not None:
            itemlist.append(self.nu)
            itemlist.append(self.kappa)
        with open(root+"\\"+name+".fshmm", 'wb') as fp:
            pickle.dump(itemlist, fp)
    
    def load(self,rootr):
        """
        Loads a model, must be an fshmm file 

        Parameters
        ----------
        rootr : TYPE str
            DESCRIPTION. direction of the saved model

        Returns
        -------
        str
            DESCRIPTION. in case of  wrong filename, returns error

        """
        if rootr[-6:] != ".fshmm":
            return "The file is not an fshmm file"
        with open(rootr, "rb") as  fp:
            loaded = pickle.load(fp)
        self.N          = loaded[0]
        self.K          = loaded[1]
        self.rho        = loaded[2]
        self.A          = loaded[3]
        self.pi         = loaded[4]
        self.mu         = loaded[5]
        self.sigma      = loaded[6]
        self.epsi       = loaded[7]
        self.tau        = loaded[8]
        self.mus        = loaded[9]
        self.covs       = loaded[10]
        self.dictionary = loaded[11]
        self.b          = loaded[12]
        self.bic        = loaded[13]
        self.LogLtrain  = loaded[14]
        self.rho_bar    = loaded[15]
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
        self.psi = None
        self.phi = None
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
        
  
    def prob_BN_t(self,O,mu,sigma,epsi,tau,rho):
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
        for t in range(T):
            [p,ft,gt,qt] = self.prob_BN(O[t],mu,sigma,epsi,tau,rho)
            full_p.append(p)
            dep_p.append(ft)
            ind_p.append(gt)
            mix_p.append(qt)
        self.probt = np.array(full_p)
        self.pfit = np.array(dep_p)
        self.pgt = np.array(ind_p)
        self.probtk =np.array(mix_p)
        
    def prob_BN(self,o,mu,sigma,epsi,tau,rho):  #Este si toca cambiar 
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
        f  = self.prob_normal(o,mu,sigma)
        g  = self.prob_normal(o,epsi,tau)
        arg = rho*np.exp(f)+(1.-rho)*np.exp(g)
        inds = self.reportzero(arg)
        for j in range(len(inds)):
            arg[inds[j][0]][inds[j][1]] = np.exp(-740)
        q = np.log(arg)
        p = np.sum(q,axis=1)
        return [p,f,g,q]
    
    def prob_normal(self,o,mu,sigma):
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
        p= -0.5*(np.log(2.*np.pi)+2*np.log(sigma)+((o-mu)/sigma)**2)
        return p
    
        
    def forward_backward(self,A,pi,rho,O,N): 
        """
        Does the scaled forward-backward algorithm using logarithms

        Parameters check AsFS_AR_ASLG_HMM for more information 
        """
        T = len(O)
        pi2 = self.checkzero(pi)
        alfa = np.log(pi2)+ self.probt[0]
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
        self.compute_gamma()
        
        self.phi = []
        self.psi = []
        for i in range(N):
            psii =  rho    *np.exp(-self.probtk[:,i,:]+ self.pfit[:,i]+(self.gamma.T[i])[np.newaxis].T)
            phii =  (1-rho)*np.exp(-self.probtk[:,i,:]+ self.pgt      +(self.gamma.T[i])[np.newaxis].T)
            psii = self.checkzero(psii)
            phii = self.checkzero(phii)
            self.phi.append(phii)
            self.psi.append(psii)         
        self.phi = np.array(self.phi)
        self.psi = np.array(self.psi)
        self.gamma = np.exp(self.gamma)
    
        indices_psi = np.argwhere(self.psi == float("inf"))
        indices_phi = np.argwhere(self.phi == float("inf"))

        self.psi[indices_psi.T[0]] = np.exp(-740)
        self.phi[indices_phi.T[0]] = np.exp(-740)
    
    def compute_gamma(self):
        """
        Compute Gamma
        """
        num = self.alpha +self.beta
        den = np.log(np.sum(np.exp(self.alpha+self.beta),axis=1))[np.newaxis].T
        self.gamma = num-den
    
    def forward_step(self,alfa,A,t,k_eval=False,prob=None):
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
        if k_eval == False:
            arg = np.dot(np.exp(alfa),A)
            arg = self.checkzero(arg)
            return self.probt[t]+ np.log(arg)
        else:
            arg = np.dot(np.exp(alfa),A)
            arg = self.checkzero(arg)
            return prob[t]+ np.log(arg)   
    
    def backward_step(self,beta,A,t,k_eval=False,prob=None):
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
        if k_eval== False:
            maxi = np.max(beta)
            arg =np.dot(A,np.exp(self.probt[t]+beta-maxi))
            arg = self.checkzero(arg)
            return  maxi+np.log(arg)
            # return np.dot(A,np.exp(np.log(beta)+self.probt[t]))
        else:
            arg = np.dot(A,np.exp(prob[t]+beta))
            arg = self.checkzero(arg)
            return  np.log(arg)
            
    
    def act_aij(self,A,N,O): 
        """
        Updates the parameter A

        Parameters check AsFS_AR_ASLG_HMM for more information 
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
        
    def act_coef(self,O):
        """
        Updates parameters from B

        Parameters check AsFS_AR_ASLG_HMM for more information 
        ----------
        """
        # print("psi: " +str(self.psi))
        self.coefvec = np.sum(self.psi*O,axis=1)
        self.coefmat = np.sum(self.psi,axis=1)
    
    def act_sigma(self,O,mu,N):
        """
        Updates sigma parameter

        Parameters check AsFS_AR_ASLG_HMM for more information 
        """
        w= self.psi
        nums = []
        dens = []
        for i in range(N):
            wi= w[i].T
            num = np.sum(wi*((O-mu[i]).T)**2,axis=1)
            den = np.sum(wi,axis=1)
            nums.append(num)
            dens.append(den)
        self.numsig = np.array(nums)
        self.densig = np.array(dens)
        
    def checkzero(self,z):
        """
        Check rho
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)
        y[inds] = np.exp(-740)
        return y
    
    def reportzero(self,z):
        """
        Check rho
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)
        return inds
    
    def act_rho(self):
        """
        Updates feature relevancies 

        Parameters check AsFS_AR_ASLG_HMM for more information 
        """
        self.numrho = np.sum(np.sum(self.psi,axis=0),axis=0)
        self.denrho = float(self.psi.shape[1])

    def act_epsi(self,O):
        """
        Updates mean of irrelevant parameters

        Parameters check AsFS_AR_ASLG_HMM for more information 
        """
        self.numepsi = np.sum(np.sum(self.phi,axis=0)*O,axis=0)
        self.denepsi = np.sum(np.sum(self.phi,axis=1),axis=0)
        
    def act_tau(self,O,epsi):
        """
        Updates variance of irrelevant variables

        Parameters check AsFS_AR_ASLG_HMM for more information 
        """
        self.numtau = np.sum(np.sum(self.phi*(O-epsi)**2,axis=1),axis=0)
        self.dentau = np.sum(np.sum(self.phi,axis=1),axis=0)



