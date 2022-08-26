"""
Created on Thu Jul 19 14:07:03 2018
@author: Carlos Puerto-Santana at Aingura IIoT, San Sebastian, Spain
@Licence: CC BY 4.0
"""
from sklearn.cluster import KMeans
import pickle
import math
import sys
_eps = sys.float_info.epsilon
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
class VAR_MOG_HMM:    
    def __init__(self,O,lengths,N,nC=None,C=None,P=None,A=None,pi=None,B=None,sigma=None,p_cor=5,diag=False):
        """
        Creates an object VAR_MOG_HMM
        Based on the paper:
            A. Poritz, "Linear predictive hidden markov models and the speech signal", in IEEE Xplore
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
        nC : TYPE, optinonal int
            DESCRIPTION. The default is None. Number of mixture components for each hidden state
        C : TYPE, optional Numpy array of size NxnC
            DESCRIPTION. The default is None. Mixture weights
        P : TYPE, optional int 
            DESCRIPTION. The default is None. Maximum permited lag
        A : TYPE, optional Numpy array of size NxN
            DESCRIPTION. The default is None. Transition matrix
        pi : TYPE, optional  Numpy array of size N
            DESCRIPTION. The default is None. Initial distirbution
        B : TYPE,  optional Numpy array of size NxnCxK
            DESCRIPTION. The default is None. mixture means by hidden state
        sigma : TYPE, optional  Numpy arrays of size NxnCxKxK
            DESCRIPTION. The default is None. Mixture covariance matrix by hidden state
        p_cor : TYPE, optional int
            DESCRIPTION. The default is 5.
        diag : TYPE, optional bool
            DESCRIPTION. diagonal covariance matrix
        """
        self.nu         = None
        self.kappa      = None
        self.covs       = None
        self.mus        = None
        self.O          = O #Observacion
        self.LogLtrain  = None #Verosimilitud de los datos de entrenamiento
        self.bic        = None
        self.b          = None # lista de numero de parametros 
        self.forBack    = [] #lista de objetos forBack
        self.dictionary = None  # Orden de los estados segun severidad
        self.K          = len(O[0]) #Numero de variables
        self.N          = N #Numero de estados
        self.lengths    = lengths #vector con longitudes de las observaciones     
        
        if nC is None:
            self.nC     = 1
        else:
            self.nC     = nC #Numero de mixturas
        if P is None:
            self.P      = self.det_lags(p_cor=p_cor)
        else:
            self.P      = P
            
        if C is None:
            self.C = np.ones([N,self.nC]).astype("float")
            normC = np.sum(self.C,axis=1)
            self.C = (self.C.T/normC).T
        else:
            self.C = C
            
        if  A is None:
            self.A = np.ones([N,N])
            normA = np.sum(self.A,axis=1)
            self.A = (self.A.T/normA).T
        else:
            self.A = A
        if pi is None:
            self.pi = np.ones(N)
            self.pi = self.pi/np.sum(self.pi)
        else:
            self.pi= pi
            
        
        #B_ij tiene tamanio Kx(1+KP)
        # kmeans = KMeans(n_clusters=N).fit(O)
        # clusts = kmeans.predict(O)
        if B  is None:
            B= []
            for i in range(N):
                # indi = np.where(clusts==i)[0]
                # Oi = O[indi]
                # kmeansi = KMeans(n_clusters=nC).fit(Oi)
                # clustsi = kmeansi.predict(Oi)
                Bi = []
                for j in range(self.nC):
                    # indij = np.where(clustsi==j)[0]
                    # Bij = np.mean(Oi[indij],axis=0)
                    Biv = []
                    for k in range(self.K):
                        # Bivk = np.concatenate([[Bij[k]],np.zeros(self.P*self.K)])
                        Bivk = np.concatenate([[(i+1)*(np.max(O.T[k])-np.min(O.T[k]))/(self.N+1)+np.min(O.T[k])],np.zeros(self.P*self.K)])
                        Biv.append(Bivk)
                    Biv = np.array(Biv)
                    Bi.append(Biv)
                B.append(Bi)
            self.B = B
        else :
            self.B=B
            
        if sigma is None:
            sigma =[]
            for i in range(N):
                sigmai = []
                for j in range(self.nC):
                    if diag == False: 
                        sigmaij = np.cov(O.T)
                    else:
                        sigmaij = np.diag(np.diag(np.cov(O.T)))
                    sigmai.append(sigmaij)
                sigma.append(sigmai)
            self.sigma = sigma
        else:
            self.sigma = sigma
        
        self.act_sigma_pro()
        
                
                
        
    # Determining lags
            
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
        """
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
        b = self.N*self.nC*(self.K*(1+self.P*self.K)+self.K*(self.K+1)/2.)+self.N**2+self.N
        return [-b*0.5*np.log(T),b]
    
 
#  States management
        
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
        
    def mvn_param(self):
        """
        Obtains the parameters of each multi dimensional normal distribution 
        corresponding to each state
        """
        mu =[]
        for i in range(self.N):
            for j in range(self.nC):
                bij = -self.B[i,j,:,0]
                aij = np.zeros([self.K,self.K])
                muij = np.zeros(self.K)
                for k in range(self.K):
                    for m in range(self.K):
                        if k!= m:
                            aij[k][m] = np.sum(self.B[i,j,k,1+m*self.P:1+(m+1)*self.P])
                        if k==m :
                            aij[k][k] = np.sum(self.B[i,j,k,1+k*self.P:1+(k+1)*self.P])-1.
                muij += self.C[i][j]*np.linalg.solve(aij,bij)
            mu.append(muij)
        self.mus = np.array(mu)
        self.covs = self.sigma
    
    def states_order(self,maxg=False,absg=False):
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
        if self.kappa is None or self.nu is None:
            music = []
            for i in  range(self.N):
                if maxg == False:
                    musici = np.sum(self.mus[i])
                else:
                    musici = np.max(self.mus[i])
                music.append(musici)
            music = np.array(music)
        else:
            music = []
            for i in  range(self.N):
                if maxg == False:
                    musici =  np.sum((self.mus[i]-self.kappa)*self.nu)
                else:
                    musici =  np.max((self.mus[i]-self.kappa)*self.nu)
                music.append(musici)
            music = np.array(music)
        index = np.argsort(np.asarray(np.squeeze(music)))
        j=0
        for i in index:
            mag[j] = music[i]
            j=j+1
        self.dictionary= [index,mag,]
        
    
    
#   HMM
    def log_likelihood(self,O,pena =False,xunit=False):
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

        Returns
        -------
        TYPE float
            DESCRIPTION. The score of the observations O

        """
        fb = forback()
        fb.mu_t(O,self.B,self.N,self.P,self.K,self.nC)
        fb.prob_t(O,self.B,self.N,self.P,self.K,self.nC,self.C,self.sigma,self.invsigma,self.detsigma)
        fb.forward_backward(self.A,self.sigma,self.pi,O,self.P,self.C)
        Clist = fb.ll
        log = np.sum(-Clist)
        del fb
        if xunit == False:
            if pena == False:
                return log
            else:
                return -2*(log+self.pen(np.array([len(O)]))[0])
        else:
            if pena == False:
                return log/float(len(O))
            else:
                return (-2*(log+self.pen(np.array([len(O)]))[0]))/float(len(O))
     
        
        
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
        return np.max(delta+np.log(self.A.T),axis=1)+fb.prob_comp[t]
    
    def viterbi(self,O,plot=True,maxg=False,indexes=False,xlabel="Time units",ylabel="Values"): #esto esta bien?
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
        self.states_order(maxg=maxg)
        T =len(O)
        fb = forback()
        fb.mu_t(O,self.B,self.N,self.P,self.K,self.nC)
        fb.prob_t(O,self.B,self.N,self.P,self.K,self.nC,self.C,self.sigma,self.invsigma,self.detsigma)
        delta = np.log(self.pi)+fb.prob_comp[0]
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
            plt.figure("Sequence of labeled hidden states of VAR-MoG-HMM")
            plt.title("Sequence of labeled hidden state of VAR-MoG-HMM")
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.plot(Q[::-1])
            plt.grid("on")
            plt.tight_layout()
            plt.show()
        return Q[::-1]
    
    
    def sample(self,leng,pseed):
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
        
    
    def act_mut(self,B):
        """
        Updates the mut ( i.e., the mean for each time instance, hidden state, mixture and variable,) parameter of each forBack object

        Parameters
        ----------
        B : TYPE list
            DESCRIPTION. list of lists with the emission parameters
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].mu_t(self.O[aux_len[i]:aux_len[i+1]],B,self.N,self.P,self.K,self.nC)
            
          
    def act_gamma(self,A,sigma,pi):
        """
        Updates for each forBack object its gamma parameter or latent probabilities
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].forward_backward(A,sigma,pi,self.O[aux_len[i]:aux_len[i+1]],self.P,self.C)
            
            
    def act_A(self):
        """
        Updates the parameter A

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
            A[j] = numa[j]/dena[j]
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
        return api/len(self.lengths)
    
    def act_B(self):
        """
        Updated the emission probabilities parameters

        Returns
        -------
        B : TYPE list
            DESCRIPTION. list with lits with the parameters of the emission probabilities

        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_coef(self.O[aux_len[0]:aux_len[1]],self.P,self.N,self.nC)
        ac = self.forBack[0].coefvec
        bc = self.forBack[0].coefmat    
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_coef(self.O[aux_len[i]:aux_len[i+1]],self.P,self.N,self.nC)
            bc = bc+ self.forBack[i].coefmat
            ac = ac+ self.forBack[i].coefvec
        
        B= []
        for i in range(self.N):
            Bi=[]
            for j in range(self.nC):
                if np.prod(bc[i][j].shape)> 1:
                    Bij = np.dot(ac[i][j],np.linalg.inv(bc[i][j]))
                else:
                    Bij = (ac[i][j]/bc[i][j])[0]
                # Bij= np.reshape(Bij,(self.K,self.P+1))
                Bi.append(Bij)
            B.append(Bi)
        B = np.array(B)
        return B
        
    def act_S(self):
        """
        It updates the sigma parameter

        Returns
        -------
        TYPE numpy array of size NxnCxKxK
            DESCRIPTION. updated standard variances

        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_sigma(self.O[aux_len[0]:aux_len[1]],self.N,self.P,self.nC)
        nums = self.forBack[0].numsig
        dens = self.forBack[0].densig
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_sigma(self.O[aux_len[i]:aux_len[i+1]],self.N,self.P,self.nC)
            nums = nums+ self.forBack[i].numsig
            dens = dens+self.forBack[i].densig
        cov = []
        for i in range(self.N):
            covi = []
            for j in range(self.nC):
                covij = nums[i][j]/dens[i][j]
                if self.K==1:
                    covi.append(np.array(covij[0][0]))
                else:
                    covi.append(covij)
            cov.append(covi)
        return cov
    
    def act_sigma_pro(self):
        """
        Computes covariance-matrices inverses and determinants
        """
        if self.K>0:
            invsigma = []
            detsigma = []
            for i in range(self.N):
                isigmai = []
                detsigmai = []
                for j in range(self.nC):
                    det = np.abs(np.linalg.det(self.sigma[i][j]))
                    inv = np.linalg.inv(self.sigma[i][j])
                    if det == 0 or math.isnan(det):
                        print("Singular covariance matrix found")
                    isigmai.append(inv)
                    detsigmai.append(det)
                invsigma.append(isigmai)
                detsigma.append(detsigmai)
            self.invsigma = invsigma
            self.detsigma = detsigma
    
    def act_C(self):
        """
        It Updates the mixture weights 

        Returns
        -------
        TYPE numpy array of size NxnC
            DESCRIPTION. mixture weigths

        """
        self.forBack[0].act_mixt(self.N,self.nC)
        numc = self.forBack[0].nummix
        denc = self.forBack[0].denmix
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_mixt(self.N,self.nC)
            numc = numc+ self.forBack[i].nummix
            denc = denc+ self.forBack[i].denmix
        return numc/denc
    
    
    def EM(self,its=200,err=1e-10): #Este modelo esta mal programado, toca mirar B si se esta haciendo correctamente.
        """
        Computes the EM algorithm for parameter learning

        Parameters
        ----------
        its : TYPE, optional int
            DESCRIPTION. The default is 200. Number of iterations
        err : TYPE, optional float
            DESCRIPTION. The default is 1e-10. maximum error allowed in the EM iteration
        """
        likeli= -1e6
        eps =1
        it = 0
        del self.forBack[:]
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            fb = forback()
            fb.mu_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K,self.nC)
            fb.prob_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K,self.nC,self.C,self.sigma,self.invsigma,self.detsigma)
            self.forBack.append(fb)
        while eps >err and it<its:
            self.act_gamma(self.A,self.sigma,self.pi)
            self.A = self.act_A()
            self.pi = self.act_pi()
            self.B = self.act_B()
            self.act_mut(self.B)
            self.sigma = self.act_S()
            self.act_sigma_pro()
            self.C = self.act_C()
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-self.forBack[i].ll)
            self.LogLtrain = likelinew
            self.bic = -2*(likelinew + self.pen(self.lengths)[0])
            self.b = self.pen(self.lengths)[1]
            for i,fb in enumerate(self.forBack):
                fb.prob_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K,self.nC,self.C,self.sigma,self.invsigma,self.detsigma)
            #Calculando error
            eps = np.abs(likelinew-likeli)
            likeli = likelinew
            it = it+1   
            print("EM-Epsilon:" + str(eps) +" with BIC: "+ str(likeli))
        self.mvn_param()
        self.states_order()
        if eps<err:
            self.converged = True
            
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

        
    def save(self,root=None, name = r"hmm_"+str(datetime.datetime.now())[:10]):
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
        itemlist = [self.N,self.K,self.P,self.nC,self.C,self.A,self.pi,self.B,self.sigma,
                    self.invsigma,self.detsigma,self.mus,self.covs,self.dictionary,self.b,self.bic,self.LogLtrain]
        if self.nu is not None:
            itemlist.append(self.nu)
            itemlist.append(self.kappa)
        with open(root+"\\"+name+".varmoghmm", 'wb') as fp:
            pickle.dump(itemlist, fp)
    
    def load(self,rootr):
        """
        Loads a model, must be an varmoghmm file 

        Parameters
        ----------
        rootr : TYPE str
            DESCRIPTION. direction of the saved model

        Returns
        -------
        str
            DESCRIPTION. in case of  wrong filename, returns error

        """
        if rootr[-10:] != ".varmoghmm":
            return "The file is not a varmoghmm file"
        with open(rootr, "rb") as  fp:
            loaded = pickle.load(fp)
        self.N = loaded[0]
        self.K = loaded[1]
        self.P = loaded[2]
        self.nC = loaded[3]
        self.C = loaded[4]
        self.A = loaded[5]
        self.pi = loaded[6]
        self.B =  loaded[7]
        self.sigma = loaded[8]
        self.invsigma = loaded[9]
        self.detsigma = loaded[10]
        self.mus = loaded[11]
        self.covs = loaded[12]
        self.dictionary = loaded[13]
        self.b=loaded[14]
        self.bic = loaded[15]
        self.LogLtrain = loaded[16]
        if len(loaded)>17:
            self.nu = loaded[17]
            self.kappa= loaded[18]
    
        
class forback:
    def __init__(self):
        """
        forBack objects are useful for to compute for each dataset the latent probabilities and the forward-backward algorithm

        Returns
        -------
        None.

        """
        self.probt = None
        self.prob_comp = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.ganmix = None
        self.mut = None
        self.ll = None
        self.numa = None
        self.dena = None
        self.coefmat = None
        self.coefvec =None
        self.numsig = None
        self.densig = None
        self.nummix = None
        self.denmix = None
        
    def prob_t(self,O,B,N,P,K,nC,C,sigma,isigma,detsigma):
        """
        Computes the temporal mean for each variable for each hidden state

        Parameters: see VAR_MOG_HMM class for more information
        
        Returns
        -------
        TYPE numpy array of size Nx(T-P)xK 
            DESCRIPTION.

        """
        T = len(O.T[0])
        self.probt = []
        self.prob_comp = []
        for t in range(T-P):
            probtt = []
            probct = []
            for i in range(N):
                probtti = []
                for v in range(nC):
                    probtti.append(self.probtiv(O[t+P],self.mut[i][v][t],sigma[i][v],isigma[i][v],detsigma[i][v],K))
                maxpi = np.max(np.array(probtti))
                toadd = maxpi+ np.log(np.sum(np.exp(np.log(C[i])+probtti-maxpi)))
                probct.append(toadd)
                probtt.append(probtti)
            self.prob_comp.append(probct)
            self.probt.append(probtt)
        self.probt = np.array(self.probt)
        self.prob_comp = np.array(self.prob_comp)
                    
    def probtiv(self,o,mu,sigma,isigma,detsigma,K):
        """
        Computes the log of the emission probabilities [P(O_t|q_t=i,lambda)]_i

        Parameters
        ----------
        o : TYPE numpy array of size K
            DESCRIPTION. Observation instance
        mu : TYPE numpy array of size K
            DESCRIPTION. mean of the variables 
        sigma : TYPE numpy array of size KxK
            DESCRIPTION. covariance matrix
        isigma : TYPE numpy array of size KxK
            DESCRIPTION. inverse of covariance matrix
        detsigma : TYPE numpy array of size 1
            DESCRIPTION. determinant of the covariance matrix
        Returns
        -------
        p : TYPE numpy array of size 1
            DESCRIPTION. log probabilities from a normal multivariate distribution

        """
        oij = (o-mu)
        if K>1:
            deti = np.abs(detsigma)
            invi = isigma
        else:  
            deti = sigma
            invi = 1./sigma
        logp = -0.5*(K*np.log(2.0*np.pi)+np.log(deti)+np.dot(np.dot(oij,invi),oij.T))
        return logp
    
    def mu_t(self,O,B,N,P,K,nC): 
        """
        Computes the temporal mean for each variable for each hidden state

        Parameters: see VAR_MOG_HMM class for more information

        Returns
        -------
        TYPE numpy array of size NxKxnCx(T-P)xK 
            DESCRIPTION.

        """
        T =len(O.T[0])
        mut = []
        acum = np.ones([T-P,1])
        for k in range(K):
            for l in range(P):
                x = np.roll(O.T[k],l+1)[P:]
                x = x.reshape((len(x),1))
                acum = np.concatenate([acum,x],axis=1)
        mut = []
        for i in range(N):
            muti =[]
            for j in range(nC):
                mutij = (np.dot(B[i][j],acum.T)).T
                muti.append(mutij)
            mut.append(muti)
        self.mut = mut
                
    
    def forward_backward(self,A,sigma,pi,O,P,C): 
        """
        Does the scaled forward-backward algorithm using logarithms

        Parameters check VAR_MOG-HMM for more information 
        """
        T = len(O)
        alfa = np.log(pi)+ self.prob_comp[0]
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
        self.gamma = self.alpha +self.beta -np.max(self.alpha+self.beta,axis=1)[np.newaxis].T -np.log(np.sum(np.exp(self.alpha+self.beta-np.max(self.alpha+self.beta,axis=1)[np.newaxis].T),axis=1))[np.newaxis].T
        self.mix_gam(O,C,sigma,P)
        
    def mix_gam(self,O,C,sigma,P):
        """
        Compute the latent probabilities of the mixture components
        
        Parameters check VAR_MOG-HMM for more information 
        """
        gammix = []
        for i in range(len(sigma)):
            gammi = self.gamma.T[i]
            gami = []
            for j in range(len(C.T)):
                logc = np.log(C[i][j])
                logg = gammi
                logpij = self.probt[:,i,j]
                logpi  = self.prob_comp.T[i]
                gammij = np.exp(logc+logg+logpij-logpi)
                gami.append(gammij)
            gammix.append(gami)
        self.ganmix = np.array(gammix)
        self.gamma = np.exp(self.gamma)
        # gammix = []
        # for i in range(len(sigma)):
        #     gammi = self.gamma.T[i]
        #     gami = []
        #     for j in range(len(C.T)):
        #         gammij = np.exp(np.log(C[i][j])+gammi+self.probt[:,i,j]-self.prob_comp.T[i])
        #         gami.append(gammij)
        #     gammix.append(gami)
        # self.ganmix = np.array(gammix)
        # self.gamma = np.exp(self.gamma)
                
        
    def forward_step(self,alfa,A,t):
        """
        Does an inductive step in the alfa variable
        alfa_t(i) = P(O1,..., Ot, q_t=i|lambda)

        Parameters check VAR_MOG_HMM for more information 
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
        maxi =np.max(alfa)
        logaa = np.log(np.dot(np.exp(alfa-maxi),A))
        return self.prob_comp[t] +maxi+logaa
        # return np.exp(np.log(np.dot(alfa,A))+self.prob_comp[t])
    
    def backward_step(self,beta,A,t):
        """
        Does an inductive step in the beta variable
        beta_t(i) = P(Ot+1,..., OT| q_t=i,lambda)

        Parameters check VAR_MOG_HMM for more information 
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
        logba = np.log(np.dot(A,np.exp(self.prob_comp[t]+beta-maxi)))
        return maxi+logba
        # return np.dot(A,np.exp(np.log(beta)+self.prob_comp[t]))
        
    def act_aij(self,A,N,O,P): #TODO hacerlo de forma totalmente vectorial
        """
        Updates the parameter A

        Parameters check VAR_MOG_HMM for more information 
        """
        T = len(O)-P
        bj = self.prob_comp
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
        
    def act_coef(self,O,P,N,nC): #TODO: este falta modificarlo
        """
        Updates parameters from B

        Parameters check VAR_MOG_HMM for more information 
        """
        w = self.ganmix
        ac =[]
        bc = []
        K = len(O[0])
        T = len(O)
        acum = np.ones([T-P,1])
        for k in range(K):
            for l in range(P):
                x = np.roll(O.T[k],l+1)[P:]
                x = x.reshape((len(x),1))
                acum = np.concatenate([acum,x],axis=1)
    
        for i in range(N):
            numi= []
            deni =[]
            for j in range(nC):
                wij = w[i][j]
                numij = wij[0]*np.dot((O[P])[np.newaxis].T,(acum[0])[np.newaxis])
                denij = wij[0]*np.dot((acum[0])[np.newaxis].T,acum[0][np.newaxis])
                for t in range(1,T-P):
                    numij += wij[t]*np.dot(O[P+t][np.newaxis].T,(acum[t])[np.newaxis])
                    denij += wij[t]*np.dot((acum[t])[np.newaxis].T,acum[t][np.newaxis])
                numi.append(numij)
                deni.append(denij)
            ac.append(numi)
            bc.append(deni)
        self.coefmat = np.array(bc)
        self.coefvec = np.array(ac)
    

    def act_sigma(self,O,N,p,nC): #TODO: falta modificar este tambien
        """
        Updates sigma parameter

        Parameters check VAR_MOG_HMM for more information 
        """
        w= self.ganmix
        K = len(O[0])
        nums = []
        dens = []
        for i in range(N):
            wi= w[i]
            numsi = []
            densi = []
            for j in range(nC):
                wij = wi[j]
                num = np.zeros([K,K])
                for t in range(len(O[p:])):
                    num = num+ wij[t]*np.outer(O[p:][t]-self.mut[i][j][t],(O[p:][t]-self.mut[i][j][t]).T)
                den = np.sum(wij)
                numsi.append(num)
                densi.append(den)
            nums.append(numsi)
            dens.append(densi)
        self.numsig = np.array(nums)
        self.densig = np.array(dens)

    def act_mixt(self,N,nC):
        """
        Updates the mixtur components
        
        Parameters check VAR_MOG_HMM for more information 
        """
        numix =[]
        denmix = []
        for i in range(N):
            numixi = []
            denmixi = []
            for j in range(nC):
                numixij  = np.sum(self.ganmix[i][j])
                denmixij = np.sum(self.ganmix[i])
                numixi.append(numixij)
                denmixi.append(denmixij)
            numix.append(numixi)
            denmix.append(denmixi)
        self.nummix = np.array(numix)
        self.denmix = np.array(denmix)