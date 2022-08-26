# -*- coding: utf-8 -*-
"""
@date: Created on Mon Oct  4 16:30:12 2021
@author: Carlos Puerto-Santana at Aingura IIoT, San Sebastian, Spain
@Licence: CC BY 4.0 
"""
import pickle
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

class DHMM:
    def __init__(self,O,lengths,N,A=None, pi=None,B=None,ltr=False):
        """
        Creates an object DHMM
        Based on the paper:
            E. Baum and T. Petrie, “Statistical inference for probabilistic functions of finite state Markov chains,”Ann. Math. Stat.,

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
        A : TYPE, optional Numpy array of size NxN
            DESCRIPTION. The default is None. Transition matrix
        pi : TYPE, optional  Numpy array of size N
            DESCRIPTION. The default is None. Initial distirbution
        B : TYPE,  optional Numpy array of size NxK
            DESCRIPTION. The default is None.
        ltr : TYPE, bool
            DESCRIPTION. The default is False. assumes a ltr transition matrix
            
        Returns
        -------
        None.

        """
        self.forBack = [] 
        self.N = N
        self.O = O
        self.T = len(O)
        Os = []
        self.dictionary = {}
        for t in range(self.T):
            Os.append(str(O[t]))
        keys = list(set(Os))    
        self.K = len(keys)
        for k in range(self.K):
            self.dictionary[keys[k]] = k
                
        if A is None:
            if ltr==False:
                self.A = np.ones([N,N])/N
            else:
                self.A = np.ones([N,N])*np.exp(-744)
                for i in range(self.N):
                    self.A[i,i:] = 1
                    self.A[i] = self.A[i]/np.sum(self.A[i])
        else:
            self.A = A
        if pi is None:
            self.pi = np.ones(N)/N 
        else:
            self.pi = pi
        if B is None:
            B = np.ones([N,self.K])
            for i in range(N):
                Oi = O[int(i*self.T/(N)):int((i+1)*self.T/(N))]
                B[i] = np.log(self.prob_est(Oi))
            self.B=B
        else:
            self.B = B
            
        self.b = np.prod(self.A.shape)+np.prod(self.B.shape)+len(self.pi)
        self.bic = None
        self.forBack = []
        self.lengths = lengths
                
    def prob_est(self,Oi):
        T = len(Oi)
        temp_dic = {}
        keys = self.dictionary.keys()
        for k in keys:
            temp_dic[k] = 0
    
        for t in range(T):
            temp_dic[str(Oi[t])] += 1
            
        probs = np.ones(len(keys))*np.exp(-744)
        for j,k in enumerate(keys):
            if temp_dic[k] != 0:
                probs[j] = temp_dic[k]/float(T)
        return probs
    
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
        T= len(O)
        fb.prob_BN_t(O,self.B,self.dictionary)
        fb.forward_backward(self.A,self.pi,T)
        Clist = fb.ll
        log = np.sum(-Clist)
        del fb
        if pena ==False:
            return log
        else:
            return -2*(log+self.pen(np.array([T]))[0])
        
    def states_order(self):
        """
        Order the states depending on the magnitude of the expected value of each hidden state

        Parameters
        ----------
        maxg : TYPE, optional bool
            DESCRIPTION. The default is False.uses max function in the labeling
        absg : TYPE, optional bool
            DESCRIPTION. The default is True. uses absolute value in the labeling
        """
        self.labels = {}
        if (self.O.shape[1] ==1):
            for i in range(self.N):   
                mean  = 0
                for k in self.dictionary.keys():
                    index = self.dictionary[k]
                    val = k
                    val = val.replace('[','')
                    val = val.replace(']','')
                    val = float(val)
                    mean += val*np.exp(self.B[i][index])
                self.labels[i] = mean
                
    def viterbi_step(self,fb,delta,t):
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
    
    def viterbi(self,O,plot=True,indexes=False,xlabel="Time units",ylabel="Values"): 
        """
        Computes the most likely sequence of hidden states 

        Parameters
        ----------
        O : TYPE numpy array of size TxK
            DESCRIPTION. testing dataset
        plot : TYPE, optional bool
            DESCRIPTION. The default is True. 
        indexes : TYPE, optional bool
            DESCRIPTION. The default is False. Report indexes instead of labels

        Returns 
        -------
        TYPE numpy array of size T 
            DESCRIPTION. Most likely sequence of hidden states

        """
        T =len(O)
        fb = forback()
        fb.prob_BN_t(O, self.B, self.dictionary)
        delta = np.log(self.pi)+fb.probt[0]
        psi = [np.zeros(len(self.pi))]
        for i in range(1,T):
            delta = self.viterbi_step(fb,delta,i)
            psi.append(np.argmax(delta+np.log(self.A.T),axis=1))
            
        psi = np.flipud(np.array(psi)).astype(int)
        Q = [np.argmax(delta)]
        for t in np.arange(0,T-1):
            Q.append(psi[t][Q[t]])
        Q=np.array(Q)
        
        if indexes == False:
           Q = Q.astype(float)
           self.states_order()
           if len(self.labels)>0:
               for t in range(len(Q)):
                   Q[t] = self.labels[Q[t]]

        if plot==True:
            plt.figure("Sequence of  hidden states by DHMM")
            plt.clf()
            plt.title("Sequence of labeled hidden state")
            plt.ylabel("State magnitude" )
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            if len(self.labels)>0:
                plt.plot(O,label="Real")
                plt.ylabel(ylabel)
            plt.plot(Q[::-1],label ="Segmentation")
            plt.grid("on")
            plt.legend(loc = 1 )
            if indexes==True:
                plt.yticks(np.arange(self.N),np.arange(self.N))
            plt.show()
            plt.tight_layout()
            plt.pause(0.2)
        return Q[::-1]
    
    def act_gamma(self):
        """
        Updates for each forBack object its gamma parameter or latent probabilities
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].forward_backward(self.A,self.pi,aux_len[i+1]-aux_len[i])
    
    def act_A(self,A):
        """
        Updates for each forBack object its gamma parameter or latent probabilities
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_aij(A,self.O[aux_len[0]:aux_len[1]],self.N)
        numa = self.forBack[0].numa
        dena = self.forBack[0].dena
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_aij(A,self.O[aux_len[i]:aux_len[i+1]],self.N)
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
        Updates the parameter B
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_B(self.O[aux_len[0]:aux_len[1]],self.dictionary)
        ac = self.forBack[0].coefmat
        bc = self.forBack[0].coefvec    
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_B(self.O[aux_len[i]:aux_len[i+1]],self.dictionary)
            bc = bc+ self.forBack[i].coefvec
            ac2 = self.forBack[i].coefmat
            for j in range(self.N):
                for k in self.dictionary.keys():
                    ac[j][k]+= ac2[j][k]
        
        
        B= np.zeros(self.B.shape)
        for i in range(self.N):
            for k  in self.dictionary.keys():
                index = self.dictionary[k]
                Bik = ac[i][k]/bc[i]
                B[i][index] = Bik
            
        B = self.checkzero2d(B)
        B = B/np.sum(B,axis=1)[np.newaxis].T
        
        return np.log(B)
        
        
    def checkzero2d(self,z):
        """
        Check rho
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)
        for i in inds:
            y[i[0],i[1]] = np.exp(-744)
        return y

    def EM(self,its=100,err=1e-2): #Puede que tenga errores, toca revisar con calma
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
        eps =100
        self.it = 0
        del self.forBack[:]
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            fb = forback()
            fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.dictionary)
            self.forBack.append(fb)
        while eps >err and self.it<its:
            self.act_gamma()
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-self.forBack[i].ll)
            eps = likelinew-likeli
            if np.mod(self.it,1) == 0:
                print("it: " +str(self.it)+". EM error: " + str(eps) +" with log-likelihood: " + str(likelinew))
            if eps<0:
                self.bic   = likelinew + self.pen(self.lengths)[0]
                # break
            self.A     = self.act_A(self.A)
            self.pi    = self.act_pi()
            # print("it: "+str(self.it))
            self.B    = self.act_B()
            self.LogLtrain = likelinew
            self.bic   = likelinew + self.pen(self.lengths)[0]
            for i,fb in enumerate(self.forBack):
                fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.dictionary)
            likeli = likelinew
            self.it = self.it+1   
        self.bic = -2*self.bic
        self.states_order()
        
    def expected_value(self,epochs,state):
        """
        dado un horizonte de tiempo y asumiendo el estado oculto retorna 
        """
        if len(self.labels)>0:
            epochs = int(epochs)
            An= np.linalg.matrix_power(self.A, epochs+1)
            Ani = An[state]
            expected = 0
            for i in range(self.N):
                expected+= Ani[i]*self.labels[i]
            var = 0
            for i in range(self.N):
                var += Ani[i]*(self.labels[i]-expected)**2
            return [expected,var**0.5,Ani]
    
    def predict_s(self,t_final,state,plot=True,xlabel="Time units",ylabel="Expected value"):
        """
        makes evolution of expected values assuming that it is always in the same hidden state
        t_final debe ser un int 
        """    
        if len(self.labels)>0:
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
                plt.fill_between(range(t_final),exs+2*var,exs-2*var,color="red",alpha=0.2)
                plt.plot(exs,color="blue",label = ylabel)
                plt.ylim(ymin=0)
                plt.grid("on")
                plt.legend(loc=1)
                plt.tight_layout()
            return [exs,var] 
            
    def predict(self,t_final,plot=True,xlabel="Time units",ylabel="E value"):
        """
        makes evolution of expected values assuming that it is always in the same hidden state
        t_final debe ser un int 
        """    
        if len(self.labels)>0:
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
                    plt.title("Prediction State: "+str(i)+" label: "+str(round(self.labels[i],2)),fontsize=10)
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
            valores = np.zeros([T,self.N])
            for t in range(T):
                valores[t] = np.linalg.matrix_power(self.A,t)[j]
            if plot== True:
                plt.figure("DHMM Evolution of probabilities of state " + str(j)+" label "+str(round(self.labels[j],2)))
                plt.clf()
                plt.title( "Evolution of probabilities of state " + str(j)+" label:"+str(round(self.labels[j],2)))
                for i in range(self.N):
                    if i!=j:
                        plt.plot(valores[:,i],label="Transit to " +str(round(self.labels[i],2)))
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
        itemlist = [self.N,self.K,self.A,self.pi,self.B,
                    self.dictionary, self.labels,
                    self.b,self.bic,self.LogLtrain]
        with open(root+"\\"+name+".dhmm", 'wb') as fp:
            pickle.dump(itemlist, fp)
    
    def load(self,rootr):
        """
        Loads a model, must be an dhmm file 

        Parameters
        ----------
        rootr : TYPE str
            DESCRIPTION. direction of the saved model

        Returns
        -------
        str
            DESCRIPTION. in case of  wrong filename, returns error

        """
        if rootr[-5:] != ".dhmm":
            return "The file is not an dhmm file"
        with open(rootr, "rb") as  fp:
            loaded = pickle.load(fp)
        self.N          = loaded[0]
        self.K          = loaded[1]
        self.A          = loaded[2]
        self.pi         = loaded[3]
        self.B          = loaded[4]
        self.dictionary = loaded[5]
        self.labels     = loaded[6]
        self.b          = loaded[7]
        self.bic        = loaded[8]
        self.LogLtrain  = loaded[9]
            
    
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
        self.probt = None
        self.ll = None
        self.numa = None
        self.dena = None
        self.matB = None
        self.coefmat = None
        self.coefvec =None
     
        
  
    def prob_BN_t(self,O,B,dictionary):
        """
        Computes the temporal probabilities of the dataset O

        Parameters 
        ----------
        B : TYPE numpy array of size NxK
            DESCRIPTION. mean of the variables 
        dictionary : TYPE dictionary
            DESCRIPTION. changes observations to indexes to search in B
        """
        T = len(O)
        full_p = []
        for t in range(T):
            pt = self.prob_BN(O[t],B,dictionary)
            full_p.append(pt)
        self.probt = np.array(full_p)

    
    def prob_BN(self,o,B,dictionary):  #Este si toca cambiar 
        """
        Computes the log of the emission probabilities [P(O_t|q_t=i,lambda)]_i

        Parameters
        ----------
        o : TYPE numpy array of size K
            DESCRIPTION. Observation instance
        B : TYPE numpy array of size NxK
            DESCRIPTION. mean of the variables 
        dictionary : TYPE dictionary
            DESCRIPTION. changes observations to indexes to search in B

        Returns
        -------
        p : TYPE numpy array of size N
            DESCRIPTION. log probabilities for each hidden state 

        """
        index = dictionary[str(o)]
        pt = B[:,index]
        return pt
 
    def forward_backward(self,A,pi,T): 
        """
        Does the scaled forward-backward algorithm using logarithms

        Parameters check DHMM for more information 
        """
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
        self.gamma = np.exp(self.gamma)

    
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

        Parameters check DHMM for more information 
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

        Parameters check DHMM for more information 
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
        else:
            arg = np.dot(A,np.exp(prob[t]+beta))
            arg = self.checkzero(arg)
            return  np.log(arg)
            
    def checkzero(self,z):
        """
        Check rho
        """
        y = np.copy(z)
        inds = np.argwhere(y==0)
        y[inds] = np.exp(-740)
        return y
    
    
    def act_aij(self,A,O,N): 
        """
        Updates the parameter A

        Parameters check DHMM for more information 
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
                
    def act_B(self,Oi,dictionary):
        """
        Updates parameters from B

        Parameters check DHMM for more information 
        """
        T = len(Oi)
        w= self.gamma
        N = len(w[0])
        self.coefmat = []
        self.coefvec = []
        for i in range(N):
            temp_dic = {}
            keys = dictionary.keys()
            for k in keys:
                temp_dic[k] = 0
            for t in range(T):
                temp_dic[str(Oi[t])] += w[t][i]
            self.coefmat.append(temp_dic)
            self.coefvec.append(np.sum(w[:,i]))
        
            
        
        