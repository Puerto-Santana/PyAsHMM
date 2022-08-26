# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:30:24 2021
@author: Carlos Puerto-Santana at Aingura IIoT, San Sebastian, Spain
@Licence: CC BY 4.0
"""
import pickle
import datetime
import sys
import os
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import linprog
#%%
class CT_DHMM:
    def __init__(self, times, O, lengths, N, Q=None, pi=None, B=None,ltr=False):
        self.forBack = [] 
        self.N = N
        self.O = O
        self.tau = []
        self.lengths = lengths
        self.aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for l in range(len(self.lengths)):
            dal = times[self.aux_len[l]:self.aux_len[l+1]]
            self.tau.append(np.diff(dal))
        self.T = len(O)
        Os = []
        self.dictionary = {}
        for t in range(self.T):
            Os.append(str(O[t]))
        keys = list(set(Os))    
        self.K = len(keys)
        for k in range(self.K):
            self.dictionary[keys[k]] = k
        self.b = self.N+self.N**2+self.N*self.K
        if Q is None:
            if ltr == False:
                self.Q = np.log(N)*np.ones([N,N])
                for i in range(N):
                    self.Q[i][i] = -np.sum(np.ones(N-1)*np.log(N))
                self.Q = self.Q
            else: #asegurarse que qii = -sum(qij)
                self.Q = np.log(N)*np.ones([N,N])
                for i in range(N):
                    for j in range(i+1,N):
                        self.Q[j][i] = np.exp(-744) 
                for i in range(N):   
                     self.Q[i][i] = -np.sum(np.ones(N-1)*np.log(N))
                self.Q = self.Q
                
        else:
            self.Q = Q
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
        G is a list of graphs
        Returns a 2 index vector, where the first component is the penalty
            and the second is the number of parameters
        """
        T =np.sum(lengths)
        b = self.b
        return [-b*0.5*np.log(T),b]
    

    
    def prob_transi(self,Q,tau):
        ""
        A = expm(tau*Q)
        return A
    

    def log_likelihood(self,times,O,pena =False):
        """
        Computed the log likelihood of an observation O, it uses the learned parameters of the model.
        """
        fb = forback()
        tau = np.diff(times)
        T= len(O)
        fb.prob_BN_t(O,self.B,self.dictionary)
        fb.time_A(self.Q,tau)
        fb.forward_backward_As(self.pi, self.B, T)
        Clist = fb.ll
        log = np.sum(-Clist)
        del fb
        if pena ==False:
            return log
        else:
            return -2*(log+self.pen(np.array([T]))[0])
        
    def viterbi_step(self,fb,delta,o,A):
        """
        Paso para determinar sucesion de estados mas probable
        delta_t(i) = max_{q1,...,qt-1} P(q1,...,qt-1,O1,...,Ot,q_t=i|lambda)
        delta_t(i) = max_j [delta_t-1(j)a_ji]*log(P(o|q_t=i,lambda))
        A es la matriz de transiciones [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        o es una observacion de las K variables en el tiempo t
        retorna delta_t(i) 
        """
        return np.max(delta+np.log(A.T),axis=1)+fb.prob_BN(o,self.B,self.dictionary)
    
    def states_order(self):
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
                
            
        
    
    def viterbi(self,times,O,plot=True,indexes=False,xlabel="Time units",ylabel="Values"): 
        """
        Algoritmo para determinar sucesion de estados mas probable
        A es la matriz de transiciones [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        o es una observacion de las K variables en el tiempo t
        retorna sucesion de estados mas probable de forma ordenada usando el diccionario.
        """
        T =len(O)
        tau = np.diff(times)
        fb = forback()
        delta = np.log(self.pi)+fb.prob_BN(O[0],self.B,self.dictionary)
        psi = [np.zeros(len(self.pi))]
        for i in range(1,T):
            A = self.prob_transi(self.Q, tau[i-1])
            delta = self.viterbi_step(fb,delta,O[i],A)
            psi.append(np.argmax(delta+np.log(A.T),axis=1))
            
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
            plt.figure("Sequence of  hidden states by CT-DHMM")
            plt.clf()
            plt.title("Sequence of labeled hidden state")
            # plt.ylabel("Expected symbol" )
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.plot(times,Q[::-1],label="Segmentation")
            if len(self.labels)>0:
                plt.plot(times,O,label="Real")
                plt.legend(loc = 1 )
                plt.ylabel(ylabel)
            plt.grid("on")
            if indexes==True:
                plt.yticks(np.arange(self.N),np.arange(self.N))
            
            plt.show()
            plt.tight_layout()
            plt.pause(0.2)
        return Q[::-1]
    
    def act_gamma(self):
        """
        actualiza gamma de cada forback
        """
        for i in range(len(self.lengths)):
            self.forBack[i].time_A(self.Q,self.tau[i])
            self.forBack[i].forward_backward_As(self.pi,self.B,self.aux_len[i+1]-self.aux_len[i])
            
    def act_Q(self):
        """
        Actualiza el parametro Q, usando el end state probability (no del todo revisado)
        """
        self.forBack[0].compute_end_state(self.tau[0],self.Q)
        numa = self.forBack[0].n_t
        dena = self.forBack[0].e_t
        for i in range(1,len(self.lengths)):
            self.forBack[i].compute_end_state(self.tau[i],self.Q)
            numa = numa +self.forBack[i].n_t
            dena = dena +self.forBack[i].e_t 
        Q = np.ones([self.N,self.N])
        for j in range(self.N):
            Q[j] = numa[j]/dena[j]
        return Q
    
    def act_Q2(self):
        """
        Actualiza el parametro Q,  la probabilidad de transicion se asume una exponencial
        """
        self.forBack[0].act_Q(self.tau[0],self.N)
        numa = self.forBack[0].qnum
        dena = self.forBack[0].qden
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_Q(self.tau[i],self.N)
            numa = numa +self.forBack[i].qnum
            dena = dena +self.forBack[i].qden
        Q = np.ones([self.N,self.N])
        for j in range(self.N):
            Q[j] = numa[j]/dena[j]
            Q[j][j] = 0
            Q[j][j] = -np.sum(Q[j])
        return Q
    
    def act_Q3(self):
        """
        Actualiza el parametro Q, se maximiza la función auxiliar del EM usando programación lineal
        dado que la funcion auxiliar es lineal con respecto a Q.
        """
        c = np.zeros([self.N*self.N])
        for i in range(self.N):
            for j in range(self.N):
                for d in range(len(self.lengths)):
                    c[j+self.N*i] += np.sum(self.forBack[d].zita[:,i,j]*self.tau[d])
        b_eq = np.zeros([self.N])
        A_eq  = np.zeros([self.N,self.N*self.N])
        for i in range(self.N):
            A_eq[i][i*self.N:self.N+self.N*i] = 1
            A_eq[i][i+self.N*i]               =-1
        res = linprog(c,A_eq=A_eq,b_eq=b_eq,method ="revised simplex",x0 = self.Q.reshape([self.N*self.N]))
        Q =(res.x).reshape([self.N,self.N])
        return Q
        
    def act_pi(self):
        """
        Actualiza el parametro pi
        """
        api = self.forBack[0].gamma[0]
        for i in range(1,len(self.lengths)):
            api = api+self.forBack[i].gamma[0]
        return api/len(self.lengths)
    
    def act_B(self):
        """
        Actualiza el parametro B 
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


    
    def EM(self,its=100,err=1e-2,Qopt = "approx"): #Puede que tenga errores, toca revisar con calma
        """
        Realiza el algoritmo EM
        """
        likeli= -1e6
        eps =100
        self.it = 0
        del self.forBack[:]
        for i in range(len(self.lengths)):
            fb = forback()
            fb.prob_BN_t(self.O[self.aux_len[i]:self.aux_len[i+1]],self.B,self.dictionary)
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
            if (Qopt == "linear"):
                self.Q     = self.act_Q3()
            if (Qopt == "aprox"):
                self.Q     = self.act_Q2()
            self.pi    = self.act_pi()
            # print("it: "+str(self.it))
            self.B    = self.act_B()
            self.LogLtrain = likelinew
            self.bic   = likelinew + self.pen(self.lengths)[0]
            for i,fb in enumerate(self.forBack):
                fb.prob_BN_t(self.O[self.aux_len[i]:self.aux_len[i+1]],self.B,self.dictionary)
            likeli = likelinew
            self.it = self.it+1   
        self.bic = -2*self.bic
        self.states_order()
        
    def expected_value(self,tau,state):
        """
        time must be in timestamp format, the time must be with respecto to 
        timestamp=0
        """
        An= self.prob_transi(self.Q, tau)
        Ani = An[state]
        expected = 0
        for i in range(self.N):
            expected+= Ani[i]*self.labels[i]
        var = 0
        for i in range(self.N):
            var += Ani[i]*(self.labels[i]-expected)**2
        return [expected,var**0.5,Ani]
    
    def predict_s(self,times,state,plot=True,xlabel="Time units",ylabel ="Expected value"):
        """
        makes evolution of expected values assuming that it is always in the same hidden state
        t_final debe ser un int 
        """    
        if len(self.labels)>0:

            exs = []
            var = []
            for t in times:
                [et,vt,at] = self.expected_value(t, state)
                exs.append(et)
                var.append(vt)
            exs = np.array(exs)
            var =np.array(var)
            if plot==True:
                plt.figure("Prediction at hidden state " + str(state) +" by CT-DHMM")
                plt.clf()
                plt.title("Prediction at hidden state " + str(state) +" by CT-DHMM")
                plt.ylabel(ylabel)
                plt.xlabel(xlabel)
                plt.plot(times,exs+1.96*var,color="red", label = "Prediction Deviation")
                plt.plot(times,exs-1.96*var,color="red")
                plt.fill_between(times,exs+2*var,exs-2*var,color="red",alpha=0.2)
                plt.plot(times,exs,color="blue",label = ylabel)
                plt.ylim(ymin=0)
                plt.grid("on")
                plt.legend(loc=1)
                plt.tight_layout()
            return [exs,var] 
            
    def predict(self,times,plot=True,xlabel="Time units", ylabel = "E value"):
        """
        makes evolution of expected values assuming that it is always in the same hidden state
        t_final debe ser un int 
        """    
        if len(self.labels)>0:
            exs = []
            std = []
            for i in range(self.N):
                [ex,va] = self.predict_s(times, i,plot=False)
                exs.append(ex)
                std.append(va)
            if plot==True:
                plt.figure("Predictions by CT-DHMM")
                plt.clf()
                n_rows = int(np.ceil(self.N**0.5))
                n_columns = int(np.round(self.N/n_rows))
                for i in range(self.N):
                    plt.subplot(n_rows,n_columns,i+1)
                    plt.title("Prediction State: "+str(i)+" label: "+str(round(self.labels[i],2)),fontsize=10)
                    plt.ylabel(ylabel,fontsize=9)
                    plt.xlabel(xlabel,fontsize=9)
                    plt.plot(times,exs[i]+1.96*std[i],color="red", label = "Prediction Deviation")
                    plt.plot(times,exs[i]-1.96*std[i],color="red")
                    plt.fill_between(times,exs[i]+2*std[i],exs[i]-2*std[i],color="red",alpha=0.2)
                    plt.plot(times,exs[i],color="blue",label = ylabel)
                    plt.ylim(ymin=0)
                    plt.grid("on")
                plt.tight_layout()
            return [exs,std] 
        
    def prob_ev(self,times,j,plot=True,xlabel="Time units"):
            T = len(times)
            valores = np.zeros([T,self.N])
            for i,t in enumerate(times):
                valores[i] = self.prob_transi(self.Q,t)[j]
            if plot== True:
                plt.figure("CT-DHMM Evolution of probabilities of state " + str(j)+" label "+str(round(self.labels[j],2)))
                plt.clf()
                plt.title( "Evolution of probabilities of state " + str(j)+" label:"+str(round(self.labels[j],2)))
                for i in range(self.N):
                    if i!=j:
                        plt.plot(times,valores[:,i],label="Transit to " +str(round(self.labels[i],2)))
                    if i==j:
                        plt.plot(times,valores[:,i],label="Remain")
                plt.grid("on")
                plt.ylabel("Probability")
                plt.xlabel(xlabel)
                plt.legend(loc=1)
                plt.tight_layout()
                    
        
                
            
            
    
class forback:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.zita = None
        self.probt = None
        self.ll = None
        self.numa = None
        self.dena = None
        self.matB = None
        self.coefmat = None
        self.coefvec =None
        self.At = None
        self.qnum = None
        self.qden = None
      

    def time_A(self, Q,tau):
        At = []
        for t in range(len(tau)):
            At.append(self.prob_transi(Q, tau[t]))
        self.At = At
            
  
    def prob_BN_t(self,O,B,dictionary):
        """
        Calcula [P(O_t|qt=i)]_ti
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las desviaciones estandar para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        t0 es un tiempo inicial
        t1 es un tiempo final
        Retorna una  matriz B, donde B[t][i] = P(Ot|qt=i)
        """
        T = len(O)
        full_p = []
        for t in range(T):
            pt = self.prob_BN(O[t],B,dictionary)
            full_p.append(pt)
        self.probt = np.array(full_p)

    
    def prob_BN(self,o,B,dictionary):  #Este si toca cambiar 
        """
        Calcula [P(O_t|q_t=i,lambda)]_i
        o es una observacion, tiene longitud K, donde K es el numero de variables
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las desviaciones estandar para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        Retorna un vector de probabilidades de longitud N, donde N es el numero de
            estados
        """
        index = dictionary[str(o)]
        pt = B[:,index]
        return pt
    
    def prob_transi(self,Q,tau):
        ""
        A = expm(Q*tau)
        return A
     
    def forward_backward_As(self,pi,B,T): 
        """
        ALgortimo Forward Backward escladao.
        A es la matriz de transicion [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las desviaciones estandar para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        pi es vector de longitud N que determina la probabilidad de iniciar en estado i
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        Retorna 3 d-arrays, ALFA es una matriz, donde ALFA[t] es un vector de longitud N
            que determina el valor de la variable forward en el tiempo t, para los N estados.
            Similar sucede con la matriz BETA. Clist es el vector de escalamiento, 
            Clist permite calcular log-verosimilitudes.
        """
        N = len(pi)
        pi2 = self.checkzero(pi)
        alfa = np.log(pi2)+ self.probt[0]
        Cd = -np.max(alfa)-np.log(np.sum(np.exp(alfa-np.max(alfa))))
        Clist = np.array([Cd])
        Clist = np.array([Cd])
        alfa = alfa+Cd
        ALFA = [alfa]
        for t in range(1,T):
            A = self.At[t-1]
            alfa = self.forward_step_continuous(alfa,A,t)
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
            A = self.At[T-t-1]
            beta = self.backward_step_continuous(beta,A,T-t)
            beta = beta + Clist[t]
            BETA.append(beta)
        BETA= np.flipud(np.array(BETA))
        self.beta = BETA
        self.ll = Clist
        self.compute_gamma()
        self.gamma = np.exp(self.gamma)
        self.compute_zita(t,N)
        
        

    def compute_zita(self,T,N):
        num = np.zeros([T,N,N])
        for t in range(T-1):
            A = self.At[t]
            for i in range(N):
                for j in range(N):
                    alfat = self.alpha[t][i]
                    betat = self.beta[t+1][j]
                    num[t][i][j] = A[i][j]*np.exp(alfat+betat+self.probt[t][j])
            num[t] = num[t]/np.sum(num[t])
        self.zita = np.array(num)

    def compute_end_state(self,tau,Q): #TODO mirar aquí si va lento, esta muy mal programado
        N = len(Q)
        A1 = np.zeros([N,N])
        A2 = np.zeros([N,N])
        A1 = np.concatenate([Q,A1],axis=1)
        A2 = np.concatenate([A2,Q],axis=1)
        A = np.concatenate([A1,A2],axis=0)
        self.e_t = np.zeros(N)
        self.n_t = np.zeros([N,N])
        for i in range(N):
            for j in range(N):
                tau_aux = np.copy(A)
                n_aux  = np.copy(A)
                tau_aux[i,N+i] = 1
                n_aux[i,N+j] = 1
                Dij = 0
                Nij = 0
                for t in range(len(tau)):
                    Dij_temp = (self.prob_transi(tau_aux,tau[t]))[:N,N:2*N]
                    Dij += self.zita[t,i,j]*Dij_temp/((self.At[t])[i,j])
                    Nij += Q[i,j]*Dij
                if i==j :
                    self.e_t[i] = np.sum(Dij)
                else:
                    self.n_t[i][j] = np.sum(Nij)
                    
    def act_Q(self,tau,N):
        qnum = np.ones([N,N])
        qden = np.ones([N,N])
        for i in range(N):
            for j in range(N):
                if i != j:
                    qnum[i][j] = np.sum(self.zita[:,i,j])
                    qden[i][j] = np.sum(self.gamma[:,i]*tau)
        self.qnum = qnum
        self.qden = qden
            
        

    def compute_gamma(self):
        num = self.alpha +self.beta
        den = np.log(np.sum(np.exp(self.alpha+self.beta),axis=1))[np.newaxis].T
        self.gamma = num-den
    
    def forward_step_continuous(self,alfa,A,t,k_eval=False,prob=None):
        """
        Hace un paso inductivo en la variabe alfa
        alfa_t(i) = P(O1,..., Ot, q_t=i|lambda)
        al_t(i) = sum_j alfa_(t-1)(j)*a_ji*P(O_t|q_t=i)
        alfa es un vector que represente alfa_(t-1)
        A es la matriz de transicion [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        o es una observacion de las K variables en el tiempo t
        retorna alfa_t
        """
        if k_eval == False:
            arg = np.dot(np.exp(alfa),A)
            arg = self.checkzero(arg)
            return self.probt[t]+ np.log(arg)
        else:
            arg = np.dot(np.exp(alfa),A)
            arg = self.checkzero(arg)
            return prob[t]+ np.log(arg)   
    
    def backward_step_continuous(self,beta,A,t,k_eval=False,prob=None):
        """
        Hace un paso inductivo en la variabe beta
        beta_t(i) = P(Ot+1,..., OT| q_t=i,lambda)
        beta_t(i) = sum_i beta_(t+1)(i)*a_ji*P(O_t+1|q_t=i)
        beta es un vector que represente beta_(t+1)
        A es la matriz de transicion [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        o es una observacion de las K variables en el tiempo t
        retorna beta_t
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
    

                
    def act_B(self,Oi,dictionary):
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