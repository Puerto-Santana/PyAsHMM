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
class AR_MOG_RHMM:    
    def __init__(self,O,lengths,K,N,st,nC=None,C=None,P=None,A=None,pi=None,B=None,p_cor=5,nums=None,dens=None,nu=None,kappa=None):
        """
        Create an object AR-MOG-HMM
        O: the observations, it has dimension (n_1+n_2,+...+n_M)xd, 
            where n_m are the number of observations of the first dataset
            and d is the number of variables
        lengths: is a numpy array with the lengts of observations 
        N: the number of hidden States
        p: is a matrix NxK with the number of time lags per state and variable 
            p[i][k] is the number of lags for variable k in state i
        P: is the maximun number of time lags allowed
        A: the transition Matrix type numpy
        pi: the initial distribution vector type numpy
        G: is a list of lists, where each list is a graph
        L: is a list with numpy arrays, where each array is a topological order
            of the corresponding index graph
        B: is a list with lists, where each list has the linear dependies coefficients
            of the corresponding index graph.
        sigma: is a list of vectors, where each vector has got the variance coefficients
            of the corresponding index graph
        m: number of possible initial relationships within context bayesian graphs.
        """
        self.st = st 
        self.nu = None
        self.kappa = None
        self.covs = None
        self.mus  = None
        self.O = O #Observacion
        self.LogLtrain = None #Verosimilitud de los datos de entrenamiento
        self.bic = None
        self.b = None # lista de numero de parametros 
        self.forBack = [] #lista de objetos forBack
        self.dictionary = None  # Orden de los estados segun severidad
        self.K = K #Numero de variables
        self.N = N #Numero de estados
        self.lengths = lengths #vector con longitudes de las observaciones     
        
        if nC is None:
            self.nC = 1
        else:
            self.nC = nC #Numero de mixturas
        if P is None:
            self.P = self.det_lags(p_cor=p_cor)
        else:
            self.P = P
            
        if C is None:
            self.C = np.ones([N,nC]).astype("float")
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
        kmeans = KMeans(n_clusters=N).fit(O)
        clusts = kmeans.predict(O)
        if B  is None:
            B= []
            for i in range(N):
                indi = np.where(clusts==i)[0]
                Oi = O[indi]
                kmeansi = KMeans(n_clusters=nC).fit(Oi)
                clustsi = kmeansi.predict(Oi)
                Bi = []
                for j in range(nC):
                    indij = np.where(clustsi==j)[0]
                    Bij = np.mean(Oi[indij],axis=0)
                    Biv = []
                    for k in range(K):
                        Bivk = np.concatenate([[Bij[k]],np.zeros(self.P)])
                        Biv.append(Bivk)
                    Biv = np.array(Biv)
                    Bi.append(Biv)
                B.append(Bi)
            self.B = B
        else :
            self.B=B
            
        
    # Determining lags
            
    def correlogram(self,O,p=0):
        """
        """
        if p==0:
            p = int(len(O)/4.)
        n = len(O)
        y = O[::-1]
        mean = np.mean(y)
        var = [np.sum((y-mean)**2)/n]
        for k in range(1,p+1):
            ylag = np.roll(y,-k)
            num = np.sum((y[:-k]-mean)*(ylag[:-k]-mean))*(n-k)**(-1)
            var.append(num)
        rho = np.array(var)/var[0]
        return rho[1:]

    def pcorrelogram(self,O,p=0,title="",plot=True):
        """
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
        G is a list of graphs
        Returns a 2 index vector, where the first component is the penalty
            and the second is the number of parameters
        """
        T =np.sum(lengths)
        b = self.N*self.nC*(self.K*(1+self.P))+self.N**2+self.N
        return [-b*0.5*np.log(T),b]
#  States management
    def test_mvn(self,O,alpha):
        """
        Does the Mardia's skewness multivaribale normality test
        O is a nxd new observation O[i]= [O_1,...,O_d]
        return True if O does not correspond to any state mvn distribution
            False if O matches any state mvn distribution
        """
        n,d = O.shape
        tru = []
        for k in range(self.N):
            mu = self.mus[k]
            covin =self.covs[k]
            covin = np.linalg.inv(covin)
            b = 0
            for i in range(n):
                for j in range(n):
                    b = b + np.dot(np.dot(O[i]-mu,covin),O[j]-mu)**3
            b = b/(n**2)
            prueba = n*b
            critico = 6*stat.chi2.ppf(alpha,int(d*(d+1)*(d+2)/6.))
            if prueba > critico:
                tru.append(1)
            else:
                tru.append(0)
        tru = np.prod(np.array(tru))
        tru = np.prod(tru)
        if tru ==1:
            return True
        else:
            return False
        
    def label_parameters(self,kappa,nu):
        """
        Change the parameters for labelling
        """
        if len(kappa)!= self.K:
            return "Wrong dimension"
        self.kappa = kappa
        self.nu = nu
        
    def mvn_param(self):
        """
        """
        mu =[]
        for i in range(self.N):
            mui = []
            for k in range(self.K):
                muik = 0
                for j in range(self.nC):
                    if self.P>0:
                        bijk = self.B[i][j][k][0]/(1.-np.sum(self.B[i][j][k][1:]))
                    else:
                        bijk = self.B[i][j][k]
                    muik = muik + self.C[i][j]*bijk
                mui.append(muik)
            mu.append(mui)
        mu = np.array(mu)
        self.mus = np.array(mu)
        self.covs = np.zeros([self.N,self.K,self.K])
        for i in range(self.N):
            self.covs[i] = np.diag(np.diag(np.ones(self.K)))

    
    def states_order(self,maxg=False):
        """
        Order the states depending on the energy of the mvn mean vector taking into account AR coefficients
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
        Calcula la log verosimilitud de una observacion dado unos paramatros
        A matriz de transicion
        pi distribucion inicial
        mu lista de mu_i, donde mu_i son las medias para el estado i
        sigma lista de sigam_i donde sigma_i son las desviaciones para estado i
        O es matriz de observacion
        retorn log verosimilitud de la observacion
        """
        fb = forback()
        fb.mu_t(O,self.B,self.N,self.P,self.K,self.nC)
        fb.prob_t(O,self.B,self.N,self.P,self.K,self.nC,self.C,self.st)
        fb.forward_backward_As(self.A,self.pi,O,self.P,self.C)
        Clist = fb.ll
        log = np.sum(-Clist)
        del fb
        if xunit == False:
            if pena == False:
                return log
            else:
                return -2.*(log+self.pen(np.array([len(O)]))[0])
        else:
            if pena == False:
                return log/float(len(O))
            else:
                return (-2*(log+self.pen(np.array([len(O)]))[0]))/float(len(O))
     
    def viterbi_step(self,delta,A,fb,t):
        """
        Paso para determinar sucesion de estados mas probable
        delta_t(i) = max_{q1,...,qt-1} P(q1,...,qt-1,O1,...,Ot,q_t=i|lambda)
        delta_t(i) = max_j [delta_t-1(j)a_ji]*P(o|q_t=i,lambda)
        A es la matriz de transiciones [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        o es una observacion de las K variables en el tiempo t
        retorna delta_t(i) 
        """
        return np.max(delta+np.log(self.A.T),axis=1)+fb.prob_comp[t]
    
    def viterbi(self,O,plot=True,maxg=False,indexes=False): #esto esta bien?
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
        self.states_order(maxg=maxg)
        T =len(O)
        fb = forback()
        fb.mu_t(O,self.B,self.N,self.P,self.K,self.nC)
        fb.prob_t(O,self.B,self.N,self.P,self.K,self.nC,self.C,self.st)
        delta = np.log(self.pi)+fb.prob_comp[0]
        psi = [np.zeros(len(self.pi))]
        for i in range(1,T-self.P):
            delta = self.viterbi_step(delta,self.A,fb,i)
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
            plt.figure("Sequence of labeled hidden states of AR-MoG-HMM")
            plt.title("Sequence of labeled hidden state of AR-MoG-HMM")
            plt.ylabel("Magnitude of "+ r"$||\mu||$")
#            plt.ylabel("% Change respect to "+r"$||\mu_{\min}||_2 =$" +  str("{0:.2f}".format(np.min(np.sqrt(np.sum(np.array(self.mus)**2,axis=1))))))
            plt.xlabel("Time units")
            plt.plot(Q[::-1])
            plt.grid("on")
            plt.tight_layout()
#            plt.plot(Q)
            plt.show()
        return Q[::-1]
    
    def act_mut(self,B,aux_len):
        """
        actualiza mut de cada forback
        """
        for i in range(len(self.lengths)):
            self.forBack[i].mu_t(self.O[aux_len[i]:aux_len[i+1]],B,self.N,self.P,self.K,self.nC)
            
          
    def act_gamma(self,A,pi):
        """
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].forward_backward_As(A,pi,self.O[aux_len[i]:aux_len[i+1]],self.P,self.C)
            
            
    def act_A(self,A):
        """
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_aij(A,self.N,self.O[aux_len[0]:aux_len[1]],self.P)
        numa = self.forBack[0].numa
        dena = self.forBack[0].dena
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_aij(A,self.N,self.O[aux_len[i]:aux_len[i+1]],self.P)
            numa = numa +self.forBack[i].numa 
            dena = dena +self.forBack[i].dena 
        A = np.ones([self.N,self.N])
        for j in range(self.N):
            A[j] = numa[j]/dena[j]
        return A
        
    def act_pi(self):
        """
        """
        api = self.forBack[0].gamma[0]
        for i in range(1,len(self.lengths)):
            api = api+self.forBack[i].gamma[0]
        return api/len(self.lengths)
    
    def act_B(self,P):
        """
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
                Bij= []
                for k in range(self.K):
                    if np.prod(bc[i][j][k].shape)> 1:
                        Bijk = np.linalg.solve(bc[i][j][k], ac[i][j][k])
                        Bij.append(Bijk)
                    else:
                        Bij = (ac[i][j][k]/bc[i][j][k])[0]
                Bi.append(Bij)
            B.append(Bi)
        B = np.array(B)
        return B
        
    
    def act_C(self):
        """
        """
        self.forBack[0].act_mixt(self.N,self.nC)
        numc = self.forBack[0].nummix
        denc = self.forBack[0].denmix
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_mixt(self.N,self.nC)
            numc = numc+ self.forBack[i].nummix
            denc = denc+ self.forBack[i].denmix
        return numc/denc
    
    def EM(self,its=200,err=1e-10):
        """
        Realiza el algoritmo EM
        G es una lista tal que G_i es un gafo para el estado i
        L es una lista, donde L_i es el ordenamiento topologico del grafo Gi.
            L_i Se usa para dar el orden del calculo del parametro mu.
        A prior de matriz de transicion
        pi prior de distribucion incial
        mu prior de medias, es una lista, donde mu_i es las medias de las variables
            para el i-esimo estado
        sigma prior de desviaciones estandar, es una lista, donde sigma_i es las 
            desviacion de las variables para el i-esimo estado
        O es el conjunto de observaciones, donde O[t] tiene longitud K, y K es el
            numero de variables.
        Retorna los parametros optimizados
        """
        likeli= -1e6
        eps =1
        it = 0
        del self.forBack[:]
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            fb = forback()
            fb.mu_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K,self.nC)
            fb.prob_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K,self.nC,self.C,self.st)
            self.forBack.append(fb)
        while eps >err and it<its:
            self.act_gamma(self.A,self.pi)
            self.A = self.act_A(self.A)
            self.pi = self.act_pi()
            self.B = self.act_B(self.P)
            self.C = self.act_C()
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-self.forBack[i].ll)
            self.LogLtrain = likelinew
            self.bic = -2*(likelinew + self.pen(self.lengths)[0])
            self.b = self.pen(self.lengths)[1]
            for i,fb in enumerate(self.forBack):
                fb.mu_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K,self.nC)
                fb.prob_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K,self.nC,self.C,self.st)
            #Calculando error
            eps = np.abs(likelinew-likeli)
            likeli = likelinew
            it = it+1   
            print("EM-Epsilon:" + str(eps)+" with BIC: "+ str(likeli))
        self.mvn_param()
        self.states_order()
        if eps<err:
            self.converged = True

    def report(self,root=None,name = "CIG"):
        """
        Report all the models parameters    
        """
        if root == None:
            root = os.getcwd()
        F =open(root+"/param_"+name+".txt","w")
        F.write("AR_MOG parameters:"+"\n"+"\n")
        F.write("B: "+"\n")
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.B[i]) +"\n")
        F.write(r"sigma: "+"\n")
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.sigma[i]) +"\n")
        F.write("pi: "+"\n")
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.pi[i]) +"\n")
        F.write("A: "+"\n") 
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.A[i]) +"\n")
        F.write("C: "+"\n") 
        for i in range(self.N):
            F.write("state "+str(i)+": " +str(self.C[i]) +"\n")
        F.write("Dictionary: "+"\n") 
        F.write(str(self.dictionary) +"\n")   
        F.close()
        
    def save(self,root=None, name = r"rhmm_"+str(datetime.datetime.now())[:10]):
        """
        Saves the current model
        """
        if root == None:
            root = os.getcwd()
        itemlist = [self.N,self.K,self.P,self.nC,self.C,self.A,self.pi,self.B,
                    self.mus,self.dictionary,self.lengths,self.O,self.b,self.bic,self.LogLtrain]
        if self.nu is not None:
            itemlist.append(self.nu)
            itemlist.append(self.kappa)
        with open(root+"\\"+name+".armogrhmm", 'wb') as fp:
            pickle.dump(itemlist, fp)
    
    def load(self,rootr):
        """
        loads a model, must be a moghmm file 
        """
        if rootr[-10:] != ".armogrhmm":
            return "The file is not a moghmm file"
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
        self.mus = loaded[8]
        self.dictionary = loaded[9]
        self.lengths = loaded[10]
        self.O = loaded[11]
        self.b=loaded[12]
        self.bic = loaded[13]
        self.LogLtrain = loaded[14]
        if len(loaded)>15:
            self.nu = loaded[15]
            self.kappa= loaded[16]
    
        
class forback:
    def __init__(self):
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
        
    def prob_t(self,O,B,N,P,K,nC,C,st):
        """
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
                    probtti.append(self.probtiv(O[t+P],self.mut[i,:,v,t],K,st))
                maxpi = np.max(np.array(probtti))
                toadd = maxpi+ np.log(np.sum(np.exp(np.log(C[i])+probtti-maxpi)))
                probct.append(toadd)
                probtt.append(probtti)
            self.prob_comp.append(probct)
            self.probt.append(probtt)
        self.probt = np.array(self.probt)
        self.prob_comp = np.array(self.prob_comp)
                    
    def probtiv(self,o,mu,K,st):
        """
        Computes a normal distribution
        """
        oij = (o-mu)
        if K>1:
            deti = 1.
            invi = np.diag(np.ones(K))
        else:  
            deti = 1.
            invi = 1.
        logp = -0.5*(K*np.log(2.0*np.pi)+np.log(deti)+np.sum(np.log(st))+np.dot(np.dot(oij,invi),oij.T))
        return logp
    
    def mu_t(self,O,B,N,P,K,nC): 
        """
        Calcula los valores de medias para las observaciones usando los valores de B
        O es un set de datos de K variables en T tiempos O[t]= [o_1,...,o_k]
        B es una lista que tiene coeficientes de cada estado, B_i =[B_i1,...,B_ik]
        L es el orden topologico, solo se usa para ordenar el calculo.
        G es una lista de grafos de cada estado G=[G_1,...,G_N]
        retorna mu_t una lista para los valore de mu para cada estado, cada variable y tiempo
        mut tiene 4 dimensioanes: mut[N,nC,T-P,K]
        """
        T =len(O.T[0])
        mut = []
        acum = []
        for k in range(K):
            acumk = np.ones([T-P,1])
            for l in range(P):
                x = np.roll(O.T[k],l+1)[P:]
                x = x.reshape((len(x),1))
                acumk = np.concatenate([acumk,x],axis=1)
            acum.append(acumk)
        acum = np.array(acum)
        for i in range(N):
            muti =[]
            for k in range(K):
                mutik = []
                for j in range(nC):
                    mutikj = (np.dot(B[i][j][k],(acum[k]).T)).T
                    mutik.append(mutikj)
                muti.append(mutik)
            mut.append(muti)
        self.mut = np.array(mut)
                
    
    def forward_backward_As(self,A,pi,O,P,C): 
        """
        ALgortimo Forward Backward escladao.
        A es la matriz de transicion [a_ij]
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        pi es vector de longitud N que determina la probabilidad de iniciar en estado i
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        Retorna 3 d-arrays, ALFA es una matriz, donde ALFA[t] es un vector de longitud N
            que determina el valor de la variable forward en el tiempo t, para los N estados.
            Similar sucede con la matriz BETA. Clist es el vector de escalamiento, 
            Clist permite calcular log-verosimilitudes.
        """
        T = len(O)
        alfa = np.log(pi)+ self.prob_comp[0]
        Cd = -np.max(alfa)-np.log(np.sum(np.exp(alfa-np.max(alfa))))
        Clist = np.array([Cd])
        Clist = np.array([Cd])
        alfa = alfa+Cd
        ALFA = [alfa]
        for t in range(1,T-P):
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
        for t in range(1,T-P):
            beta = self.backward_step_continuous(beta,A,T-t-P)
            beta = beta + Clist[t]
            BETA.append(beta)
        BETA= np.flipud(np.array(BETA))
        self.beta = BETA
        self.ll = Clist
        self.gamma = self.alpha +self.beta -np.max(self.alpha+self.beta,axis=1)[np.newaxis].T -np.log(np.sum(np.exp(self.alpha+self.beta-np.max(self.alpha+self.beta,axis=1)[np.newaxis].T),axis=1))[np.newaxis].T
        self.mix_gam(O,C,P)
        
    def mix_gam(self,O,C,P):
        """
        """
        gammix = []
        for i in range(len(C)):
            gammi = self.gamma.T[i]
            gami = []
            for j in range(len(C.T)):
                gammij = np.exp(np.log(C[i][j])+gammi+self.probt[:,i,j]-self.prob_comp.T[i])
                gami.append(gammij)
            gammix.append(gami)
        self.ganmix = np.array(gammix)
        self.gamma = np.exp(self.gamma)
                
        
    def forward_step_continuous(self,alfa,A,t):
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
        maxi =np.max(alfa)
        logaa = np.log(np.dot(np.exp(alfa-maxi),A))
        return self.prob_comp[t] +maxi+logaa
    
    def backward_step_continuous(self,beta,A,t):
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
        maxi = np.max(beta)
        logba = np.log(np.dot(A,np.exp(self.prob_comp[t]+beta-maxi)))
        return maxi+logba
        
    def act_aij(self,A,N,O,P): #TODO hacerlo de forma totalmente vectorial
        """
        Actualiza la matriz de transicion
        A es la matriz de transicion
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        ALFA es una matriz, ALFA[t] determina los valores de las variables forward
            en el estado t para los N estados
        BETA es una matriz, BETA[t] determina los valores de las variables backward
            en el estado t para los N estados
        Retorna matriz de transicion actualizada
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
        Actualiza los coeficientes de la cl de mu para cada variable
        G es una lista tal que G_i es un gafo para el estado i
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        ALFA es una matriz, ALFA[t] determina los valores de las variables forward
            en el estado t para los N estados
        BETA es una matriz, BETA[t] determina los valores de las variables backward
            en el estado t para los N estados
        Retorna una lista B, donde Bi es una lista para el estado i, Bik posee los 
            coeficientes de mixtura de media de la variable k para el grafo i.
        """
        w = self.ganmix
        ac =[]
        bc = []
        K = len(O[0])
        T = len(O)
        acum = []
        for k in range(K):
            acumk = np.ones([1,T-P])
            for j in range(1,P+1):
                z = np.roll(O.T[k],j)[P:]
                z = z.reshape((1,len(z)))
                acumk = np.concatenate([acumk,z],axis=0)
            acum.append(acumk.T)
        acum = np.array(acum)
        
        for i in range(N):
            acj = []
            bcj = []
            for j in range(nC):
                wij = w[i][j]
                acjk = []
                bcjk = []
                for k in range(K):
                    a = np.sum((wij*O.T[k][P:])[np.newaxis].T*acum[k],axis=0)
                    b = wij[0]*np.outer(acum[k][0],acum[k][0])
                    for t in range(1,T-P):
                        b +=  wij[t]*np.outer(acum[k][t],acum[k][t])
                    bcjk.append(b)
                    acjk.append(a)
                bcj.append(bcjk)
                acj.append(acjk)
            ac.append(acj)
            bc.append(bcj)
        # for i in range(N):
        #     numi =[]
        #     deni =[]
        #     for j in range(nC):
        #         wij = w[i][j]
        #         numij= []
        #         denij = []
        #         for k in range(K):
        #             numijk = np.zeros([(1+P)])
        #             denijk = np.zeros([(1+P),(1+P)])
        #             for r in range(P+1):
        #                 if r == 0:
        #                     numijk[0] = np.sum(wij*(O.T[k])[P:])
        #                 else:
        #                     numijk[r] = np.sum(wij*(O.T[k])[P:]*(O.T[k])[P-r:-r])
        #                 for l in range(P+1):
        #                     if r == 0 and l!=0:
        #                         denijk[0][l] = np.sum(wij*(O.T[k])[P-l:-l])
        #                     if r== 0 and l == 0:
        #                         denijk[r][l] = np.sum(wij)
        #                     if r!= 0 and l != 0:
        #                         denijk[r][l] = np.sum(wij*(O.T[k])[P-r:-r]*(O.T[k])[P-l:-l])
        #                     if r!= 0 and l == 0:
        #                         denijk[r][l] = np.sum(wij*(O.T[k])[P-r:-r])
        #             numij.append(numijk)
        #             denij.append(denijk)
        #         numi.append(numij)
        #         deni.append(denij)
        #     ac.append(numi)
        #     bc.append(deni)
        self.coefmat = np.array(bc)
        self.coefvec = np.array(ac)
        

    def act_mixt(self,N,nC):
        """
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
            
        