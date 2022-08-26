"""
Created on Thu Jul 19 14:07:03 2018
@author: Carlos Puerto-Santana at Aingura IIoT, San Sebastian, Spain
@Licence: CC BY 4.0
"""
from sklearn.cluster import KMeans
import itertools
import pickle
import datetime
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

class BMM:    
    def __init__(self,O,lengths,K,N,nC=None,C=None,P=None,A=None,pi=None,B=None,z=None,sigma=None,p_cor=5,nums=None,dens=None):
        """
        Create an object ASLGHMM_AR
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
        B: is a list of matrix of size Kx(P*K+1) that repre
        sigma: is a list of vectors, where each vector has got the variance coefficients
            of the corresponding index graph
        m: number of possible initial relationships within context bayesian graphs.
        """
        self.kappa = None
        self.nu    = None
        self.covs  = None
        self.mus   = None
        self.O    = O #Observacion
        self.LogLtrain = None #Verosimilitud de los datos de entrenamiento
        self.bic = None
        self.b = None # lista de numero de parametros 
        self.forBack = [] #lista de objetos forBack
        self.dictionary = None  # Orden de los estados segun severidad
        self.K = K #Numero de variables
        self.N = N #Numero de estados
        self.lengths = lengths #vector con longitudes de las observaciones     
        self.invsigma = None
        self.detsigma =None
        if nC is None:
            self.nC = 1
        else:
            self.nC = nC #Numero de mixturas

        
        if P is None:
#            self.P = self.det_lags(p_cor=p_cor)
            self.P = 0
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
                    Biv = np.concatenate([Bij[np.newaxis],np.zeros([self.P*K,K])],axis=0)
                    Bi.append(Biv.T)
                B.append(Bi)
            self.B = B
        else :
            self.B=B
            
        if z is None:
            z = []
            for i in range(N):
                z.append([])
            self.z = z
        else:
            self.z = z
            
        if sigma is None:
            sigma =[]
            for i in range(N):
                sigmai = []
                for j in range(nC):
                    sigmaij = np.cov(O.T)*10.0
                    sigmai.append(sigmaij)
                sigma.append(sigmai)
            self.sigma = sigma
        else:
            self.sigma = sigma
        self.act_sigma_pro(self.sigma)
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
        b = 0
        for i in range(self.N):
            for j in range(self.nC):
                b = b + len(np.where(self.B[i][j]!=0)[0])
        b = b + self.nC*self.N*self.K*(self.K+1)/2+self.N**2+self.N
        return [-b*0.5*np.log(T),b]
    
 
#  States management
                   
    def mvn_param(self):
        """
        """
        mu = []
        for i in range(self.N):
            mui = []
            zi = self.z[i]
            for j in range(self.nC):
                b = -self.B[i][j].T[0]
                a = -1.0*np.diag(np.ones(self.K))
                for l,z in enumerate(zi):
                    for k in range(self.K):
                        a[k][z[0]] += self.B[i][j][k][l] 
                muij = np.linalg.solve(a,b)
                mui.append(muij)
            mu.append(mui)
        self.mus = np.array(mu)
        self.covs = self.sigma
                
            
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
    
    def states_order(self,maxg=False):
        """
        Order the states depending on the energy of the mvn mean vector taking into account AR coefficients
        """
        mag = {}
        if self.kappa is None or self.nu is None:
            music = []
            for i in  range(self.N):
                musici = 0
                for j in range(self.nC):
                    if maxg == False:
                        musici = musici + np.sum((self.mus[i][j]*self.C[i][j]))
                    else:
                        musici = musici + np.max((self.mus[i][j]*self.C[i][j]))
                music.append(musici)
            music = np.array(music)
        else:
            music = []
            for i in  range(self.N):
                musici = 0
                for j in range(self.nC):
                    if maxg == False:
                        musici = musici + np.sum(((self.mus[i][j]-self.kappa)*self.nu)*self.C[i][j])
                    else:
                        musici = musici + np.max(((self.mus[i][j]-self.kappa)*self.nu)*self.C[i][j])
                music.append(musici)
            music = np.array(music)
        index = np.argsort(np.asarray(np.squeeze(music)))
        j=0
        for i in index:
            mag[j] = music[i]
            j=j+1
        self.dictionary= [index,mag]
        
    
    
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
        fb.mu_t(O,self.B,self.z,self.N,self.P,self.K,self.nC)
        fb.prob_BN_t(O,self.B,self.N,self.P,self.K,self.nC,self.C,self.sigma,self.invsigma,self.detsigma)
        fb.forward_backward_As(self.A,self.sigma,self.pi,O,self.P,self.C)
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
     
        
        
    def viterbi_step(self,delta,fb,o,t):
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
        return np.max(delta+np.log(self.A.T),axis=1)+ fb.prob_comp[t]
    
    def viterbi(self,O,plot=True,maxg=False,indexes=False): # Toca tocarlo
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
        fb.mu_t(O,self.B,self.z,self.N,self.P,self.K,self.nC)
        fb.prob_BN_t(O,self.B,self.N,self.P,self.K,self.nC,self.C,self.sigma,self.invsigma,self.detsigma)
        delta = np.log(self.pi)+fb.prob_comp[0]
        psi = [np.zeros(len(self.pi))]
        for i in range(1,T-self.P):
            delta = self.viterbi_step(delta,fb,O[i+self.P],i)
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
            plt.figure("Sequence of labeled hidden states of BMM")
            plt.title("Sequence of labeled hidden state of BMM")
            plt.ylabel("Magnitude of "+ r"$||\mu||$")
#            plt.ylabel("% Change respect to "+r"$||\mu_{\min}||_2 =$" +  str("{0:.2f}".format(np.min(np.sqrt(np.sum(np.array(self.mus)**2,axis=1))))))
            plt.xlabel("Time units")
            plt.plot(Q[::-1])
#            plt.plot(Q)
            plt.show()
        return Q[::-1]
        return Q[::-1]
    
    
    def sample(self,leng,ini=None):
#        ret = []
#        a = np.random.multinomial(1, self.pi, size=1)
        return 0
        
    
    def act_mut(self,B):
        """
        actualiza mut de cada forback
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].mu_t(self.O[aux_len[i]:aux_len[i+1]],B,self.z,self.N,self.P,self.K,self.nC)
            
    def act_gamma(self,A,sigma,pi):
        """
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        for i in range(len(self.lengths)):
            self.forBack[i].forward_backward_As(A,sigma,pi,self.O[aux_len[i]:aux_len[i+1]],self.P,self.C)
            
    def act_A(self,A,sigma):
        """
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_aij(A,self.N,sigma,self.O[aux_len[0]:aux_len[1]],self.P)
        numa = self.forBack[0].numa
        dena = self.forBack[0].dena
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_aij(A,self.N,sigma,self.O[aux_len[i]:aux_len[i+1]],self.P)
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
    
    def act_B(self,P,sigma):
        """
        """
        aux_len = np.concatenate([[0],self.lengths.cumsum()])
        self.forBack[0].act_coef(self.O[aux_len[0]:aux_len[1]],self.z,self.P,self.N,self.sigma)
        ac = self.forBack[0].coefvec
        bc = self.forBack[0].coefmat    
        for i in range(1,len(self.lengths)):
            self.forBack[i].act_coef(self.O[aux_len[i]:aux_len[i+1]],self.z,self.P,self.N,self.sigma)
            bc = bc+ self.forBack[i].coefmat
            ac = ac+ self.forBack[i].coefvec
        
        B= []
        for i in range(self.N):
            Bi=[]
            for j in range(self.nC):
                if np.prod(bc[i][j].shape)> 1:
                    Bij = np.dot(ac[i][j],np.linalg.inv(bc[i][j]))
                else:
                    Bij = (ac[i][j]/bc[i][j])
                Bi.append(Bij)
            B.append(Bi)
        return B
        
    def act_S(self):
        """
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
    
    def act_sigma_pro(self,sigma):
        """
        """
        if self.K>0:
            invsigma = []
            detsigma = []
            for i in range(self.N):
                isigmai = []
                detsigmai = []
                for j in range(self.nC):
                    det = np.abs(np.linalg.det(sigma[i][j]))
                    inv = np.linalg.inv(sigma[i][j])
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
            fb.mu_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.z,self.N,self.P,self.K,self.nC)
            fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K,self.nC,self.C,self.sigma,self.invsigma,self.detsigma)
            self.forBack.append(fb)
        while eps >err and it<its:
            self.act_gamma(self.A,self.sigma,self.pi)
            self.A = self.act_A(self.A,self.sigma)
            self.pi = self.act_pi()
            self.B = self.act_B(self.P,self.sigma)
            self.act_mut(self.B)
            self.sigma = self.act_S()
            self.act_sigma_pro(self.sigma)
            self.C = self.act_C()
            likelinew = 0
            for i in range(len(self.lengths)):
                likelinew = likelinew + np.sum(-self.forBack[i].ll)
            self.LogLtrain = likelinew
            self.bic = -2*(likelinew + self.pen(self.lengths)[0])
            self.b = self.pen(self.lengths)[1]
            for i in range(len(self.lengths)):
                fb.prob_BN_t(self.O[aux_len[i]:aux_len[i+1]],self.B,self.N,self.P,self.K,self.nC,self.C,self.sigma,self.invsigma,self.detsigma)
            self.forBack.append(fb)
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
        
    def save(self,root=None, name = r"hmm_"+str(datetime.datetime.now())[:10]):
        """
        Saves the current model
        """
        if root == None:
            root = os.getcwd()
        itemlist = [self.N,self.K,self.P,self.nC,self.C,self.A,self.pi,self.B,self.sigma,self.z
                    ,self.mus,self.covs,self.dictionary,self.lengths,self.O,self.b,self.bic,self.LogLtrain]
        if self.nu is not None:
            itemlist.append(self.nu)
            itemlist.append(self.kappa)
        with open(root+"\\"+name+".bmm", 'wb') as fp:
            pickle.dump(itemlist, fp)
        return None
    
    def build_dependencies(self,tu=0,Nq=20,tq=2.25e-3,tc=6.1e-3,tg=0.75):
        """
        busca por pares la mejor estructura para cada variable para cada estado
        la busqueda solo es entre cortes de tiempo no al interior de ellos.
        """
        [P,mi,fmi] = self.f_dependencies()
        print("Mutual informations computed")
        z = []
        dic = []
        for kd in range(self.K):
            for pd in range(P):
                dic.append([kd,pd+1])
        for i in range(self.N):
            zi = []
            for k in range(self.K):
                zik =[]
                miik = mi[k,:,i] 
                fiik = fmi[k]
                uik = mi[k,:,i]-fmi[k] 
                indexes = np.argsort(uik)[::-1]
                addi = 0
                iterator = 0
                while(addi<=Nq and uik[indexes[iterator]]>tu):
                    agregar = True
                    j = indexes[iterator]
                    varzj = dic[j][0]
                    pzj  = dic[j][1]
                    if  miik[j]<=tq and fiik[j]>=tc :
                        agregar = False
                    else:
                        for zet in zik:
                            zl = zet[0] 
                            zp = zet[1]
                            muti = self.mutual_information(varzj,pzj,zl,zp)[i]
                            if muti >= tg*miik[j] :
                                agregar = False    
                    if agregar == True:
                        zik.append(dic[j])
                        addi= addi+1
                    iterator = iterator+1
                    if iterator >= self.K*P:
                        break
                zi.append(zik)
            z.append(zi)
        return [P,z]
    
    def build_Bim(self):
        """
        Builds the new self.B parameter, it is assumed that the current BMM is a
        HMM with no intertemporal relationships, reset all the learned parameters.
        """
        [P,ff] = self.build_dependencies()
        B = []
        z = []
        for i in range(self.N):
            Bi =[]
            zi = []
            for k in range(self.K):
                for l in ff[i][k]:
                    zi.append(l)
            zi.sort()
            zi = list(zi for zi,_ in itertools.groupby(zi))
            lsi = zi
            si = len(list(zi))
            for j in range(self.nC):
                Biv = (i+1)*(np.max(self.O,axis=0)-np.min(self.O,axis=0))/(self.N+1)+np.min(self.O,axis=0)*np.random.uniform(0.9,1.1)
                Biv = Biv[np.newaxis]
                Biv = np.concatenate([Biv,np.zeros([si,self.K])])
                for k in range(self.K):
                    for zb in ff[i][k]:
                        index = lsi.index(zb)
                        Biv[index+1][k] = 0.5
                Bi.append(Biv.T)
            B.append(Bi)
            z.append(lsi)
        self.B = B
        self.z = z
        self.P = P 
        
        self.C = np.ones([self.N,self.nC]).astype("float")
        normC = np.sum(self.C,axis=1)
        self.C = (self.C.T/normC).T

        self.A = np.ones([self.N,self.N])
        normA = np.sum(self.A,axis=1)
        self.A = (self.A.T/normA).T
        
        self.pi = np.ones(self.N)
        self.pi = self.pi/np.sum(self.pi)
        
        sigma =[]
        for i in range(self.N):
            sigmai = []
            for j in range(self.nC):
                sigmaij = np.cov(self.O.T)*10.0
                sigmai.append(sigmaij)
            sigma.append(sigmai)
        self.sigma = sigma

    
    def f_dependencies(self):
        """
        Se asume que el estado actual del HMM es de un MoG sin relaciones intertemporales
        retorna mi donde mi[k][l] son las informaciones mutuas entre X_k y la variable l 
        para todos los estados ocultos, i.e len(mi[j][l]) = N.
        retorna fmi[k][l] son las informaciones mutuas entre X_k y la variable l sin
        dependencia del estado oculto
        """
        mi = []
        fmi= [ ]
        P = self.det_lags(p_cor=5)
        self.P = P
        for k in range(self.K):
            mik = []
            fmik = []
            for l in range(self.K):
                for p in range(1,P+1):
                    mik.append(self.mutual_information(k,0,l,p))
                    fmik.append(self.full_mutual_information(k,0,l,p))
            mi.append(np.array(mik))
            fmi.append(np.array(fmik))
        return  [P,np.array(mi),np.array(fmi)]
            
    def mutual_information(self,k,q,l,p):
        """
        Computes the mutual information of two variables given the hidden state
        using the current parameters
        k is the index of a variables
        l is the index of another variable, l!=k
        p is the lag of l, p>0
        i is the hidden state
        I(k,l^{-p}|i) = H(k|i)+H(l^{-p}|i)-H(k,p^{-p}|i)
        H(k|i) ~ T**-1(sum_t(log(P(k^t|i))))
        I is the mutual information, H is the entropy ,T is the length of the data
        """
        return self.mono_entropy(k,q)+self.mono_entropy(l,p)-self.bi_entropy(k,q,l,p)
    
    def full_mutual_information(self,k,q,l,p):
        """
        Computes the mutual information of two variables 
        using the current parameters
        k is the index of a variables
        l is the index of another variable, l!=k
        p is the lag of l, p>0
        i is the hidden state
        I(k,l^{-p}) = H(k)+H(l^{-p})-H(k,p^{-p})
        H(k) ~ T**-1 sum_i(P(i)sum_t(log(P(k^t|i)P(i))))
        I is the mutual information, H is the entropy ,T is the length of the data
        """
        return self.full_mono_entropy(k,q)+self.full_mono_entropy(l,p)-self.full_bi_entropy(k,q,l,p)
    
    def full_bi_entropy(self,k,q,l,p):
        """
        Returns H(K,l^{-p}|i)
        it assumes that the current HMM is a MoG without intertemporal relationships 
        """
        f = forback()
        sigmt = np.array(self.sigma)[:,:,[k,l]]
        bent = []
        for i in range(self.N):
            benti = []
            for j in range(self.nC):
                benti.append([self.B[i][j][k],self.B[i][j][l]])
            bent.append(benti)
        sigmt = (sigmt[:,:].T[[k,l]]).T
        if k ==l:
            det_sigmt = []
            inv_sigmt = []
            for i in range(self.N):
                det_sigmt_i = []
                inv_sigmt_i = []
                for j in range(self.nC):
                    sigmt[i][j] = np.diag(np.diag(sigmt[i][j]))
                    det_sigmt_i.append(np.linalg.det(sigmt[i][j]))
                    inv_sigmt_i.append(np.linalg.inv(sigmt[i][j]))
                det_sigmt.append(det_sigmt_i)
                inv_sigmt.append(inv_sigmt_i)
        if k!= l:
            det_sigmt = []
            inv_sigmt = []
            for i in range(self.N):
                det_sigmt_i = []
                inv_sigmt_i = []
                for j in range(self.nC):
                    det_sigmt_i.append(np.linalg.det(sigmt[i][j]))
                    inv_sigmt_i.append(np.linalg.inv(sigmt[i][j]))
                det_sigmt.append(det_sigmt_i)
                inv_sigmt.append(inv_sigmt_i)
            
        if p>q:
            O2 = np.array([(self.O.T[k])[p-q:],(self.O.T[l])[:-p+q]]).T
        if p==q:
            O2 = np.array([(self.O.T[k])[p:],(self.O.T[l])[p:]]).T
        if q>p:
            O2 = np.array([(self.O.T[k])[:-q+p],(self.O.T[l])[q-p:]]).T
        f.mu_t(O2,bent,self.z,self.N,0,2,self.nC)
        f.prob_BN_t(O2,bent,self.N,0,2,self.nC,self.C,sigmt,inv_sigmt,det_sigmt) 
        f.forward_backward_As(self.A,sigmt,self.pi,O2,0,self.C) 
        ent = 0
        T = len(O2)
        for t in range(T):
            ent = ent+ np.sum(f.gamma[t]*np.log(np.sum(f.gamma[t]*np.exp(f.prob_comp[t]))**(1./float(T))))
        return -ent
        
    def full_mono_entropy(self,k,p):
        """
        Computes the entropy of H(k,l^{-p}|i) dfor each hidden state i
        it assumes that the current HMM is a MoG without intertemporal relationships 
        """
        f = forback()
        bent = []
        for i in range(self.N):
            benti = []
            for j in range(self.nC):
                benti.append(self.B[i][j][k])
            bent.append(benti)
        sigmt = []
        for i in range(self.N):
            sigmti = []
            for j in range(self.nC):
                sigmtij = np.array([self.sigma[i][j][k][k]])
                sigmti.append(sigmtij)
            sigmt.append(sigmti)
        O = self.O.T[k]
        O = O[np.newaxis].T
        f.mu_t(O[p:],bent,self.z,self.N,0,2,self.nC)
        if p>0:
            f.prob_BN_t(O[:-p],bent,self.N,self.P,self.K,self.nC,self.C,sigmt,sigmt,sigmt)
            f.forward_backward_As(self.A,sigmt,self.pi,O[:-p],self.P,self.C)
            T = len(O[:-p])
        else:
            f.prob_BN_t(O,bent,self.N,self.P,self.K,self.nC,self.C,sigmt,sigmt,sigmt)
            f.forward_backward_As(self.A,sigmt,self.pi,O,self.P,self.C)  
            T = len(O)
        ent = 0
        for t in range(len(f.gamma)):
            ent = ent+ np.sum(f.gamma[t]*np.log(np.sum(f.gamma[t]*np.exp(f.prob_comp[t]))**(1./float(T))))
        return -ent
        
        
    def bi_entropy(self,k,q,l,p):
        """
        Computes the entropy of H(k,l^{-p}|i) dfor each hidden state i
        it assumes that the current HMM is a MoG without intertemporal relationships 
        """
        f = forback()
        sigmt = np.array(self.sigma)[:,:,[k,l]]
        bent = []
        for i in range(self.N):
            benti = []
            for j in range(self.nC):
                benti.append([self.B[i][j][k],self.B[i][j][l]])
            bent.append(benti)
        sigmt = (sigmt[:,:].T[[k,l]]).T
        if k ==l:
            det_sigmt = []
            inv_sigmt = []
            for i in range(self.N):
                det_sigmt_i = []
                inv_sigmt_i = []
                for j in range(self.nC):
                    sigmt[i][j] = np.diag(np.diag(sigmt[i][j]))
                    det_sigmt_i.append(np.linalg.det(sigmt[i][j]))
                    inv_sigmt_i.append(np.linalg.inv(sigmt[i][j]))
                det_sigmt.append(det_sigmt_i)
                inv_sigmt.append(inv_sigmt_i)
        if k!= l:
            det_sigmt = []
            inv_sigmt = []
            for i in range(self.N):
                det_sigmt_i = []
                inv_sigmt_i = []
                for j in range(self.nC):
                    det_sigmt_i.append(np.linalg.det(sigmt[i][j]))
                    inv_sigmt_i.append(np.linalg.inv(sigmt[i][j]))
                det_sigmt.append(det_sigmt_i)
                inv_sigmt.append(inv_sigmt_i)
            
        if p>q:
            O = np.array([(self.O.T[k])[p-q:],(self.O.T[l])[:-p+q]]).T
        if p==q:
            O = np.array([(self.O.T[k])[p:],(self.O.T[l])[p:]]).T
        if q>p:
            O = np.array([(self.O.T[k])[:-q+p],(self.O.T[l])[q-p:]]).T
        f.mu_t(O,bent,self.z,self.N,0,2,self.nC)
        f.prob_BN_t(O,bent,self.N,self.P,self.K,self.nC,self.C,sigmt,inv_sigmt,det_sigmt)
        ent = np.zeros(self.N)
        T = len(O)
        # ent = np.sum(np.log(f.probt**(1./float(T))),axis=0)
        for t in range(len(f.prob_comp)):
            ent = ent+ f.prob_comp[t]*(1./float(T))
        return -ent
    
    def mono_entropy(self,k,p):
        """
        Computes the entropy of H(k,l^{-p}|i) dfor each hidden state i
        it assumes that the current HMM is a MoG without intertemporal relationships 
        """
        f = forback()
        bent = []
        for i in range(self.N):
            benti = []
            for j in range(self.nC):
                benti.append(self.B[i][j][k])
            bent.append(benti)
        sigmt = []
        for i in range(self.N):
            sigmti = []

            for j in range(self.nC):
                sigmtij = np.array([self.sigma[i][j][k][k]])
                sigmti.append(sigmtij)
            sigmt.append(sigmti)
        O = self.O.T[k]
        O = O[np.newaxis].T
        f.mu_t(O[p:],bent,self.z,self.N,0,2,self.nC)
        if p>0:
            f.prob_BN_t(O[:-p],bent,self.N,self.P,self.K,self.nC,self.C,sigmt,sigmt,sigmt)
        else:
            f.prob_BN_t(O,bent,self.N,self.P,self.K,self.nC,self.C,sigmt,sigmt,sigmt)
        ent = np.zeros(self.N)
        T = len(O)
        # ent = np.sum(np.log(f.probt**(1./float(T))),axis=0)
        for t in range(len(f.prob_comp)):
            ent = ent+ f.prob_comp[t]*(1./float(T))
        return -ent
    

    
    def load(self,rootr):
        """
        loads a model, must be a moghmm file 
        """
        if rootr[-4:] != ".bmm":
            return "The file is not a bmm file"
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
        self.z = loaded[9]
        self.mus = loaded[10]
        self.covs = loaded[11]
        self.dictionary = loaded[12]
        self.lengths = loaded[13]
        self.O = loaded[14]
        self.b=loaded[15]
        self.bic = loaded[16]
        self.LogLtrain = loaded[17]
        if len(loaded)>18:
            self.nu = loaded[18]
            self.kappa = loaded[19]
    
        
class forback:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.ganmix = None
        self.mut = None
        self.ll = None
        self.probt = None
        self.prob_comp = None
        self.numa = None
        self.dena = None
        self.coefmat = None
        self.coefvec =None
        self.numsig = None
        self.densig = None
        self.nummix = None
        self.denmix = None

        
    def mu_t(self,O,B,z,N,P,K,nC): 
        """
        Calcula los valores de medias para las observaciones usando los valores de B
        O es un set de datos de K variables en T tiempos O[t]= [o_1,...,o_k]
        B es una lista que tiene coeficientes de cada estado, B_i =[B_i1,...,B_ik]
        L es el orden topologico, solo se usa para ordenar el calculo.
        G es una lista de grafos de cada estado G=[G_1,...,G_N]
        retorna mu_t una lista para los valore de mu para cada estado, cada variable y tiempo
        """
        T =len(O.T[0])
        mut = []
        for  i in range(N):
            acum = np.concatenate([np.ones([T-P,1])],axis=1)
            zi = z[i]
            for l in range(len(z[i])):
                x = np.roll(O.T[zi[l][0]],zi[l][1])[P:]
                x = x.reshape((len(x),1))
                acum = np.concatenate([acum,x],axis=1)
            mui = []
            for j in range(nC):
                muij = np.dot(B[i][j],acum.T)
                if np.sum(muij.shape) == T:
                    muij = muij[np.newaxis]
                mui.append(muij.T)
            mut.append(mui)
        self.mut = np.array(mut)
    
    def prob_BN(self,o,mu,sigma,isigma,detsigma,K):  #Este si toca cambiar 
        """
        Calcula [P(O_t|q_t=i,lambda)]_i
        o es una observacion, tiene longitud K, donde K es el numero de variables
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        Retorna un vector de probabilidades de longitud N, donde N es el numero de
            estados
        """
        """
        Computes a normal distribution
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
    

    
    
    def prob_BN_t(self,O,B,N,P,K,nC,C,sigma,isigma,detsigma): #Se le puede indicar tiempo de inicio y de final 
        """
        Calcula [P(O_t|qt=i)]_ti
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        sigma es una lista, donde sigma_i es una lista que contiene las varianzas para 
            las variables en el estado i. sigma_i=[sigma_i1,...,sigma_ik]
        t0 es un tiempo inicial
        t1 es un tiempo final
        Retorna una  matriz B, donde B[t][i] = P(Ot|qt=i)
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
                    probtti.append(self.prob_BN(O[t+P],self.mut[i][v][t],sigma[i][v],isigma[i][v],detsigma[i][v],K))
                maxpi = np.max(np.array(probtti))
                toadd = maxpi+ np.log(np.sum(np.exp(np.log(C[i])+probtti-maxpi)))
                probct.append(toadd)
                probtt.append(probtti)
            self.prob_comp.append(probct)
            self.probt.append(probtt)
        self.probt = np.array(self.probt)
        self.prob_comp = np.array(self.prob_comp)
        

    
    def forward_backward_As(self,A,sigma,pi,O,P,C): 
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
        self.mix_gam(C,sigma,P)
        
    def mix_gam(self,C,sigma,P):
        """
        """
        gammix = []
        for i in range(len(sigma)):
            gammi = self.gamma.T[i]
            gami = []
            for j in range(len(C.T)):
                gammij = np.exp(np.log(C[i][j])+gammi+self.probt[:,i,j]-self.prob_comp.T[i])
                gami.append(gammij)
            gammix.append(gami)
        self.ganmix = np.array(gammix)
        self.gamma = np.exp(self.gamma)
                
    def prob_BN_mix(self,O,C,sigma,P,i,j):
        """
        """
        K = len(O[0])
        T = len(O.T[0])
        gg = []
        Oij = (O[P:]-self.mut[i,j])
        if K>1:
            deti = np.linalg.det(sigma[i][j])
            invi = np.linalg.inv(sigma[i][j])
        else:
            deti = sigma[i][j]
            invi = 1./sigma[i][j]
        for t in range(T-P):
            gg.append(np.sqrt((2*np.pi)**K*deti)**-1*np.exp(-0.5*np.dot(np.dot(Oij[t],invi),Oij[t].T))) 
        gg = np.array(gg)
        return gg
        

    
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
        # return np.exp(np.log(np.dot(alfa,A))+self.prob_comp[t])
    
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
        # return np.dot(A,np.exp(np.log(beta)+self.prob_comp[t]))
        
    def act_aij(self,A,N,sigma,O,P): #TODO hacerlo de forma totalmente vectorial
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
        
    def act_coef(self,O,z,P,N,sigma): 
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
        nC = len(sigma[0])
        w = self.ganmix
        ac =[]
        bc = []
        T = len(O)
        for i in range(N):
            zi = z[i]
            y = np.concatenate([np.ones([1,T-P])],axis=0)
            for l in range(len(z[i])):
                x = np.roll(O.T[zi[l][0]],zi[l][1])[P:]
                x = x.reshape((1,len(x)))
                y = np.concatenate([y,x],axis=0)
            y=y.T
            wi = w[i]
            ai =[]
            bi= []
            for j in range(nC):
                wij = wi[j]                                
                aij =  wij[0]*np.outer(O[P],y[0])
                for t in range(1,T-P):
                    aij = aij + wij[t]*np.outer(O[P+t],y[t])
                ai.append(aij)
            
                bij = wij[0]*np.outer(y[0],y[0])
                for t in range(1,T-P):
                    bij = bij + wij[t]*np.outer(y[t],y[t])
                bi.append(bij)
            ac.append(ai)
            bc.append(bi)
        self.coefmat = bc
        self.coefvec = ac
    

    def act_sigma(self,O,N,p,nC): #TODO: falta modificar este tambien
        """
        Actualiza los valores de desviaciones estandar 
        O es una matriz de Observaciones O[t], son las observaciones de las K variables
            en el tiempo t.
        mu es una lista, donde mu_i es una lista que contiene las medias para 
            las variables en el estado i. mu_i=[mu_i1,...,mu_ik]
        ALFA es una matriz, ALFA[t] determina los valores de las variables forward
            en el estado t para los N estados
        BETA es una matriz, BETA[t] determina los valores de las variables backward
            en el estado t para los N estados
        Restorna una lista sigma, donde sigma_i son los valores de desviacion estandar
            para el estado i, sigma_ik es la desviacion de la variable k en el estado i
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
        """
        numix =[]
        denmix = []
        for i in range(N):
            numixi = []
            denmixi = []
            for j in range(nC):
                numixij = np.sum(self.ganmix[i][j])
                denmixij =np.sum(self.ganmix[i])
                numixi.append(numixij)
                denmixi.append(denmixij)
            numix.append(numixi)
            denmix.append(denmixi)
        self.nummix = np.array(numix)
        self.denmix = np.array(denmix)
        
    