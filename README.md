## Introduction

In this repository you can find the files to use and try asymmetric kernel bases hidden Markov models or KDE-AsHMM for short. The models files can be found in the models folder. I am working in ways to deliver this models and other kinds of HMMs as a package.

Inside the model folder there is another folder with the KDE-AsHMM validation files. Three files can be found, a file for synthetic data, one for cnc machines and another for audio classification. 
 
## Model

KDE-AsHMM chages the emission probabilitities such that depending on the hidden state, a KDE model is used to model the observation density.  Also, a Bayesian network is used to enable the model to share information between features to improve their respective kernels.  
Let $\boldsymbol{X}^{0:T}= (\boldsymbol{X}^0,...,\boldsymbol{X}^T)$ be an observable stochastic process with $\boldsymbol{X}^t=(X^t_1,...,X^t_M)$ a random vector of $M$ variables/features.  Assume that the process $\boldsymbol{X}^{0:T}$ relies on a hidden or non-observable stochastic process $\boldsymbol{Q}^{0:T}= (Q^0,...,Q^T)$, where the  values of the range of $Q^t$ is finite, i.e, $R(Q^t) :=\{1,2,...,N\}$, $t=0,1,...,T$, these values are called states and determine the process $\boldsymbol{X}^{0:T}$. Assume that the training data  is $\boldsymbol{y}^{0:L}$, during the training phase.  The emission probabilitites are defined as:
$$
	b_i(\boldsymbol{X}^t) = \sum_{l= P^*}^T \omega_{il}\prod_{m=1}^M\frac{1}{h_{im}}K\left( \frac{X^t_m -\mu_{im}^l}{h_{im}} \right)
$$
Where the added latent variable $\boldsymbol{W}^t$ is a categorical distribution, which depends on the latent variables $\boldsymbol{Q}^t$, which is used to determine the most representative instances to describe the kernel of each hidden state.   Assume that each variable $X^t_m$ for each hidden state $i$, is associated to  $\boldsymbol{U}^t_{im}= (V^t_{im1},...,V^t_{im\kappa_{im}},X^{t-1}_m,...,X^{t-p_{im}}_m)$ a context-specific random vector, which contains the $\kappa_{im}$ parents and $p_{im}$ AR dependencies of the variables $X^t_m$. Let $\boldsymbol{v}^l_{Q^tm}$ be the instantiate of the random vector $\boldsymbol{U}^t_{Q^t,m}$ in $\boldsymbol{y}^{0:L}$, then the reference value  for each kernel component is computed as:
$$
	\mu_{Q^t,m}^l := y_m^l +\boldsymbol{M}_{Q^t,m}(\boldsymbol{U}^t_{Q^t,m}-\boldsymbol{v}^l_{Q^t,m})^{\top}\boldsymbol{v}^l_{Q^t,m})^{\top}
$$
In this manner, the deviations on parents and AR values can be used to correct the kernel as the data requires.
## Training 

For the training model, the SEM algorithm is applied.  The updating formulas for the latent a posteriori probabilities and parameters are provided in the respective article. 
## References 
[1.]  Silverman, B., 1986. Density Estimation for statistics and data analysis. Routledge.
[2.] Rabiner, L.R., 1990. A tutorial on hidden Markov Models and selected applications in Speech Recognition, in: Readings in speech recognition. Morgan Kaufmann, pp. 267-296.
[3.] Puerto-Santana, C., Bielza, C., Larrañaga, P. and Henter Eje G., 2022,  Context-Specific  Kernel-based Hidden Markov model for time series analysis
