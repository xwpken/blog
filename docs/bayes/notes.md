# Bayesian inference with discrete cosine transformation

## Bayesian inference
Bayesian inference is a statistical method that updates the probability distribution of model parameters $\boldsymbol{\theta}$ based on observed data $\boldsymbol{d}$, which is based on the **Bayes' theorem**:
$$
P(\boldsymbol{\theta}\vert\boldsymbol{d}) = \frac{P(\boldsymbol{d}\vert\boldsymbol{\boldsymbol{\theta}})\cdot P(\boldsymbol{\theta})}{P(\boldsymbol{d})}
$$
where $P(\boldsymbol{\theta}\vert\boldsymbol{d})$ is the posterior distribution, $P(\boldsymbol{d}\vert\boldsymbol{\boldsymbol{\theta}})$ is the likelihood function, $P(\boldsymbol{\theta})$ is the prior distribution, and $P(\boldsymbol{d})$ is the marginal likelihood or evidence, which can be expressed as:
$$
P(\boldsymbol{d}) = \int P(\boldsymbol{d}\vert\boldsymbol{\theta})\cdot         P(\boldsymbol{\theta})d\boldsymbol{\theta}
$$
In the MCMC sampling, the evidence $P(\boldsymbol{d})$ dose not affect the results. Therefore, we have:
$$
P(\boldsymbol{\theta}\vert\boldsymbol{d})\propto P(\boldsymbol{d}\vert\boldsymbol{\theta})\cdot P(\boldsymbol{\theta})
$$

## Discrete Cosine Transformation
The ​Discrete Cosine Transform (DCT)​​ is a widely used linear transformation that converts a signal from the time (or spatial) domain into the frequency domain. It is particularly effective for compressing signals with strong correlations, such as images and audio, because it concentrates most of the signal's energy into a few low-frequency coefficients.
### Formulations
Assume that we have a time-domain signal $\boldsymbol{x}\in[x_0,x_1,\cdots,x_{N-1}]^{\textrm{T}}\in\mathbb{R}^N$, the DCT-II transformation can be expressed as:

$$
y_k = a_k\sum_{n=1}^{N-1}x_n\cos\left(\frac{\pi(2n+1)k}{2N}\right),\quad k=0,1,\cdots,N-1
$$
where 
$$
a_k = \begin{cases} 
\sqrt{\frac{1}{N}}, & k = 0 \\
\sqrt{\frac{2}{N}}, & k > 0 
\end{cases} 
$$
The matrix form can be written as:
$$
\boldsymbol{y} = \boldsymbol{D}_N\boldsymbol{x}
$$
with the component of $\boldsymbol{D}_N\in\mathbb{R}^{N\times N}$ as:
$$
\boldsymbol{D}_N[k,n] = a_k\cos\left(\frac{\pi\left(2n+1\right)k}{2N}\right)
$$

For 2D signal $\boldsymbol{X}\in\mathbb{R}^{N\times M}$, its 2D DCT-II transformation can be expressed as:
$$
Y_{k_1,k_2} = a_{k_1}a_{k_2}\sum_{n_1=0}^{N-1}\sum_{n_2=0}^{M-1}\boldsymbol{X}_{n_1,n_2}\cos\left(\frac{\pi\left(2n_1+1\right)k_1}{2N}\right)\cos\left(\frac{\pi\left(2n_2+1\right)k_2}{2N}\right)
$$
The matrix form can be expressed as
$$
\boldsymbol{Y} = \boldsymbol{D}_N\boldsymbol{X}\boldsymbol{D}_M^{\textrm{T}}
$$
Due to the orthogonality of the DCT matrix, the inverse DCT-II transformation can be expressed as:
$$
\boldsymbol{X} = \boldsymbol{D}_N^T\boldsymbol{Y}\boldsymbol{D}_M
$$

### Derivations for latent-space sampling
#### Forward transformation
Consider the original time-domain variables $\boldsymbol{\theta}\in\mathbb{R}^{M^2}$ defined on a square domain of size $M\times M$, the DCT-II transformation can be expressed as
$$
\boldsymbol{x}_{\text{full}} = \boldsymbol{D}_{2D}\boldsymbol{\theta}
$$
where $\boldsymbol{x}_{\text{full}}\in\mathbb{R}^{M^2}$ is the frequency-domain variables, and $\boldsymbol{D}_{\textrm{2D}}\in\mathbb{R}^{M^2\times M^2}$ is the transformation matrix:
$$
\boldsymbol{D}_{\textrm{2D}} = \boldsymbol{D}_N\otimes\boldsymbol{D}_N
$$
The non-zero value are concentrated on the low-frequencey parts of $\boldsymbol{x}_{\text{full}}$. Therefore, we can select the $d$ componnets $(d\ll M^2)$ corresponding to the low frequencies:
$$
\boldsymbol{x} = \boldsymbol{Q}\boldsymbol{x}_{\text{full}} = \boldsymbol{Q}\boldsymbol{D}_{2D}\boldsymbol{\theta}
$$
where $\boldsymbol{x}\in\mathbb{R}^d$ is the low-frequency coefficients vector. $\boldsymbol{Q}\in\mathbb{R}^{d\times M^2}$ is the selection matrix with one for selected coefficient and zeros elsewhere on each row (one-hot vector).  

### Inverse transformation
For the frequency variable $\boldsymbol{x}$ with lower dimensions, we can reconstruct the time-domain variable:
$$
\boldsymbol{\theta}_{\textrm{app}} = \boldsymbol{D}_{\textrm{2D}}^T\boldsymbol{Q}^T\boldsymbol{x}
$$
where $\boldsymbol{\theta}_{\textrm{app}}\in\mathbb{R}^{M^2}$ is the reconstructed time-domain vector.

## Examples

This section provides two examples to illustrate the proposed sampling scheme. First, we derive the posterior distribution of the classical linear regression model and then incorporate the DCT-II into it.

### Linear regression
We consider the following linear regression model:
$$
\boldsymbol{d} = \boldsymbol{C}\boldsymbol{\theta} + \boldsymbol{b} + \boldsymbol{\varepsilon} 
$$
where $\boldsymbol{d}\in\mathbb{R}^N$ is the observed vector, $\boldsymbol{C}\in\mathbb{R}^{N\times M}$ is the design matrix, $\boldsymbol{b}\in\mathbb{R}^N$ is the bias vector, and $\boldsymbol{\theta}\in\mathbb{R}^M$ is the inference parameters. $\boldsymbol{\varepsilon}\in\mathbb{R}^N$ is the noise vector following an i.i.d Gaussain distribution:
$$
\boldsymbol{\varepsilon}\sim N(\boldsymbol{0},\sigma_n^2\boldsymbol{I})
$$
where $\sigma_n$ is the standard deviation. $N(\cdot)$ is the Gaussian distribution. For random variable $x$, we have:
$$
N(x\vert \mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$
For random vector $\boldsymbol{x}\in\mathbb{R}^N$, we have:
$$
N(\boldsymbol{x}\vert\boldsymbol{\mu},\boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^N\vert\boldsymbol{\Sigma}\vert}}\exp(-\frac{(\boldsymbol{x}-\boldsymbol{\mu})^{\textrm{T}}\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})}{2})
$$
where $\vert\boldsymbol{\Sigma}\vert$ is the determinant of the covariance matrix. The logarithm can be expressed as:
$$
\begin{align*}
    \log N(\boldsymbol{x}\vert\boldsymbol{\mu},\boldsymbol{\Sigma}) &= -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\textrm{T}}\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\\
    &=-\frac{1}{2}\boldsymbol{x}^{\textrm{T}}\boldsymbol{\Sigma}^{-1}\boldsymbol{x}+\boldsymbol{x}^{\textrm{T}}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu} + \textrm{const}
\end{align*}
$$
We assume a Gaussian prior distribution over $\boldsymbol{\theta}$:
$$
P(\boldsymbol{\theta}) = N(\boldsymbol{\theta}\vert\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)
$$
where $\boldsymbol{\mu}_0\in\mathbb{R}^M$ is the prior mean. $\boldsymbol{\Sigma}_0\in\mathbb{R}^{M\times M}$ is the covariance matrix, which is **symmetric positive definite**. The likelihood function of the observed data $\boldsymbol{d}$ with the paramter vector $\boldsymbol{\theta}$ is expressed as:
$$
P(\boldsymbol{d}\vert\boldsymbol{\theta}) = N(\boldsymbol{d}\vert\boldsymbol{C}\boldsymbol{\theta}+\boldsymbol{b},\sigma_n^2\boldsymbol{I})
$$ 
Therefore, the posterior distribution $P(\boldsymbol{\theta}\vert\boldsymbol{d})$ satisfies:
$$
\begin{align*}
    P(\boldsymbol{\theta}\vert\boldsymbol{d})&\propto N(\boldsymbol{d}\vert\boldsymbol{C}\boldsymbol{\theta},\sigma_n^2\boldsymbol{I})\cdot N(\boldsymbol{\theta}\vert\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)\\
&\propto \exp\left(-\frac{1}{2\sigma_n^2}(\boldsymbol{d}-\boldsymbol{C}\boldsymbol{\theta}-\boldsymbol{b})^{\textrm{T}}(\boldsymbol{d}-\boldsymbol{C}\boldsymbol{\theta}-\boldsymbol{b})-\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{\mu}_0)^{\textrm{T}}\boldsymbol{\Sigma}_0^{-1}(\boldsymbol{\theta}-\boldsymbol{\mu}_0)\right)
\end{align*}
$$
Take the logarithm:
$$
\begin{align*}
    \log  P(\boldsymbol{\theta}\vert\boldsymbol{d}) &= -\frac{1}{2\sigma_n^2}(\boldsymbol{d}-\boldsymbol{C}\boldsymbol{\theta}-\boldsymbol{b})^{\textrm{T}}(\boldsymbol{d}-\boldsymbol{C}\boldsymbol{\theta}-\boldsymbol{b})-\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{\mu}_0)^{\textrm{T}}\boldsymbol{\Sigma}_0^{-1}(\boldsymbol{\theta}-\boldsymbol{\mu}_0) + \text{const}\\
    &=-\frac{1}{2\sigma_n^2}\left[(\boldsymbol{d}-\boldsymbol{b})^{\textrm{T}}(\boldsymbol{d}-\boldsymbol{d})-2\boldsymbol{\theta}^{\textrm{T}}\boldsymbol{C}^{\textrm{T}}(\boldsymbol{d}-\boldsymbol{b})+\boldsymbol{\theta}^{\textrm{T}}\boldsymbol{C}^{\textrm{T}}\boldsymbol{C}\boldsymbol{\theta}\right] -\frac{1}{2}\left(\boldsymbol{\theta}^{\textrm{T}}\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\theta}-2\boldsymbol{\theta}^{\textrm{T}}\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0+\boldsymbol{\mu}_0^{\textrm{T}}\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\right)+\textrm{const}\\
&=-\frac{1}{2}\boldsymbol{\theta}^{\textrm{T}}\left(\frac{1}{\sigma_n^2}\boldsymbol{C}^{\textrm{T}}\boldsymbol{C}+\boldsymbol{\Sigma_0^{-1}}\right)\boldsymbol{\theta} + \boldsymbol{\theta}^{\textrm{T}}\left[\frac{1}{\sigma_n^2}\boldsymbol{C}^{\textrm{T}}(\boldsymbol{d}-\boldsymbol{b})+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\right]+\textrm{const}\\
&=-\frac{1}{2}\boldsymbol{\theta}^{\textrm{T}}\boldsymbol{\Sigma}^{-1}_{\textrm{post}}\boldsymbol{\theta} + \boldsymbol{\theta}^{\textrm{T}}\boldsymbol{\Sigma}^{-1}_{\textrm{post}}\boldsymbol{\mu}_{\textrm{post}} + \textrm{const}
\end{align*}
$$
Therefore, we have:
$$
\begin{align*}
    &\boldsymbol{\Sigma}^{-1}_{\textrm{post}} = \frac{1}{\sigma_n^2}\boldsymbol{C}^{\textrm{T}}\boldsymbol{C}+\boldsymbol{\Sigma_0^{-1}}\\
    &\boldsymbol{\mu}_{\textrm{post}} = \boldsymbol{\Sigma}_{\textrm{post}}\left[\frac{1}{\sigma_n^2}\boldsymbol{C}^{\textrm{T}}(\boldsymbol{d}-\boldsymbol{b})+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0\right]
\end{align*}
$$
> TODO: validate the 'const' term?

### Linear regression with DCT-II
Since the components of $\boldsymbol{x}$ are of different orders, we first standardlize $\boldsymbol{x}$:
$$
\boldsymbol{x} = \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_x^{\frac{1}{2}}\boldsymbol{z}
$$
where $\boldsymbol{\mu}_x$ and $\boldsymbol{\Sigma}_x$ are the empirically estimated mean and covariance matrix. The latent variables $\boldsymbol{z}\in\mathbb{R}^d$ belongs to:
$$
\boldsymbol{z}\sim N(0,\boldsymbol{I}_d)
$$
Therefore, the linear regression model with DCT-II can be expressed as:
$$
\begin{aligned}
    \boldsymbol{d} &= \boldsymbol{C}\boldsymbol{D}_{2D}^{\textrm{T}}\boldsymbol{Q}^{\textrm{T}}\boldsymbol{x} + \boldsymbol{\varepsilon}\\
    &=\boldsymbol{C}\boldsymbol{D}_{2D}^{\textrm{T}}\boldsymbol{Q}^{\textrm{T}}\boldsymbol{\Sigma}_x^{\frac{1}{2}}\boldsymbol{z} + \boldsymbol{C}\boldsymbol{D}_{2D}^{\textrm{T}}\boldsymbol{Q}^{\textrm{T}}\boldsymbol{\mu}_x+\boldsymbol{\varepsilon}\\
    &=\boldsymbol{A}\boldsymbol{z} + \boldsymbol{b} + \boldsymbol{\varepsilon}
\end{aligned}
$$
where $\boldsymbol{A}=\boldsymbol{C}\boldsymbol{D}_{2D}^{\textrm{T}}\boldsymbol{Q}^{\textrm{T}}\boldsymbol{\Sigma}_x^{\frac{1}{2}}$ and $\boldsymbol{b} = \boldsymbol{C}\boldsymbol{D}_{2D}^{\textrm{T}}\boldsymbol{Q}^{\textrm{T}}\boldsymbol{\mu}_x$. The posterior distribution can be expressed as:
$$
P(\boldsymbol{z}\vert\boldsymbol{d}_{\textrm{obs}})\propto P(\boldsymbol{d}_{\textrm{obs}}\vert\boldsymbol{z})\cdot P(\boldsymbol{z}) 
$$
Take the logarithm, we have:
$$
\begin{aligned}
    \log P(\boldsymbol{z}\vert\boldsymbol{d}_{\textrm{obs}})&=-\frac{1}{2\sigma_n^2}(\boldsymbol{d}_{\textrm{obs}}-\boldsymbol{A}\boldsymbol{z}-\boldsymbol{b})^{\textrm{T}}(\boldsymbol{d}_{\textrm{obs}}-\boldsymbol{A}\boldsymbol{z}-\boldsymbol{b}) -\frac{1}{2}\boldsymbol{z}^{\textrm{T}}\boldsymbol{z} + \textrm{const}\\
    &=  - \frac{1}{2\sigma_n^2}[\boldsymbol{z}^{\textrm{T}}\boldsymbol{A}^{\textrm{T}}\boldsymbol{A}\boldsymbol{z}-2(\boldsymbol{d}_{\textrm{obs}}-\boldsymbol{b})^{\textrm{T}}\boldsymbol{A}\boldsymbol{z}] -\frac{1}{2}\boldsymbol{z}^{\textrm{T}}\boldsymbol{z} + \textrm{const}\\
&=-\frac{1}{2}\boldsymbol{z}^{\textrm{T}}(\frac{1}{\sigma_n^2}\boldsymbol{A}\boldsymbol{A}^T+\boldsymbol{I}_d)\boldsymbol{z} + \frac{1}{\sigma_n^2}(\boldsymbol{d}_{\textrm{obs}}-\boldsymbol{b})^{\textrm{T}}\boldsymbol{A}\boldsymbol{z}+\textrm{const}\\
&=-\frac{1}{2}\boldsymbol{z}^{\textrm{T}}(\frac{1}{\sigma_n^2}\boldsymbol{A}\boldsymbol{A}^T+\boldsymbol{I}_d)\boldsymbol{z} + \boldsymbol{z}^{\textrm{T}}\frac{1}{\sigma_n^2}\boldsymbol{A}^{\textrm{T}}(\boldsymbol{d}_{\textrm{obs}}-\boldsymbol{b})+\textrm{const}
\end{aligned}
$$
Compare with the logarithm of the standard Guassian distribution, we have
$$
P(\boldsymbol{z}\vert\boldsymbol{d}_{\textrm{obs}}) = N(\boldsymbol{\mu}_{\textrm{post}},\boldsymbol{\Sigma}_{\textrm{post}})
$$
where
$$
\begin{aligned}
    &\boldsymbol{\Sigma}_{\textrm{post}}^{-1} = \frac{1}{\sigma_n^2}\boldsymbol{A}^{\textrm{T}}\boldsymbol{A}+\boldsymbol{I}_d\\
    &\boldsymbol{\mu}_{\textrm{post}} = \frac{1}{\sigma_n^2}\boldsymbol{\Sigma}_{\textrm{post}}\boldsymbol{A}^{\textrm{T}}(\boldsymbol{d}_{\textrm{obs}}-\boldsymbol{b})
\end{aligned}
$$


$$
\begin{aligned}
\boldsymbol{\Sigma}_{z,\textrm{post}}^{-1} &= \frac{1}{\sigma_{\text{noise}}^2}\boldsymbol{S}^{\textrm{T}}\boldsymbol{S}+\boldsymbol{I}_d\\
&=\frac{1}{\sigma_{\text{noise}}^2}\boldsymbol{\Sigma}_x^{\frac{1}{2}}\boldsymbol{Q}\boldsymbol{D}_{2D}
\boldsymbol{D}_{2D}^{\textrm{T}}\boldsymbol{Q}^{\textrm{T}}\boldsymbol{\Sigma}_x^{\frac{1}{2}}
+\boldsymbol{I}_d\\
&=\frac{1}{\sigma_{\text{noise}}^2}\boldsymbol{\Sigma}_x
+\boldsymbol{I}_d\\
\end{aligned}
$$

$$
\begin{aligned}
\boldsymbol{\Sigma}_{E,\textrm{post}} &= \boldsymbol{S}(\frac{1}{\sigma_{\text{noise}}^2}\boldsymbol{\Sigma}_x
+\boldsymbol{I}_d)^{-1}\boldsymbol{S}^{\text{T}}\\
&= \boldsymbol{S}\boldsymbol{H}\boldsymbol{S}^{\text{T}}\\
&=\boldsymbol{D}_{2D}^{\textrm{T}}\boldsymbol{Q}^{\textrm{T}}\boldsymbol{\Sigma}_x^{\frac{1}{2}}
\boldsymbol{H}\boldsymbol{\Sigma}_x^{\frac{1}{2}}\boldsymbol{Q}\boldsymbol{D}_{2D}\\
&=\boldsymbol{D}_{2D}^{\textrm{T}}\boldsymbol{Q}^{\textrm{T}}\text{diag} (\frac{\boldsymbol{\sigma}_{\text{noise}}^2 *\boldsymbol{x}_{g}^2}{\boldsymbol{\sigma}_{\text{noise}}^2 +\boldsymbol{x}_{g}^2})\boldsymbol{Q}\boldsymbol{D}_{2D}
\end{aligned}
$$