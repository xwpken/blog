# Understanding Diffusion Models: A Unified Perspective

> Reference: [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
>
> Last updated: 28 Jun 2026

## Introduction: Generative Models

The goal of generative modeling is to learn to $model$ its true data distribution $p(\boldsymbol{x})$.

Several approaches have been proposed to achieve this goal, including:
- **Generative Adversarial Networks (GANs)**: model the sampling procedure of a complex distribution by learning in an adversarial manner.
- **Likelihood-based models**: learn to maximize the likelihood of the observed data, including *Autoregressive models*, *Normalizing Flows*, and *Variational Autoencoders (VAEs)*.
- **Energy-based models**: learn to model the energy function of the data distribution.
- **Score-based models**: learn to model the score function of the data distribution.

## Background: ELBO, VAE, and Hierarchical VAE
We aim to learn **lower-dimensional latent representations** $\boldsymbol{z}$ of the data $\boldsymbol{x}$, which can be used to generate new samples from the data distribution. 

### Evidence Lower Bound (ELBO)

The Evidence Lower Bound (ELBO) is a lower bound of the evidence $p(\boldsymbol{x})$:

$$
\begin{align}
\text{ELBO} = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\right]
\end{align}
$$

We can prove that the ELBO is indeed a lower bound of the evidence by using Jensen's inequality:

$$
\begin{align}
\log p(\boldsymbol{x}) = \log \int p(\boldsymbol{x}, \boldsymbol{z}) d\boldsymbol{z} = \log \int q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})} d\boldsymbol{z} \ge \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\right]
\end{align}
$$

which can also be obtained using the **Kullback-Leibler (KL) divergence**:

$$
\begin{align}
\log p(\boldsymbol{x}) &=\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}[\log p(\boldsymbol{x})]
\\
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\right] + \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\left[\log\frac{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}{p(\boldsymbol{z}|\boldsymbol{x})}\right]\\
 &= \text{ELBO} + D_\text{KL}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) || p(\boldsymbol{z}|\boldsymbol{x})\right)
 \geq \text{ELBO}
\end{align}
$$

which holds because the KL divergence is always **non-negative**. Since the ELBO and the KL divergence sum to the log evidence, which is constant with respect to the parameters $\boldsymbol{\phi}$, maximizing the ELBO is equivalent to **minimizing the KL divergence** between the approximate posterior $q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})$ and the true posterior $p(\boldsymbol{z}|\boldsymbol{x})$. Additionally, the trained ELBO can be used to **estimate the likelihood** of the oberved or generated data.

### Variational Autoencoder (VAE)
In a Variational Autoencoder (VAE), we directly maximize the ELBO with respect to the parameters $\boldsymbol{\phi}$ of the **encoder** $q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})$ and the parameters $\boldsymbol{\theta}$ of the **decoder** $p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z})$:

$$
\begin{align}
    \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\left[\log \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z})p(\boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\right] &= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\left[\log \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z})p(\boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\right] \\
    &= \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z})\right]}_{\text{Reconstruction term}} - \underbrace{D_\text{KL}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) || p(\boldsymbol{z})\right)}_{\text{Prior matching term}}
\end{align}
$$

The encoder $q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})$ is typically modeled as a multivariate Gaussian distribution with a diagonal covariance matrix:

$$
\begin{align}
q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) = \mathcal{N}(\boldsymbol{z}; \boldsymbol{\mu}_{\boldsymbol{\phi}}(\boldsymbol{x}), \boldsymbol{\sigma}^2_{\boldsymbol{\phi}}(\boldsymbol{x})\mathbf{I})
\end{align}
$$

and the prior $p(\boldsymbol{z})$ is usually chosen to be a standard multivariate Gaussian distribution:

$$
\begin{align}
p(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{z}; \mathbf{0}, \mathbf{I})
\end{align}
$$

The reconstruction term can be estimated using Monte Carlo sampling, and the KL divergence term can be computed analytically, leading to the objective:

$$
\begin{align}
&\arg\max_{\boldsymbol{\phi}, \boldsymbol{\theta}} \left[\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z})\right] - D_\text{KL}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) || p(\boldsymbol{z})\right)\right]\\ = &\arg\max_{\boldsymbol{\phi}, \boldsymbol{\theta}} \left[\sum_{l=1}^{L} \log p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z}^{(l)}) - D_\text{KL}\left(q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x}) || p(\boldsymbol{z})\right)\right]
\end{align}
$$

Since latents ${\boldsymbol{z}^{(l)}}_{l=1}^{L}$ are sampled from $q_{\boldsymbol{\phi}}(\boldsymbol{z}|\boldsymbol{x})$, which is a stochastic process that is generally non-differentiable, we can use the **reparameterization trick** to make the sampling process differentiable:

$$
\begin{align}
\boldsymbol{z} = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\boldsymbol{x}) + \boldsymbol{\sigma}_{\boldsymbol{\phi}}(\boldsymbol{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{align}
$$

where $\odot$ denotes element-wise multiplication. This allows gradients to flow through the sampling process, enabling end-to-end training of the VAE.

### Hierarchical Variational Autoencoders
The Hierarchical Variational Autoencoder (HVAE) extends the VAE framework by introducing multiple layers of latent variables. Whereas in the general HVAE with $T$ layers, each latent is allowed to condition on all previous latents, we only consider a special case called the **Markovian HVAE**, where each latent variable $\boldsymbol{z}_t$ only depends on the previous latent variable $\boldsymbol{z}_{t-1}$. The joint distribution of the data $\boldsymbol{x}$ and the posterior of a Markovian HVAE can be factorized as follows:

$$
\begin{align}
&p(\boldsymbol{x}, \boldsymbol{z}_{1:T}) = p(\boldsymbol{z}_T)p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z}_1)\prod_{t=2}^{T} p_{\boldsymbol{\theta}}(\boldsymbol{z}_{t-1}|\boldsymbol{z}_t)\\
&q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}|\boldsymbol{x}) = q_{\boldsymbol{\phi}}(\boldsymbol{z}_1|\boldsymbol{x})\prod_{t=2}^{T} q_{\boldsymbol{\phi}}(\boldsymbol{z}_t|\boldsymbol{z}_{t-1})
\end{align}
$$

which can be substituted into the ELBO to obtain the objective function for training the Markovian HVAE:

$$
\begin{align}
\text{ELBO} &= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}|\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z}_{1:T})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}|\boldsymbol{x})}\right]\\
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}|\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{z}_T)p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z}_1)\prod_{t=2}^{T} p_{\boldsymbol{\theta}}(\boldsymbol{z}_{t-1}|\boldsymbol{z}_t)}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_1|\boldsymbol{x})\prod_{t=2}^{T} q_{\boldsymbol{\phi}}(\boldsymbol{z}_t|\boldsymbol{z}_{t-1})}\right]
\end{align}
$$

## Variational Diffusion Models 

