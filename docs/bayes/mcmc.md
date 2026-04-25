# Markov Chain Monte Carlo (MCMC)

> Editor: [Weipeng Xu](https://github.com/xwpken) & Copilot
> Last updated: 24 Mar 2026

For high-dimensional or complicated probability density functions, direct sampling to obatin a sequence of random sampled could be challenging due to **the intractability of normalization** and **the curse of dimensionality**. One commonly-used approach to tackle this challenge is the Markov chain Monte Carlo (MCMC) method, which constructs a Markov chain  with the stationary distribution being target distribution. Due to their effectiveness and flexibility, MCMC methods have been widely used in modern Bayesian inference.

## Markov Chain

A Markov chain is a stochastic process describing a sequence of possible events where the probability of possible events only depends on the state of the previous event.

Mathematically, a Markov chain is defined by a set of states $\{\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots\}$, where each $\boldsymbol{x} \in \mathbb{R}^d$ is a $d$-dimensional vector. The Markov property (memorylessness) can be written as
$$
\tag{1}
\mathcal{P}(\boldsymbol{x}_{t+1} | \boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_t) = \mathcal{P}(\boldsymbol{x}_{t+1} | \boldsymbol{x}_t)
$$
where $\mathcal{P}(\boldsymbol{x}_{t+1} | \boldsymbol{x}_t)$ is the conditional probability of transitioning from $\boldsymbol{x}_t$ to $\boldsymbol{x}_{t+1}$. In MCMC, this conditional probability is often expressed via the transition kernel $K(\boldsymbol{x}' | \boldsymbol{x})$, which governs the evolution of the chain. For any measurable set $A \subseteq \mathbb{R}^d$,
$$
\tag{2}
\mathcal{P}(\boldsymbol{x}_{t+1} \in A \mid \boldsymbol{x}_t = \boldsymbol{x}) = \int_A K(\boldsymbol{x}' | \boldsymbol{x}) d\boldsymbol{x}'
$$
Here, $\boldsymbol{x}$ denotes the current state, and $\boldsymbol{x}'$ denotes a possible next state. The distribution of the next state $\boldsymbol{x}_{t+1}$ is then given by
$$
\tag{3}
p_{t+1}(\boldsymbol{x}') = \int K(\boldsymbol{x}' | \boldsymbol{x}) p_t(\boldsymbol{x}) \, d\boldsymbol{x}
$$
where $p_t(\boldsymbol{x})$ is the distribution at time $t$.

If the transition kernel $K(\boldsymbol{x}' | \boldsymbol{x})$ satisfies the **detailed balance condition** with respect to a distribution $\pi(\boldsymbol{x})$, i.e.,
$$
\tag{4}
\pi(\boldsymbol{x}) K(\boldsymbol{x}' | \boldsymbol{x}) = \pi(\boldsymbol{x}') K(\boldsymbol{x} | \boldsymbol{x}')
$$
then $\pi(\boldsymbol{x})$ is a stationary distribution of the Markov chain. In MCMC, we design the transition kernel so that the stationary distribution $\pi(\boldsymbol{x})$ is exactly the distribution we wish to sample from.

Different choices of the transition kernel $K(\boldsymbol{x}' | \boldsymbol{x})$ lead to different MCMC algorithms, which will be introduced in the next sections.