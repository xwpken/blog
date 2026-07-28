# Chapter 1 Preliminary Concepts

> Last updated: 26 Apr 2026

## 1.2 Vector and Tensor calculus

### 1.2.1 Vector and tensor

Cartesian vectors: ~

Cartesian tensors: $\boldsymbol{T} = T_{ij} \boldsymbol{e}_i \otimes \boldsymbol{e}_j$, where $\otimes$ is the dyadic product that increases the order of the tensor by one. It should be noted that $\boldsymbol{u}\otimes\boldsymbol{v} \neq \boldsymbol{v}\otimes\boldsymbol{u}$.

Several properties of the dyadic product are as follows:
1. $a\boldsymbol{u}\otimes\boldsymbol{v} = \boldsymbol{u}\otimes a\boldsymbol{v} = a(\boldsymbol{u}\otimes\boldsymbol{v})$, where $a$ is a scalar.
2. $\boldsymbol{u}\otimes(\boldsymbol{v}+\boldsymbol{w}) = \boldsymbol{u}\otimes\boldsymbol{v} + \boldsymbol{u}\otimes\boldsymbol{w}$
3. $(\boldsymbol{u}\otimes\boldsymbol{v})\cdot\boldsymbol{w} = \boldsymbol{u}(\boldsymbol{v}\cdot\boldsymbol{w})$, which means the inner product is applied to the adjacent vectors. For rank-$m$ and rank-$n$ tensors, the inner product yields a rank-$(m+n-2)$ tensor. 


The transpose of a tensor is defined as $\boldsymbol{T}^\top = T_{ji} \boldsymbol{e}_i \otimes \boldsymbol{e}_j$.

Symmetric tensors: $\boldsymbol{S} = \boldsymbol{S}^\top$ and Skew tensors: $\boldsymbol{W} = -\boldsymbol{W}^\top$. Every rank-2 tensor can be decomposed into a symmetric part and a skew part, i.e., $\boldsymbol{T} = \boldsymbol{S} + \boldsymbol{W}$, where $\boldsymbol{S} = (\boldsymbol{T} + \boldsymbol{T}^\top)/2$ and $\boldsymbol{W} = (\boldsymbol{T} - \boldsymbol{T}^\top)/2$.

> The symmetric part of displacement gradient$\nabla\boldsymbol{u}=\partial\boldsymbol{u}/\partial\boldsymbol{x}$ is the strain tensor, and the skew part is the spin tensor.

The **contraction operator** or **double inner product** of two rank-2 tensors is defined as $\boldsymbol{A}:\boldsymbol{B} = A_{ij}B_{ij}$, yielding a scalar. It can also be used to define the norm of a rank-2 tensor as $\|\boldsymbol{A}\| = \sqrt{\boldsymbol{A}:\boldsymbol{A}}$.

Several properties of the contraction operator are as follows:
1. $\boldsymbol{A}:\boldsymbol{B} = \boldsymbol{B}:\boldsymbol{A}$
2. $\boldsymbol{A}:(\boldsymbol{B}+\boldsymbol{C}) = \boldsymbol{A}:\boldsymbol{B} + \boldsymbol{A}:\boldsymbol{C}$
3. $\boldsymbol{A}:(\boldsymbol{B}\boldsymbol{C}) = (\boldsymbol{B}^\top\boldsymbol{A}):\boldsymbol{C} = (\boldsymbol{A}\boldsymbol{C}^\top):\boldsymbol{B}$


The trace of a rank-2 tensor is defined as $\text{tr}(\boldsymbol{A}) = \boldsymbol{A}:\boldsymbol{I} = \boldsymbol{I}:\boldsymbol{A} = A_{ii}$.

The orthogonal tensor represents the rotational relation between two coordinate systems: $\boldsymbol{u}^* = \boldsymbol{\beta}\cdot\boldsymbol{u}$ and $\boldsymbol{u} = \boldsymbol{\beta}^\top\cdot\boldsymbol{u}^*$, where the component of $\boldsymbol{\beta}$ is $\beta_{ij} = \boldsymbol{e}_i^* \cdot \boldsymbol{e}_j$. We also have $\boldsymbol{\beta}^\top\cdot\boldsymbol{\beta} = \boldsymbol{\beta}\cdot\boldsymbol{\beta}^\top = \boldsymbol{I}$. The coordinate transformation of a rank-2 tensor is given by $\boldsymbol{T}^* = \boldsymbol{\beta}\boldsymbol{T}\boldsymbol{\beta}^\top$, with $T_{ij}^* = \beta_{ik}T_{kl}\beta_{jl}$.

The permutation symbol $e_{ijk}$ is defined as follows:
$e_{ijk} = \begin{cases}
1 & \text{if } (i,j,k) \text{ is an even permutation of } (1,2,3) \\
-1 & \text{if } (i,j,k) \text{ is an odd permutation of } (1,2,3) \\
0 & \text{otherwise}
\end{cases}$
which can define the vector product as $\boldsymbol{u}\times\boldsymbol{v} = e_{ijk}u_jv_k\boldsymbol{e}_i$. 

> An important property: $e_{ijk}e_{lmk} = \delta_{il}\delta_{jm} - \delta_{im}\delta_{jl}$.

The dual vector of a skew tensor $\boldsymbol{W}$ is defined as $\boldsymbol{w} = -\frac{1}{2}e_{ijk}W_{jk}\boldsymbol{e}_i$, and the skew tensor can be expressed in terms of its dual vector as $W_{ij} = e_{ijk}w_k$. Then we can express the vector product as $\boldsymbol{W}\boldsymbol{u} = \boldsymbol{w}\times\boldsymbol{u}$.

### 1.2.2 Vector and tensor calculus

The gradient operator is defined as a vector

$$
\nabla = \frac{\partial}{\partial \boldsymbol{x}}=\boldsymbol{e}_i\frac{\partial}{\partial x_i}
$$

For a vector field $\boldsymbol{u}(\boldsymbol{x})$, the gradient is a rank-2 tensor defined as

$$
\nabla\boldsymbol{u} = \boldsymbol{u}\otimes\nabla = \frac{\partial u_i}{\partial x_j}\boldsymbol{e}_i\otimes\boldsymbol{e}_j = u_{i,j}\boldsymbol{e}_i\otimes\boldsymbol{e}_j
$$

The divergence is given by

$$
\nabla\cdot\boldsymbol{u} = \text{tr}(\nabla\boldsymbol{u}) = \frac{\partial u_i}{\partial x_i}
$$

The Laplace operator is defined as

$$
\nabla^2 = \nabla\cdot\nabla = \frac{\partial^2}{\partial x_i \partial x_i}
$$

### 1.2.3 Integral theorems

The **divergence theorem** states that if a tensor $\boldsymbol{A}$ is continuously differentiable in $\Omega$, then:

$$
\iint_\Omega \nabla\cdot\boldsymbol{A} \, d\Omega = \int_{\Gamma} \boldsymbol{n}\cdot\boldsymbol{A} \, d\Gamma
$$

where $\Omega$ is a domain bounded by $\Gamma$ with outward normal $\boldsymbol{n}$. A variant is the gradient theorem, which states:

$$
\iint_\Omega \nabla\boldsymbol{A} \, d\Omega = \int_{\Gamma} \boldsymbol{n}\otimes\boldsymbol{A} \, d\Gamma
$$

The **Reynolds transport theorem** states that for $\boldsymbol{f}(\boldsymbol{x},t)$, we have:

$$
\frac{\rm d}{{\rm d}t}\iint_{\Omega} \boldsymbol{f} \, d\Omega = \iint_{\Omega} \frac{\partial \boldsymbol{f}}{\partial t} \, d\Omega + \int_{\Gamma} (\boldsymbol{n}\cdot\boldsymbol{v})\boldsymbol{f} \, d\Gamma
$$

where $\boldsymbol{v}$ is the velocity of the boundary $\Gamma$ and the second term on the RHS is **the convection term**. 

The **integration by parts** states that

$$
\iint_\Omega \nabla u\cdot\boldsymbol{v} \, d\Omega = \int_{\Gamma} u(\boldsymbol{v}\cdot\boldsymbol{n}) \, d\Gamma - \iint_\Omega u(\nabla\cdot\boldsymbol{v}) \, d\Omega
$$

If we replace $\boldsymbol{v}$ with $\nabla v$, we have the following Green's identity:

$$
\iint_\Omega \nabla u\cdot\nabla v \, d\Omega = \int_{\Gamma} u(\nabla v\cdot\boldsymbol{n}) \, d\Gamma - \iint_\Omega u\nabla^2 v \, d\Omega
$$