# Chapter 6 Strong and weak forms for multidimensional scalar field problems

## Divergence theorem and Green's formula

**Green's theorem**:

If $\theta(x,y)\in C^0$ and integrable, then

$\int\limits_{\Omega}\vec{\nabla}\theta d\Omega=\oint\limits_{\Gamma}\theta \vec{n}d\Gamma$ or $\int\limits_{\Omega}\pmb{\nabla}\theta d\Omega=\oint\limits_{\Gamma}\theta \vec{n}d\Gamma$

**Divergence theorem**:

If $\vec{q}$ is $C^0$ and integrable, then

$\int\limits_{\Omega}\vec{\nabla}\cdot\vec{q}d\Omega=\oint\limits_{\Gamma}\vec{q}\cdot\vec{n}d\Gamma$ or $\int\limits_{\Omega}\pmb{\nabla}^T\pmb{q}d\Omega=\oint\limits_{\Gamma}\pmb{q}^T\pmb{n}d\Gamma$

**Green's formula** / **Green's first identity**:

$\int\limits_{\Omega}w\vec{\nabla}\cdot\vec{q}d\Omega=\oint\limits_{\Gamma}w\vec{q}\cdot\vec{n}d\Gamma-\int\limits_{\Omega}\vec{\nabla}w\cdot\vec{q}$

## Strong form and weak form for heat conduction

### Strong form

Energy balance: $\vec{\nabla}\cdot\vec{q}-s=0$ on $\Omega$

Fourier's law: $\vec{q}=-k\vec{\nabla}T$ on $\Omega$

natural BC: $q_n=\vec{q}\cdot\vec{n}=\overline{q}$ on $\Gamma_q$

essential BC: $T=\overline{T}$ on $\Gamma_T$

### Weak form

Find $T\in U$ such that

$\int\limits_{\Omega}\vec{\nabla}w\cdot \vec{q}=\int\limits_{\Gamma_q}w\vec{q}d\Gamma-\int\limits_{\Omega}ws d\Omega$ for $\forall w\in U_0$

or in matrix notation

$\int\limits_{\Omega}(\pmb{\nabla}w)^T\pmb{q}=\int\limits_{\Gamma_q}w\vec{q}d\Gamma-\int\limits_{\Omega}ws d\Omega$ for $\forall w\in U_0$


$\int\limits_{\Omega}(\pmb{\nabla}w)^T\pmb{D}\pmb{\nabla}T=-\int\limits_{\Gamma_q}w\vec{q}d\Gamma+\int\limits_{\Omega}ws d\Omega$ for $\forall w\in U_0$

where the generalized Fourier's law $\pmb{q}=-\pmb{D}\pmb{\nabla}T$ is used


## Strong form and weak form for scalar steady-state Advection-Diffusion equation in 2D

General form of Advection-Diffusion equation:

$\vec{\nabla}\cdot (\theta\vec{v})+\vec{\nabla}\cdot\vec{q}-s=0$

Introduce the **continuity equation** for steady-state problems of incompressible materials

$\vec{\nabla}\cdot\vec{v}=0$

Then the above equation becomes

$\vec{v}\cdot\vec{\nabla}\theta+\vec{\nabla}\cdot\vec{q}-s=0$

Use the matrix notations and the generalized Fourier's law, the strong form can be written as

### Strong form

$\pmb{v}^T\pmb{\nabla}\theta-\pmb{\nabla}^T(\pmb{D}\pmb{\nabla}\theta)-s=0$ on $\Omega$

$\theta=\overline{\theta}$ on $\Gamma_{\theta}$

$q_n=\pmb{q}^T\pmb{n}=\overline{q}$ on $\Gamma_{q}$

### Weak form

find the trial solutions $\theta(x,y)\in U$ such that

$\int\limits_{\Omega}w\pmb{v}^T\pmb{\nabla}\theta d\Omega+\int\limits_{\Omega}(\pmb{\nabla}w)^T\pmb{D}\pmb{\nabla}\theta d\Omega+\int\limits_{\Gamma_q}w\vec{q}d\Gamma-\int\limits_{\Omega}ws d\Omega=0$

