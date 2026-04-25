# Chapter 4 Approximation of trial solutions, weight functions and Gauss quadrature for one-dimensional problems

Continuity + Completeness -----> Convergence of FEM

**Continuity** : The trial solutions and weight functions are suffciently smooth 

<blockquote>Based on the order of derivatives that appear in the weak form</blockquote>

**Completeness** : The capability of a series of functions to approximate a given smooth function with arbitary accuracy

## Approximation of trial solutions in element


### Two-node linear element

Linear ploynomial function : 

$\theta^e(x)=\alpha_0^e+\alpha_1^ex=\pmb{P}(x)\pmb{\alpha}^e$

where 
$\pmb{P}(x)=\left[
\begin{matrix}
1 & x
\end{matrix}
\right]$, 
$\pmb{\alpha}^e=\left[
\begin{matrix}
\alpha_0^e & \alpha_1^e
\end{matrix}
\right]^T$

Express $\pmb{\alpha}^e$ terms of the nodal values $\theta^e(x_i^e)\equiv \theta_i^e$

$\left[
\begin{matrix}
\theta_1^e \\
\theta_2^e
\end{matrix}
\right]=\left[
\begin{matrix}
1 & x_1^e\\
1 & x_2^e
\end{matrix}
\right]
\left[
\begin{matrix}
\alpha_0^e \\
\alpha_1^e
\end{matrix}
\right]$

$\pmb{d}^e=\pmb{M}^e\pmb{\alpha}^e$

$\pmb{\alpha}^e=(\pmb{M}^e)^{-1}\pmb{d}^e$

$\theta^e(x)=\pmb{P}(x)\pmb{\alpha}^e=\pmb{P}(x)(\pmb{M}^e)^{-1}\pmb{d}^e=\pmb{N}^e(x)\pmb{d}^e=\sum_{I=1}^{n_{en}}N_I^e(x)\theta_I^e$

where 
$\pmb{N}^e(x)=
\left[
\begin{matrix}
N_1^e & N_2^e
\end{matrix}
\right]$
is the element **shape function** matrix with
the interpolation property : $N_I^e(x_J^e)=\delta_{IJ}$

For the derivatives in the weak form

$\frac{d\theta^e}{dx}=\frac{d}{dx}(\pmb{N}^e\pmb{d}^e)=\frac{d\pmb{N}^e}{dx}\pmb{d}^e=
\left[
\begin{matrix}
\frac{dN_1^e}{dx} & \frac{dN_2^e}{dx}
\end{matrix}
\right]\pmb{d}^e=\pmb{B}^e\pmb{d}^e$

### Quadratic one-dimensional element

Three nodes in an element ...

<blockquote>In general, the third node placed at the center of the element perform better than others</blockquote>

### Direct construction of shape functions in one dimension

Lagrange interpolants ...

----

## Approximation of the weight functions

Galerkin FEM : use the same interpolants for the weight functions and trial solutions

$w^e(x)=\pmb{N}^e(x)\pmb{w}^e$

$\frac{dw^e}{dx}=\pmb{B}^e\pmb{w}^e$

-----

## Gloabl approximation and continuity

Global approximation of trial solutions and weight functions:

$\theta^h=\pmb{N}\pmb{d}=\sum_{I=1}^{n_{np}}N_Id_I$

$w^h=\pmb{N}\pmb{w}=\sum_{I=1}^{n_{np}}N_Iw_I$

$N_I\in H^1$ and satisfies the Kronecker delta property

Global shape functions are $C^0$ continuous ---> $\theta^h$ and $w^h$ are also $C^0$ continuous (Linear combination)

-----

## Guass quadrature

To find the following integral

$I=\int_a^bf(x)dx$

Use the a linear mapping to map the parent domian of Guass quadrature [$-1,1$] to physical domain [$a,b$]

$x=\frac{b-a}{1-(-1)}(\xi-1)+b=\frac{1-\xi}{2}a+\frac{\xi+1}{2}b=N_1(\xi)x_1+N_2(\xi)x_2$

$dx=\frac{b-a}{2}d\xi=Jd\xi$ , where $J=\frac{b-a}{2}$ is the Jacobi

The integral becomes

$I=\int_a^bf(x)dx=J\int_{-1}^1f(\xi)d\xi=J\hat{I}$


Approximate the integral $\hat{I}$ by

$\hat{I}=W_1f(\xi_1)+W_2f(\xi_2)+...=\pmb{W}^T\pmb{f}$

The function $f(\xi)$ is approximated by a polynominal

$f(\xi)=\alpha_1+\alpha_2\xi+...=\pmb{P}(\xi)\pmb{\alpha}$

where 
$\pmb{P}=
\left[
\begin{matrix}
1 & \xi & \xi^2 & \cdots
\end{matrix}
\right]$

Then $f(\xi)$ at the integral points can be expressed as 

$\pmb{f}=\pmb{M}\pmb{\alpha}$

where 
$\pmb{M}=
\left[
\begin{matrix}
1 & \xi_1 & \xi_1^2 & \cdots\\
1 & \xi_2 & \xi_2^2 & \cdots\\
\vdots & \vdots & \vdots & \vdots\\
1 & \xi_n & \xi_n^2 & \cdots
\end{matrix}
\right]$

The guass quadrature form:

$\hat{I}=\pmb{W}^T\pmb{M}\pmb{\alpha}$

The polynominal integration form:

$\hat{I}=\int_{-1}^{1}f(\xi)d\xi=
\int_{-1}^{1}
\left[
\begin{matrix}
1 & \xi & \xi^2 & \cdots
\end{matrix}
\right]
\pmb{\alpha}
d\xi=
\int_{-1}^{1}
\left[
\begin{matrix}
\xi & \frac{\xi^2}{2} & \frac{\xi^3}{3} & \cdots
\end{matrix}
\right]_{-1}^1
\pmb{\alpha}
d\xi=
\left[
\begin{matrix}
2 & 0 & \frac{2}{3} & \cdots
\end{matrix}
\right]_{-1}^1
\pmb{\alpha}=
\hat{\pmb{P}}\pmb{\alpha}$

To give the exact numerical integral results

$\pmb{W}^T\pmb{M}=\hat{\pmb{P}}$ -----> $\pmb{M}^T\pmb{W}=\hat{\pmb{P}}^T$

For $n_{gp}$ gauss points, there are $2n_{gp}$ adjustable parameters (weights and integral points)

$\pmb{M}$ (integral points) and $\pmb{W}$ (weights) are all unknown and $\hat{\pmb{P}}$ is already known ($P+1$ parameters for $P$-prder polynominal)

$2n_{gp}\geq P+1$

So for the $P$-order polynominal, the required number of intergral points is

$n_{gp}\geq \frac{P+1}{2}$