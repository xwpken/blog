# Chapter 5 Finite element formulation for one-dimensional problem

**Discretization** : Weak form ----> a finite number of discrete equations (System equations)

## Discrete equations for arbitrary boundary conditions

The weak form for sress analysis:

Find $u(x)\in U$ such that

$\int_{\Omega}(\frac{dw}{dt})^TAE\frac{du}{dx}dx-\int_{\Omega}w^Tbdx-(w^TA\overline{t})|_{\Gamma_t}=0$, $\forall w\in U_0$

Mesh and construct the approximate functions over individual element domains

The integral over $\Omega$ can be expressed as a sum of integrals over element domains

$\sum_{e=1}^{n_{el}}\{\int_{\Omega^e}(\frac{dw^e}{dt})^TA^eE^e\frac{du^e}{dx}dx-\int_{\Omega^e}{w^e}^Tbdx-({w^e}^TA^e\overline{t})|_{\Gamma_t^e}\}=0$, $\forall w\in U_0$

For $u^e(x)=\pmb{N}^e\pmb{d}^e$, $\frac{du^e}{dx}=\pmb{B}^e\pmb{d}^e$ and ${w^e}^T={\pmb{w}^e}^T{\pmb{N}^e}^T$, $(\frac{d{w^e}}{dx})^T={\pmb{w}^e}^T{\pmb{B}^e}^T$ and

conduct the partition 
$\pmb{d}=\left\{\begin{matrix}\overline{\pmb{d}}_E & {\pmb{d}}_F\end{matrix}\right\}^T$, 
$\pmb{w}=\left\{\begin{matrix}{\pmb{w}}_E & {\pmb{w}}_F\end{matrix}\right\}^T=\left\{\begin{matrix}0 & {\pmb{w}}_F\end{matrix}\right\}^T$

where $E$ contains the nodal values on the essential boundaries, $F$ contains all the remain nodal vlaues

$\sum_{e=1}^{n_{el}}{\pmb{w}^e}^T\{\int_{\Omega^e}{\pmb{B}^e}^TA^eE^e{\pmb{B}^e}dx\pmb{d}^e-\int_{\Omega^e}{\pmb{N}^e}^Tbdx-({\pmb{N}^e}^TA^e\overline{t})|_{\Gamma_t^e}\}=0$, $\forall \pmb{w}_F$

Use the gather matrices $\pmb{L}^e$ to replace ${\pmb{w}^e}^T$ and $\pmb{d}^e$ with $\pmb{w}$ and $\pmb{d}$, i.e. $\pmb{d}^e=\pmb{L}^e\pmb{d}$ and $\pmb{w}^e=\pmb{L}^e\pmb{w}$



$\sum_{e=1}^{n_{el}}{\pmb{w}^e}^T\{\int_{\Omega^e}{\pmb{B}^e}^TA^eE^e{\pmb{B}^e}dx\pmb{d}^e-\int_{\Omega^e}{\pmb{N}^e}^Tbdx-({\pmb{N}^e}^TA^e\overline{t})|_{\Gamma_t^e}\}=0$, $\forall \pmb{w}_F$

${\pmb{w}}^T[(\sum_{e=1}^{n_{el}}{\pmb{L}^e}^T\pmb{K}^e{\pmb{L}^e})\pmb{d}-(\sum_{e=1}^{n_{el}}{\pmb{L}^e}^T\pmb{f}^e)]={\pmb{w}}^T(\pmb{K}\pmb{d}-\pmb{f})=0$

where $\pmb{K}^e=\int_{\Omega^e}{\pmb{B}^e}^TA^eE^e{\pmb{B}^e}dx$ , $\pmb{f}^e=\int_{\Omega^e}{\pmb{N}^e}^Tbdx+({\pmb{N}^e}^TA^e\overline{t})|_{\Gamma_t^e}$

Let $\pmb{r}=\pmb{K}\pmb{d}-\pmb{f}$, the above system can be written as 

$\pmb{w}^T\pmb{r}=0$, $\forall \pmb{w}_F$, for $\pmb{w}_E=0$

$\left[\begin{matrix}\pmb{w}_E & {\pmb{w}}_F\end{matrix}\right]^T\left[\begin{matrix}\pmb{r}_E \\ {\pmb{r}}_F\end{matrix}\right]={\pmb{w}}_F^T{\pmb{r}}_F=0$

From the scalar product theorem, for ${\pmb{w}}_F$ is arbitary, ${\pmb{r}}_F=0$

$\pmb{r}=\left[\begin{matrix}\pmb{r}_E \\ 0\end{matrix}\right]=\left[\begin{matrix}\pmb{K}_E & \pmb{K}_{EF}\\\pmb{K}_{EF}^T & \pmb{K}_F\end{matrix}\right]\left[\begin{matrix}\overline{\pmb{d}}_E \\ \pmb{d}_F\end{matrix}\right]-\left[\begin{matrix}{\pmb{f}}_E \\ \pmb{f}_F\end{matrix}\right]$

$\left[\begin{matrix}\pmb{K}_E & \pmb{K}_{EF}\\\pmb{K}_{EF}^T & \pmb{K}_F\end{matrix}\right]\left[\begin{matrix}\overline{\pmb{d}}_E \\ \pmb{d}_F\end{matrix}\right]=\left[\begin{matrix}{\pmb{f}}_E+\pmb{r}_E \\ \pmb{f}_F\end{matrix}\right]$

Two-step approch:

Step 1: $\pmb{K}_F\pmb{d}_F=\pmb{f}_F-\pmb{K}_{EF}^T\pmb{d}_E$

Step 2: $\pmb{r}_E=\pmb{K}_E\overline{\pmb{d}}_E+\pmb{K}_{EF}\pmb{d}_F-\pmb{f}_E$

-----

## Two-point boundary value problem with generalized boundary conditions

+ General weak form for 1D problems-penalty method

Find $\theta(x) \in H^1$ such that

$\int_{\Omega}\frac{dw}{dx}A\kappa\frac{d\theta}{dx}dx-\int_{\Omega}wfdx-wA(\overline{\Phi}-\beta(\theta-\overline{\theta}))|_{\Gamma}=0$, $\forall w \in H^1$

Discrete equations:

$\sum_{e=1}^{n_{el}}{\pmb{w}^e}^T\{\int_{\Omega^e}{\pmb{B}^e}^TA^e\kappa^e{\pmb{B}^e}dx\pmb{d}^e+({\pmb{N}^e}^TA^e\beta\pmb{N}^e)|_{\Gamma^e}\pmb{d}^e-\int_{\Omega^e}{\pmb{N}^e}^Tfdx-({\pmb{N}^e}^TA^e(\overline{\Phi}+\beta\overline{\theta}))|_{\Gamma^e}\}=0$, $\forall \pmb{w}$

$\pmb{w}^T(\pmb{K}\pmb{d}-\pmb{f})=0$, i.e. $\pmb{K}\pmb{d}=\pmb{f}$


+ General weak form for 1D problems-partition method

Find $\theta(x) \in U$ such that

$\int_{\Omega}\frac{dw}{dx}A\kappa\frac{d\theta}{dx}dx-\int_{\Omega}wfdx-wA(\overline{\Phi}-\beta(\theta-\overline{\theta}))|_{\Gamma_{\overline{\Phi}}}=0$, $\forall w \in U_0$

Discrete equations:

$\sum_{e=1}^{n_{el}}{\pmb{w}^e}^T\{\int_{\Omega^e}{\pmb{B}^e}^TA^e\kappa^e{\pmb{B}^e}dx\pmb{d}^e+({\pmb{N}^e}^TA^e\beta\pmb{N}^e)|_{\Gamma^e_{\Phi}}\pmb{d}^e-\int_{\Omega^e}{\pmb{N}^e}^Tfdx-({\pmb{N}^e}^TA^e(\overline{\Phi}+\beta\overline{\theta}))|_{\Gamma^e_{\Phi}}\}=0$, $\forall \pmb{w}_F$

-----

## Convergence of the FEM

Lebesque ($L_2$) norm:

$\Vert f(x)\Vert_{L_2}=(\int_{x_1}^{x_2}f^2(x)dx)^{\frac{1}{2}}$

which is the norm of function $f(x)$ and measures the magnitude of the function

**The error in a finite element solution**

$\Vert e \Vert_{L_2}=\Vert u^{ex}(x)-u^h(x) \Vert_{L_2}=(\int_{x_1}^{x_2}(u^{ex}(x)-u^h(x))^2dx)^{\frac{1}{2}}$

which can be considered as a root-mean-square of the error

**The normalized error in finite element solution**

$\overline{e}_{L_2}=\frac{\Vert u^{ex}(x)-u^h(x) \Vert_{L_2}}{\Vert u^{ex}(x)\Vert_{L_2}}=\frac{(\int_{x_1}^{x_2}(u^{ex}(x)-u^h(x))^2dx)^{\frac{1}{2}}}{(\int_{x_1}^{x_2}(u^{ex}(x))^2dx)^{\frac{1}{2}}}$

**The error in energy**

$\Vert e \Vert_{en}=\Vert u^{ex}(x)-u^h(x) \Vert_{en}=(\frac{1}{2}\int_{x_1}^{x_2}E(\varepsilon^{ex}(x)-\varepsilon^h(x))^2dx)^{\frac{1}{2}}$

**The normalized error in energy**

$\overline{e}_{en}=\frac{\Vert u^{ex}(x)-u^h(x) \Vert_{en}}{\Vert u^{ex}(x)\Vert_{en}}=\frac{(\frac{1}{2}\int_{x_1}^{x_2}E(\varepsilon^{ex}(x)-\varepsilon^h(x))^2dx)^{\frac{1}{2}}}{(\frac{1}{2}\int_{x_1}^{x_2}E(\varepsilon^{ex}(x))^2dx)^{\frac{1}{2}}}$

### Convergence by numerical experiments

The error in the $L_2$ norm of the displacemnet: $\Vert e \Vert_{L_2}=Ch^{p+1}$

The error in energy: $\Vert e \Vert_{en}=Ch^{p}$

where $h$ is the element size and $p$ is the order of the complete polynominal in FE

<blockquote>P117 Gui and Babuska (1986)</blockquote>

$\Vert e \Vert_{en}=Ch^{\beta}$

where $\beta=min(p,\lambda-\frac{1}{2})$, $\lambda>\frac{1}{2}$, $p\geq 1$

Three requirements: 

**Integrability** of the exact solution;

**Continuity** with square integrable derivatives of the FE solution;

**Completeness** of the trial solution

### Convergence by analysis

The **approximate** character of the FE solution $\Rightarrow$ replace the infinite dimensional spaces $U$, $U_0$ with finite-dimensional subspaces  $U^h$, $U^h_0$

$u^h(x)$ minimzies the energy norm of error: $\Vert u-u^h \Vert_{en}=\min\limits_{u^*\in U^h}\Vert u-u^* \Vert_{en}$

<blockquote>P120 The energy norm of error for finite elements of order $p$ is bounded by $Ch^p$ provided that $\vert \frac{d^{p+1}u}{dx^{p+1}} \vert \leq \alpha$, and $C$ is indepenedent of $h$</blockquote>
