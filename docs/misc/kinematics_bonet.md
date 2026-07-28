# Chapter 4 Kinematics

> Reference: 
> Bonet, Javier, Antonio J. Gil, and Richard D. Wood. Nonlinear solid mechanics for finite element analysis: statics. Cambridge University Press, 2016.

## 4.2 The Motion

motion of particle $\rightarrow$ mapping $\phi$

$\boldsymbol{x} = \phi(\boldsymbol{X},t)$

## 4.3 Material and spatial descriptions

Lagrangian (Material, $u = u(\boldsymbol{X},t)$) and Eulerian (Spatial, $u = u(\boldsymbol{x},t)$) descriptions

> The governing equations must be formulated using a **spatial description first** !!!

> Spatial quantities can be expressed in term of initial coordinates. (Description can be transformed)

## 4.4 Deformation gradient

For two neighboring particles:

relative material position $\rightarrow$ $\boldsymbol{F}$  $\rightarrow$relative position

**deformation gradient tensor**: $\boldsymbol{F} = \frac{\partial \phi}{\partial \boldsymbol{X}} = \nabla_0 \phi$

$d\boldsymbol{x} = \boldsymbol{F}d\boldsymbol{X}$ $\rightarrow$ transforms vectors in **reference** config into **current** config

$\boldsymbol{F} = \frac{\partial\boldsymbol{x}}{\partial \boldsymbol{X}}$
$\quad$
$F_{ij} = \frac{\partial x_i}{\partial X_j}$

**Inverse of** $\boldsymbol{F}$ $\rightarrow$ $\boldsymbol{F}^{-1} = \frac{\partial\boldsymbol{X}}{\partial \boldsymbol{x}} = \nabla\phi^{-1}$
$\quad$
$F_{ji}^{-1} = \frac{\partial X_j}{\partial x_i}$

## 4.5 Strain

A **general measure** of deformation $\rightarrow$ **scalar product** of $d\boldsymbol{X}_1$ and $d \boldsymbol{X}_2$

> refer to P104 Remark 4.3

$d\boldsymbol{x}_1\cdot d\boldsymbol{x}_2 = d\boldsymbol{X}_1\cdot \boldsymbol{C}d\boldsymbol{X}_2 = d\boldsymbol{X}_1\cdot \boldsymbol{F}^T\boldsymbol{F}d\boldsymbol{X}_2$

**right Cauchy-Green deformation tensor**: $\boldsymbol{C} = \boldsymbol{F}^T\boldsymbol{F}$ $\rightarrow$ material tensor quantity

$d\boldsymbol{X}_1\cdot d\boldsymbol{X}_2 = d\boldsymbol{x}_1\cdot \boldsymbol{b}^{-1}d\boldsymbol{x}_2$

**left Cauchy-Green deformation tensor**: $\boldsymbol{b} = \boldsymbol{F}\boldsymbol{F}^T$ $\rightarrow$ spatial tensor quantity

----

**Change in scalar product**: $\frac{1}{2}(d\boldsymbol{x}_1\cdot d\boldsymbol{x}_2-d\boldsymbol{X}_1\cdot d\boldsymbol{X}_2) = d\boldsymbol{X}_1\cdot \boldsymbol{E} d\boldsymbol{X}_2 = d\boldsymbol{x}_1\cdot\boldsymbol{e}d\boldsymbol{x}_2$

**Green-Lagrangian strain tensor**: $\boldsymbol{E} = \frac{1}{2}(\boldsymbol{C}-\boldsymbol{I})$

**Almansi-Eluerian strain tensor**: $\boldsymbol{e} = \frac{1}{2}(\boldsymbol{I}-\boldsymbol{b}^{-1})$

**Transformation**: $\boldsymbol{e} = \boldsymbol{F}^{-T}\boldsymbol{E}\boldsymbol{F}^{-1}$$\quad$$\boldsymbol{E}=\boldsymbol{F}^T\boldsymbol{e}\boldsymbol{F}$

## 4.6 Polar decomposition

$\boldsymbol{F} = \boldsymbol{R}\boldsymbol{U} = \boldsymbol{V}\boldsymbol{R}$

$\boldsymbol{R}$ $\rightarrow$ orthogonal rotation tensor i.e., $\boldsymbol{R^T}\boldsymbol{R}=\boldsymbol{I}$


## 4.7 Volume change

**Reference config**:

$d\boldsymbol{X}_i = dX_i\boldsymbol{E}_i$

$dV = dX_1 dX_2dX_3$

> $\boldsymbol{E}_1\cdot (\boldsymbol{E}_2\times\boldsymbol{E}_3)=+1$

**Current config**:

$d\boldsymbol{x}_i = \boldsymbol{F}dX_i = \frac{\partial\phi}{\partial X_i}dX_i$

$dv = d\boldsymbol{x}_1\cdot(d\boldsymbol{x}_2\times\boldsymbol{x}_3) = \frac{\partial\phi}{\partial X_1}\cdot(\frac{\partial \phi}{\partial X_2}\times\frac{\partial \phi}{\partial X_3})dX_1dX_2dX_3 = det(\boldsymbol{F})dV = JdV$

> density: $\rho_0 = \rho J$

## 4.8 Distortional component of the deformation gradient

Decompose the deformation gradient into a volumetric part and a distortional part, i.e.,

$\boldsymbol{F} = \boldsymbol{F}_v\cdot\boldsymbol{F}_d$

$J = det(\boldsymbol{F}) = det(\boldsymbol{F}_v)det(\boldsymbol{F}_d)$

No volume change in distortional(isochoric) part, so

$det(\boldsymbol{F}_d) = 1$

So, $\boldsymbol{F}_d = J^{-\frac{1}{3}}\boldsymbol{F}$ to ensure that $det(\boldsymbol{F_d})=(J^{-\frac{1}{3}})^3det(\boldsymbol{F})=J^{-1}J=1$

> $\boldsymbol{F}_v=J^{\frac{1}{3}}$

The distorrtional part of right Cauchy-Green tensor $\boldsymbol{C}$:

$\boldsymbol{C}_d = \boldsymbol{F}^T_d\boldsymbol{F}_d=J^{-\frac{2}{3}}\boldsymbol{C}=det(\boldsymbol{C})^{-\frac{1}{3}}\boldsymbol{C}$

## 4.9 Area change

Reference config: $d\boldsymbol{A}=dA\boldsymbol{N}$ $\quad$ $dV = d\boldsymbol{L}\cdot d\boldsymbol{A}$

Current config: $d\boldsymbol{a} = da\boldsymbol{n}$ $\quad$ $dv = d\boldsymbol{l}\cdot d\boldsymbol{a}$

$dv = JdV = Jd\boldsymbol{L}\cdot d\boldsymbol{A} = d\boldsymbol{l}\cdot d\boldsymbol{a} = \boldsymbol{F}d\boldsymbol{L}\cdot d\boldsymbol{a}$

$d\boldsymbol{a} = J\boldsymbol{F}^{-T}d\boldsymbol{A}$

## 4.10 Linearized kinematics

### Linearized deformation gradient

$D\boldsymbol{F}[\boldsymbol{u}]=\frac{\partial}{\partial \epsilon}|_{\epsilon=0}(\frac{\partial (\phi_t+\epsilon\boldsymbol{u})}{\partial \boldsymbol{X}})=\frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} = \nabla_0\boldsymbol{u}$

### Linearized strain

Green-Lagrangian strain:

$D\boldsymbol{E}[\boldsymbol{u}] = \frac{1}{2}\boldsymbol{F}^T[\nabla \boldsymbol{u}+(\nabla \boldsymbol{u})^T]\boldsymbol{F} = \boldsymbol{F}^T\epsilon\boldsymbol{F}$

> $\epsilon$ $\rightarrow$ small strain tensor

Linearized Cauchy-Green deformation tensor

right: $D\boldsymbol{C}[\boldsymbol{u}] = 2\boldsymbol{F}^T\epsilon\boldsymbol{F}$

left: $D\boldsymbol{b}[\boldsymbol{u}] = (\nabla \boldsymbol{u})\boldsymbol{b} + \boldsymbol{b}(\nabla\boldsymbol{u})^T$

### Linearized volume change

$DJ[\boldsymbol{u}] = Jdiv\boldsymbol{u}=Jtr\boldsymbol{\epsilon}$

$D(dv)[\boldsymbol{u}] = (tr\boldsymbol{\epsilon})dv$



## 4.11 Velocity and material time derivatives





## Rate of deformation

## Spin tensor

## Rate of change of volume

## Superimposed rigid body motions and objectivity


------

# Stress and equilibrium

## Cauchy stress tensor

traction vector: $\boldsymbol{t}(\boldsymbol{n}) = \lim\limits_{\Delta a\rightarrow 0}{\frac{\Delta\boldsymbol{P}}{\Delta a}}$

> $\boldsymbol{t}(-\boldsymbol{n}) = -\boldsymbol{t}(\boldsymbol{n})$

$\boldsymbol{t}(\boldsymbol{n}) = [\displaystyle\sum_{i,j=1}^3\sigma_{ij}(\boldsymbol{e}_i\otimes\boldsymbol{e}_j)]\boldsymbol{n} = \boldsymbol{\sigma}\boldsymbol{n}$

Cauchy stress tensor: $\boldsymbol{\sigma}=\displaystyle\sum_{i,j=1}^3\sigma_{ij}(\boldsymbol{e}_i\otimes\boldsymbol{e}_j)$

Expressed inb terms of principal directions?

> refer to P141


## Equilibrium

### Translational equilibrium

Sum of all forces acting on the body vanishes:

$\int_{\partial v}\boldsymbol{t}da+\int_{v}\boldsymbol{f}dv=0$

Further expressed in terms of **Cauchy stresses**:

$\int_{\partial v}\boldsymbol{\sigma}\boldsymbol{n}da+\int_{v}\boldsymbol{f}dv=0$

Using **Gauss theorem**:

$\int_{v}(div \boldsymbol{\sigma}+\boldsymbol{f})dv=0$

$div \boldsymbol{\sigma}+\boldsymbol{f} = \boldsymbol{0}$ $\rightarrow$ point-wise spatial equilibrium equation

The pointwise out-of-balance or **residual** force per volume:

$\boldsymbol{r} = div \boldsymbol{\sigma}+\boldsymbol{f}$


### Rotational equilibrium

> refer to P144


## Principle of virtual work

Equilibrium stated by virtual work: $\delta w = \boldsymbol{r}\cdot \delta \boldsymbol{v} = 0$ $\rightarrow$ $\boldsymbol{r} = \boldsymbol{0}$

> per unit volume and time done by the residual force 
$\boldsymbol{r}$ during the virtual motion $\boldsymbol{v}$ (aribitary virtual velocity)

Weak statement of the static equilibrium: $\delta W(\phi,\delta \boldsymbol{v}) = \int_{v}(div\boldsymbol{\sigma}+\boldsymbol{f})\cdot \delta \boldsymbol{v}dv=0$

**The spatial virtual work equation**:

$\delta W = \int_{v}\boldsymbol{\sigma}:\delta \boldsymbol{d}dv-\int_{v}\boldsymbol{f}\cdot\delta\boldsymbol{v}dv-\int_{\partial v}\boldsymbol{t}\cdot\delta\boldsymbol{v}da=0$

> $\delta \boldsymbol{d}$ $\rightarrow$ symmetric virtual rate of deformation


## Work conjugacy and alternative stress representations

### The kirchhoff Stress Tensor

**work conjugate** $\rightarrow$ the product (like $\boldsymbol{\sigma}$ and $\boldsymbol{d}$) gives work per unit current volume

Express the above spatial virtual work equation with respect to the **initial volume**

$\int_{V}J\boldsymbol{\sigma}:\delta \boldsymbol{d}dV-\int_{V}\boldsymbol{f}_0\cdot\delta\boldsymbol{v}dV-\int_{\partial V}\boldsymbol{t}_0\cdot\delta\boldsymbol{v}dA=0$

> $\boldsymbol{f}_0 = J \boldsymbol{f}$ $\rightarrow$ body force per unit undeformed volume
>
> $\boldsymbol{t}_0=\boldsymbol{t}(\frac{da}{dA})$
>
> $\frac{da}{dA} = \frac{J}{\sqrt{\boldsymbol{n}\cdot\boldsymbol{b}\boldsymbol{n}}} = J\sqrt{\boldsymbol{N}\cdot\boldsymbol{C}^{-1}\boldsymbol{N}}$

$\delta W_{int} = \int_{V}\boldsymbol{\tau}:\delta \boldsymbol{d}dV$

**the Kirchhoff stress tensor**: $\boldsymbol{\tau} = J\boldsymbol{\sigma}$

The work per unit mass is invariant, and $\rho = \frac{\rho_0}{J}$. so:

$\frac{1}{\rho}\boldsymbol{\sigma}:\boldsymbol{d}=\frac{1}{\rho_0}\boldsymbol{\tau}:\boldsymbol{d}$

### The First Piola-Kirchhoff Stress Tensor

$\delta W_{int} = \int_{V}(J\boldsymbol{\sigma}\boldsymbol{F}^{-T}):\delta\dot{\boldsymbol{F}}dV$

**the first Piola-Kirchhoff stress tensor**: $\boldsymbol{P} = J\boldsymbol{\sigma}\boldsymbol{F}^{-T}$

$\boldsymbol{P} = \displaystyle\sum_{i,I=1}^{3}P_{i,I}e_i\otimes\boldsymbol{E}_I$
$\quad$
$P_{iI}=\displaystyle\sum_{i,I=1}^{3}J\sigma_{ij}(\boldsymbol{F}^{-1})_{Ij}$

$\int_{V}\boldsymbol{P}:\delta \dot{\boldsymbol{F}}dv=\int_{V}\boldsymbol{f}\cdot\delta\boldsymbol{v}dv+\int_{\partial V}\boldsymbol{t}\cdot\delta\boldsymbol{v}da$

Reverse the weak formulation, we can get an equivalent version of differential equilibrium equation:

$\boldsymbol{r}_0 = J\boldsymbol{r} = DIV\boldsymbol{P}+\boldsymbol{f}_0=\boldsymbol{0} = \boldsymbol{\nabla}_0\boldsymbol{P}:\boldsymbol{I}+\boldsymbol{f}_0 = \frac{\partial \boldsymbol{P}}{\partial \boldsymbol{X}}:\boldsymbol{I}+\boldsymbol{f}_0$

> $d\boldsymbol{p} = \boldsymbol{\sigma}d\boldsymbol{a} = \boldsymbol{P}d\boldsymbol{A}$ $\rightarrow$ current force per unit area
>
> $\boldsymbol{P}$ is unsymmetric two-point tensor

### The Second Piola-Kirchhoff Stress Tensor

$d\boldsymbol{P} = \boldsymbol{F}^{-1}d\boldsymbol{p}$

> Material force vector $\leftarrow$ Spatial force vector

$d\boldsymbol{P} = \boldsymbol{S}d\boldsymbol{A}$ $\quad$ $\boldsymbol{S} = J\boldsymbol{F}^{-1}\boldsymbol{\sigma}\boldsymbol{F}^{-T}$

$\delta W_{int} = \int_{V}\boldsymbol{S}:\delta\dot{\boldsymbol{E}}dV$

**Material virtual work equation:**

$\int_{V}\boldsymbol{S}:\delta\dot{\boldsymbol{E}}dV = \int_{V}\boldsymbol{f}\cdot\delta\boldsymbol{v}dv+\int_{\partial V}\boldsymbol{t}\cdot\delta\boldsymbol{v}da$

Relations:

$\boldsymbol{\sigma} = J^{-1}\boldsymbol{P}\boldsymbol{F}^T$
$\quad$
$\boldsymbol{\sigma} = J^{-1}\boldsymbol{F}\boldsymbol{S}\boldsymbol{F}^T$

$\boldsymbol{S} = \boldsymbol{F}^{-1}\boldsymbol{\tau}\boldsymbol{F}^{-T}$
$\quad$
$\boldsymbol{\tau} = J^{-1}\boldsymbol{F}\boldsymbol{S}\boldsymbol{F}^T$

**Piola transformation**

> refer to P151

### Deviatoric and Pressure Components

> refer to P153







## Stress rates