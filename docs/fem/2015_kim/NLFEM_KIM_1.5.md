# Chapter 1 Preliminary Concepts

> Last updated: 09 May 2026

## 1.5 Finite Element Analysis of Linear systems

### 1.5.1 Finite Element Approximation

Element level approximations (polynomial) of solutions

Linear polynomial approximation of 1D problem:

$$
u(x) = a_0 + a_1 x, \quad x \in [x_i, x_{i+1}]
$$

which can be used to derive the expression for the approximate solution in terms of nodal values:

$$
u(x) = \frac{x_{i+1} - x}{L^e} u_i + \frac{x - x_i}{L^e} u_{i+1}
$$

where $L^e = x_{i+1} - x_i$ is the length of the element.

### 1.5.2 Finite Element Equations for a One-Dimensional Problem

The differential equation along with boundary conditions is called the **boundary value problem (BVP)**.

In general, the stiffness matrix without imposing boundary conditions is **singular**.

### 1.5.3 Finite Element Equations for 3D Solid Element

The isoparametric mapping is not valid if the Jacobian is zero or negative anywhere in the element.

(interior point $\boldsymbol{\zeta}$ $\xrightarrow{mapping}$ exterior point $\boldsymbol{x}$) $\Rightarrow$ negative Jacobian

> the exterior point means outside of the physical element and vice versa.

(multiple points $\boldsymbol{\zeta}$ $\xrightarrow{mapping}$ single point $\boldsymbol{x}$) $\Rightarrow$ zero Jacobian

The integration over the physical domain can be transformed to the reference domain using:

$$
\int_{\Omega^e} f(\boldsymbol{x}) d\Omega = \int_{\hat{\Omega}} f(\boldsymbol{x}(\boldsymbol{\zeta})) |J| d\hat{\Omega}
$$

where $|J|$ is the determinant of the Jacobian matrix of the transformation from the reference domain $\hat{\Omega}$ to the physical domain $\Omega^e$.

The assemble process can be denoted using the symbol $\bigwedge(.)$

In general, $NG$-points Gaussian integration method integrates $(2NG-1)$-order plynomials exactly. 

The computational cost of Gaussian integration is proportional to $NG^2$ for 2D problems and $NG^3$ for 3D problems.

### 1.5.4 Finite Element Equations for 2D Plane-Strain Quadrilateral Element

~