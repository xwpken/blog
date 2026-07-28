# Chapter 1 Introduction to Vectors and Tensors
> Reference: 
> Holzapfel, Gerhard A. "Nonlinear solid mechanics: a continuum approach for engineering science." (2002): 489-490.


## 1.1 Algebra of Vectors

**Kronecker delta**  $\,\delta_{ij} = \boldsymbol{e}_i\cdot\boldsymbol{e}_j$

**Cross product** $\,\boldsymbol{u}\times\boldsymbol{v}\,$ or $\,\boldsymbol{u}\wedge
\boldsymbol{v}\,$

$\epsilon_{ijk}$


## 1.2 Algebra of Tensors

### second-order tensor 

$\boldsymbol{v} = \boldsymbol{A}\boldsymbol{u}$ $\rightarrow$ **linear transformation**

### Tensor product/dyad

**tensor product** (or direct or matrix) or the **dyad** of **vectors** $\boldsymbol{u}$ and $\boldsymbol{v}$ $\rightarrow$ $\boldsymbol{u}\otimes\boldsymbol{v}$ or $\boldsymbol{u}\boldsymbol{v}$

$(\boldsymbol{u}\otimes\boldsymbol{v})\boldsymbol{w} = \boldsymbol{u}(\boldsymbol{v}\cdot\boldsymbol{w})$ 
 
$(\boldsymbol{u}\otimes\boldsymbol{v})(\boldsymbol{w}\otimes\boldsymbol{x}) = (\boldsymbol{v}\cdot\boldsymbol{w})\boldsymbol{u}\otimes\boldsymbol{x}$

### dyadic

**linear combination of dyads** with scalar coefficients

$\boldsymbol{A} = A_{ij}\boldsymbol{e}_i\otimes\boldsymbol{e}_j$


or with **matrix notation**

$[\boldsymbol{A}] = 
\begin{bmatrix}
A_{11} && A_{12} && A_{13} \\
A_{21} && A_{22} && A_{23} \\
A_{31} && A_{32} && A_{33} \\
\end{bmatrix}$

$A_{ij} = \boldsymbol{e}_i\cdot\boldsymbol{A}\boldsymbol{e}_j$

### dot product：

#### dot product of tensors $\boldsymbol{AB}$

$(\boldsymbol{AB})_{ij} = A_{ik}B_{kj}$

$\boldsymbol{A}^2 = \boldsymbol{A}\boldsymbol{A}$





### Tranpose of $\boldsymbol{A}$: 

$\boldsymbol{v}\cdot\boldsymbol{A}^T\boldsymbol{u}=\boldsymbol{u}\cdot\boldsymbol{A}\boldsymbol{v} = \boldsymbol{A}\boldsymbol{v}\cdot\boldsymbol{u}$

$(\boldsymbol{AB})^T = \boldsymbol{B}^T\boldsymbol{A}^T$

$(\boldsymbol{u}\otimes\boldsymbol{v})^T = \boldsymbol{v}\otimes\boldsymbol{u}$

$(\boldsymbol{A}^T)_{ij} = A_{ji}$


### Trace and contraction

$tr(\boldsymbol{u}\otimes\boldsymbol{v}) = \boldsymbol{u}\cdot\boldsymbol{v} = u_iv_i$

$tr(\boldsymbol{A}) = A_{ij}tr(\boldsymbol{e}_i\otimes\boldsymbol{e}_j)= A_{ii}$

$tr(\boldsymbol{AB}) = tr(\boldsymbol{BA})$

#### contraction

> Identify two indices and sum over them as dummy indices

$\boldsymbol{A}:\boldsymbol{B} = tr(\boldsymbol{A}^T\boldsymbol{B}) = A_{ij}B_{ij} = \boldsymbol{B}:\boldsymbol{A}$

$\boldsymbol{A}:(\boldsymbol{BC}) = (\boldsymbol{B}^T\boldsymbol{A}):\boldsymbol{C} = (\boldsymbol{A}\boldsymbol{C}^T):\boldsymbol{B}$

$(\boldsymbol{u}\otimes\boldsymbol{v}):(\boldsymbol{w}\otimes\boldsymbol{x}) = (\boldsymbol{u}\cdot\boldsymbol{w})(\boldsymbol{v}\cdot\boldsymbol{x})$

norm of the tensor:

$|A| = (A:A)^{\frac{1}{2}} = (A_{ij}A_{ij})^{\frac{1}{2}}\geq 0$

### Determinant and inverse of a tensor

$det\boldsymbol{A}$ = $det [\boldsymbol{A}]$

$det(\boldsymbol{AB}) = det\boldsymbol{A}det\boldsymbol{B}$

$det(\boldsymbol{A}^T) = det(\boldsymbol{A})$

**singular** $\rightarrow$ $det(\boldsymbol{A}) = 0$

$(\boldsymbol{AB})^{-1} = \boldsymbol{B}^{-1}\boldsymbol{A}^{-1}$

$(\alpha\boldsymbol{A})^{-1} = \frac{1}{\alpha}\boldsymbol{A}^{-1}$

$(\boldsymbol{A}^{-1})^T = (\boldsymbol{A}^T)^{-1} = \boldsymbol{A}^{-T}$

$\boldsymbol{A}^{-2} = \boldsymbol{A}^{-1}\boldsymbol{A}^{-1}$

$det(\boldsymbol{A}^{-1}) = (det\boldsymbol{A})^{-1}$

### Orthogonal tensor

$\boldsymbol{Q}\boldsymbol{u}\cdot\boldsymbol{Q}\boldsymbol{v}=\boldsymbol{u}\cdot\boldsymbol{v}$

#### Properties

$\boldsymbol{Q}^T\boldsymbol{Q}=\boldsymbol{Q}\boldsymbol{Q}^T=\boldsymbol{I}$

$\boldsymbol{Q}^T=\boldsymbol{Q}^{-1}$

$det(\boldsymbol{Q}^T\boldsymbol{Q})=(det\boldsymbol{Q})^2=1$

$det\boldsymbol{Q}=+1$$\quad\rightarrow\quad$ proper orthogonal $\rightarrow$ rotation

$det\boldsymbol{Q}=-1$$\quad\rightarrow\quad$ improper orthogonal $\rightarrow$ reflection

### Symmetric and skew tensors

Any tensor $\boldsymbol{A}$ can be decomposed into a symmetric tensor $\boldsymbol{S}$ and a skew/antisymmetric tensor $\boldsymbol{W}$


$\boldsymbol{A} = \boldsymbol{S}+\boldsymbol{W}$

$\boldsymbol{S} = \displaystyle\frac{1}{2}(\boldsymbol{A}+\boldsymbol{A}^T)$

$\boldsymbol{W} = \displaystyle\frac{1}{2}(\boldsymbol{A}-\boldsymbol{A}^T)$

#### some properties:

$\boldsymbol{S}:\boldsymbol{W}=0$

$\boldsymbol{W}\boldsymbol{u}= \boldsymbol{w} \times \boldsymbol{u}$

> $|w|=\displaystyle\frac{1}{\sqrt{2}}|\boldsymbol{W}|$

where $\boldsymbol{w}=-\displaystyle\frac{1}{2}\varepsilon_{ijk}W_{ij}\boldsymbol{e}_k$


### Projection, spherical and deviatoric tensors

#### project tensor

which applied to any vector $\boldsymbol{u}$ and map it into the direction of $\boldsymbol{e}$; or onto the plane normal to $\boldsymbol{e}$

$\boldsymbol{u}_{||}=(\boldsymbol{u}\cdot\boldsymbol{e})\boldsymbol{e} = (\boldsymbol{e}\otimes\boldsymbol{e})\boldsymbol{u}= \underbrace{\boldsymbol{P}^{||}_e}_{project\,tensor}\boldsymbol{u}$

$\boldsymbol{u}_{\bot}=\boldsymbol{u}- \boldsymbol{u}_{||}= (\boldsymbol{I}-\boldsymbol{e}\otimes\boldsymbol{e})\boldsymbol{u}= \underbrace{\boldsymbol{P}^{\bot}_e}_{project\,tensor}\boldsymbol{u}$

##### some properties

$\boldsymbol{P}=\boldsymbol{P}^n$

### spherical part and deviatoric part

$\boldsymbol{A}=\underbrace{\alpha\boldsymbol{I}}_{spherical}+\underbrace{dev\boldsymbol{A}}_{deviatoric}$

where:

$\alpha = \displaystyle\frac{1}{3}tr\boldsymbol{A}= \displaystyle\frac{1}{3}A_{ii}$

$dev\boldsymbol{A} = \boldsymbol{A}-\frac{1}{3}tr\boldsymbol{A}\boldsymbol{I}$

$tr(dev\boldsymbol{A})=0$

## Higher-order Tensors