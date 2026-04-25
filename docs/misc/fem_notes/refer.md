# Surface integral

parameterized cell$\quad\leftarrow\quad$reference cell$\quad\leftarrow\quad$physical cell

## Step.1 Reference$\,\rightarrow\,$Parameterized

> Reference: 
> 1. https://en.wikipedia.org/wiki/Line_integral
> 2. https://en.wikipedia.org/wiki/Surface_integral
> 3. https://www.khanacademy.org/math/multivariable-calculus/integrating-multivariable-functions/surface-integrals-articles/a/surface-integrals
> 4. https://math.libretexts.org/Bookshelves/Calculus/Calculus_(OpenStax)/16%3A_Vector_Calculus/16.06%3A_Surface_Integrals

### Line intgral 

$\int_Cf(\xi,\eta)ds = \int_a^bf(\mathbf{r}(t))\underbrace{|\mathbf{r}^{\prime}(t)|dt}_{diff\,of\,arc}$

### Surface intgral 

$\iint_Af(\xi,\eta,\zeta)\mathrm{d}A=\iint_Tf(\mathbf{r}(s,t))\underbrace{\left\|\frac{\partial\mathbf{r}}{\partial s}\times\frac{\partial\mathbf{r}}{\partial t}\right\|\mathrm{d}s\mathrm{d}t}_{diff\,of\,curved\,surface}$

where $\mathbf{r}$ is the parametrization (maybe function)

> Above preparation is done in 'basis.py' in JAX-FEM (Dec 11,2023 updated)

## Step.2 Physical$\,\rightarrow\,$Reference

> Reference:
> 1. https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

### Nanson's formula/relation

$d\boldsymbol{A}=dA\boldsymbol{N}\quad dV = d\boldsymbol{L}\cdot d\boldsymbol{A}$(reference)

$d\boldsymbol{a}=dA\boldsymbol{n}\quad dv = d\boldsymbol{l}\cdot d\boldsymbol{a}$ (current)

$d\boldsymbol{l}=\boldsymbol{F}d\boldsymbol{L}\quad dv = det(\boldsymbol{F})dV$ (transformation)

$\boldsymbol{F}d\boldsymbol{L}\cdot d\boldsymbol{a} = det(\boldsymbol{F})d\boldsymbol{L}\cdot d\boldsymbol{A}$

$da\boldsymbol{n} = det(F)dA\boldsymbol{F}^{-T}\boldsymbol{N}$


### Line intgral

$\int_Lf(x,y)dl=\int_Cf(\xi,\eta)det(J)|\boldsymbol{N}\cdot \boldsymbol{J}^{-1}|ds$

### Surface intgral 

$\int_af(x,y,z)da=\int_Sf(\xi,\eta,\zeta)det(J)|\boldsymbol{N}\cdot \boldsymbol{J}^{-1}|dA$

> Above preparation is done in 'fe.py' in JAX-FEM (Dec 11,2023 updated)

## Step.3 Assemble

> Reference: https://en.wikipedia.org/wiki/Gaussian_quadrature

### Line intgral

$\int_Lf(x,y)dl=\int_a^b|\mathbf{r}^{\prime}(t)|det(J)|\boldsymbol{N}\cdot \boldsymbol{J}^{-1}|f(\mathbf{r}(t))dt =\sum_i\underbrace{\underbrace{W_i|\mathbf{r}^{\prime}(t)|}_{weights}det(J)|\boldsymbol{N}\cdot \boldsymbol{J}^{-1}|}_{nanson-scale}f(\mathbf{r}(t_i))$

### Surface intgral 

$\int_af(x,y,z)da=\iint_T\left\|\frac{\partial\mathbf{r}}{\partial s}\times\frac{\partial\mathbf{r}}{\partial t}\right\|f(\mathbf{r}(s,t))det(J)|\boldsymbol{N}\cdot \boldsymbol{J}^{-1}|dsdt= \sum_i\sum_j\underbrace{\underbrace{W_iW_j\left\|\frac{\partial\mathbf{r}}{\partial s}\times\frac{\partial\mathbf{r}}{\partial t}\right\|}_{weights}det(J)|\boldsymbol{N}\cdot \boldsymbol{J}^{-1}|}_{nanson-scale}f(\mathbf{r}(s_i,t_j))$

> Above preparation is done in 'problem.py' in JAX-FEM (Dec 11,2023 updated)






~
> This book maybe helpful: Advanced Topics in Finite Element Analysis of Structures: With Mathematica and MATLAB Computations