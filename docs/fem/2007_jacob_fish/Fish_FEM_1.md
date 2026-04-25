# Chapter 2  Direct Approach for Discrete Systems

## Basic steps for FEM:

Step 1. Preprocessing

Step 2. Element formulation

Step 3. Assembly

Step 4. Solving equations

Step 5. Postprocessing

## Bar element

### Step 2. Describe the behavior of an element 

Content: relate 'nodal internal forces' to 'nodal displacements'

nodal internal forces 

$\downarrow$ ------- Linearity

stress

$\downarrow$ ------- Hook's Law

strain

$\downarrow$ ------- Linearity

nodal displacements 

$\downarrow$ ------- Equilibrium of the element

 element formulation


<blockquote>P13 sign convention for 'internal axial force' and 'nodal internal forces'</blockquote>

-------


### Step 3. Assembly

**Essence**: equilibrium equations for each node + compatibility (local and global)

<blockquote>P15 At each node, either the external forces or the nodal dsiplacements are known, but not both</blockquote>

<blockquote>P16 The sum of internal element forces is equal to that of the external forces and reactions</blockquote>

**Basic idea**: matrix scatter and add (argument,add)

**Program implentation**: Directly assembly (direct add based on global id)

**Equations for assembly**:


nodal displacements of each element: $\pmb{d}^e=\pmb{L}^e\pmb{d}$

where $\pmb{L}^e$ gather matrices, $\pmb{d}$ global displacement matrix

$\pmb{K}^e\pmb{d}^e=\pmb{K}^e\pmb{L}^e\pmb{d}=\pmb{F}^e$

Note that $(\pmb{L}^e)^T$ scatters the nodal forces into global matrix

$\sum_{e=1}^{n_e}(\pmb{L}^e)^T\pmb{F}^e=\pmb{f}+\pmb{r}$

where $\pmb{f}$ prescribed external force, $\pmb{r}$ reactions at **essential boundary...**

$(\pmb{L}^e)^T\pmb{K}^e\pmb{L}^e\pmb{d}=(\pmb{L}^e)^T\pmb{F}^e$, $e=1,...,n^e$

adding above element equations

$\pmb{K}\pmb{d}=\pmb{f}+\pmb{r}$

where $\pmb{K}=\sum_{e=1}^{n_e}(\pmb{L}^e)^T\pmb{K}^e\pmb{L}^e$ is the global stiffness matrix

<blockquote>Above equations are equivalent to directly assembly and matrix scatter and add</blockquote>

------
### Impose boundary conditions

+ Global system partition

$\left[\begin{matrix}
\pmb{K}_{E} & \pmb{K}_{EF}\\
\pmb{K}_{EF}^T & \pmb{K}_{F} 
\end{matrix}\right]
\left[\begin{matrix}
\overline{\pmb{d}}_{E}\\
\pmb{d}_{F}
\end{matrix}\right]=
\left[\begin{matrix}
\pmb{r}_{E}\\
\pmb{f}_{F}
\end{matrix}\right]$

where E-nodes for displacements known (essential), F-nodes for displacements unknown (free)

Solve $\pmb{d}_{F}$ from second row then taken into first row to get $\pmb{r}_{E}$
+ Draw 0 and set 1

$\left[\begin{matrix}
1 & 0 & 0\\
0 & {K}_{22} & {K}_{23} \\
0 & {K}_{32} & {K}_{33} 
\end{matrix}\right]
\left[\begin{matrix}
u_{1}\\
u_{2}\\
u_{3}
\end{matrix}\right]=
\left[\begin{matrix}
\overline{u}_1\\
f_{2}-\overline{u}_1K_{21}\\
f_3-\overline{u}_1K_{31}
\end{matrix}\right]$

where $\overline{u}_1$ is the prescribed displacement

Solve $u_i$ then back substitution to get $r_1$
+ Penalty method

$\left[\begin{matrix}
\beta & {K}_{12} & {K}_{13}\\
{K}_{21} & {K}_{22} & {K}_{23} \\
{K}_{31} & {K}_{32} & {K}_{33} 
\end{matrix}\right]
\left[\begin{matrix}
u_{1}\\
u_{2}\\
u_{3}
\end{matrix}\right]=
\left[\begin{matrix}
\beta\overline{u}_1\\
f_2\\
f_3
\end{matrix}\right]$

where $\beta$ is a very large number

Almost the same as 'Draw 0 and set 1'

<blockquote>For its tendency to decrease the conditioning of the equations，only applied for matrices of moderate size (up to about 10000 unknowns)</blockquote>
------
### Application to other linear systems

Above theory applicable for system characterized by:

1 A balance or conservation law for the flux

2 A linear law relating the flux to the potential

3 A continuous potential (i.e. a compatible potential)

e.g. 

mechanical bar (pot: displacement, flux: stress); 

steady-state electrical flow in a circuit (pot: voltages, flux: current); 

fluid flow in a hydraulic piping system (pot: pressure, flux: flow rate)...

-----

### Transformation law

For **rotation from one coordinate to another** or **scatter operation**

**Objective**: Get the expression of $\overline{\pmb{K}^e}$

**Known**: 

transformation matrix $\pmb{T}^e$ 

$\widehat{\pmb{d}^e}=\pmb{T}^e\overline{\pmb{d}^e}$

$\widehat{\pmb{F}^e}=\widehat{\pmb{K}^e}\widehat{\pmb{d}^e}$

$\delta\widehat{\pmb{d}^e}$ is an infinite small element displacement ($\widehat{\pmb{F}^e}$ remains constant)

$\delta W_{int}=\delta\widehat{\pmb{d}^e}^T\widehat{\pmb{F}^e}=\delta\widehat{u_1^e}\widehat{F_1^e}+\delta\widehat{u_2^e}\widehat{F_2^e}$ is the work done by the internal forces

The work should be the same in two coordinates system

$\delta\widehat{\pmb{d}^e}^T\widehat{\pmb{F}^e}=\delta\overline{\pmb{d}^e}^T\overline{\pmb{F}^e}=\delta{\overline{\pmb{d}^e}}^T{\pmb{T}^e}^T\widehat{\pmb{F}^e}$

So $\delta\overline{\pmb{d}^e}^T({\pmb{T}^e}^T\widehat{\pmb{F}^e}-\overline{\pmb{F}^e})=0$

For arbitary $\delta\overline{\pmb{d}^e}$, above equations holds, according to **the vector scalar product theorem**

$\overline{\pmb{F}^e}={\pmb{T}^e}^T\widehat{\pmb{F}^e}={\pmb{T}^e}^T\widehat{\pmb{K}^e}\widehat{\pmb{d}^e}={\pmb{T}^e}^T\widehat{\pmb{K}^e}\pmb{T}^e\overline{\pmb{d}^e}$

So $\overline{\pmb{K}^e}={\pmb{T}^e}^T\widehat{\pmb{K}^e}\pmb{T}^e$


<blockquote>Energy is invariant with respect to the frame of reference ----->  **Principle of virtual work** & **Theorem of minimum potential energy**</blockquote>