# Tensor calculus and constitutive relations

> Reference: [[重制]有限元基础课4-张量计算及neoHookean超弹性材料本构关系推导](https://www.bilibili.com/video/BV1zh411s7pR/?spm_id_from=333.999.list.card_archive.click&vd_source=f566256e2ad94fc1d442f367a2368001)

## Some useful derivations

### Tensor-$2^{\text{nd}}$ order

$\text{tr}(\boldsymbol{A})=\boldsymbol{I}:\boldsymbol{A}=A_{ii}$

$\text{tr}(\boldsymbol{A}\boldsymbol{B})=\text{tr}(\boldsymbol{B}\boldsymbol{A})$

$\boldsymbol{A}:\boldsymbol{B}=A_{ij}B_{ij}$

$|\boldsymbol{A}|=(\boldsymbol{A}:\boldsymbol{A})^{\frac{1}{2}}=(A_{ij}A_{ij})^{\frac{1}{2}}$

$\boldsymbol{A}:(\boldsymbol{B}\boldsymbol{C})=(\boldsymbol{B}^T\boldsymbol{A}):\boldsymbol{C}=(\boldsymbol{A}\boldsymbol{C}^T):\boldsymbol{B}$

$\boldsymbol{A} = \underbrace{\frac{1}{3}\text{tr}(\boldsymbol{A})\boldsymbol{I}}_{} + \underbrace{\boldsymbol{A} - \frac{1}{3}\text{tr}(\boldsymbol{A})\boldsymbol{I}}_{\text{dev}(\boldsymbol{A})}$

### Tensor-$4^{\text{nd}}$ order

$\mathbb{A}=A_{ijkl}\boldsymbol{e}_i\otimes\boldsymbol{e}_j\otimes\boldsymbol{e}_k\otimes\boldsymbol{e}_l$

$\mathbb{C}=\displaystyle\frac{\partial\boldsymbol{\sigma}}{\partial\boldsymbol{\varepsilon}}$

$\boldsymbol{A}=\mathbb{I}:\boldsymbol{A}$
$\quad\rightarrow\quad$
$\mathbb{I}=\displaystyle\frac{\partial\boldsymbol{A}}{\partial\boldsymbol{A}}=\delta_{ik}\delta_{jl}\boldsymbol{e}_i\otimes\boldsymbol{e}_j\otimes\boldsymbol{e}_k\otimes\boldsymbol{e}_l$

Similarly, we have

$\boldsymbol{A}^T=\overline{\mathbb{I}}:\boldsymbol{A}$
$\quad\rightarrow\quad$
$\overline{\mathbb{I}}=\displaystyle\frac{\partial\boldsymbol{A}^T}{\partial\boldsymbol{A}}=\delta_{jk}\delta_{il}\boldsymbol{e}_i\otimes\boldsymbol{e}_j\otimes\boldsymbol{e}_k\otimes\boldsymbol{e}_l$


$\boldsymbol{A}=\underline{\mathbb{I}}:\boldsymbol{A}^T$
$\quad\rightarrow\quad$
$\overline{\mathbb{I}}=\displaystyle\frac{\partial\boldsymbol{A}}{\partial\boldsymbol{A}^T}=\delta_{il}\delta_{jk}\boldsymbol{e}_i\otimes\boldsymbol{e}_j\otimes\boldsymbol{e}_k\otimes\boldsymbol{e}_l$

$\mathbb{I}^s=\displaystyle\frac{1}{2}(\mathbb{I}+\mathbb{I}^T)=\frac{1}{2}(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})\boldsymbol{e}_i\otimes\boldsymbol{e}_j\otimes\boldsymbol{e}_k\otimes\boldsymbol{e}_l$

and $\boldsymbol{A}^s=\mathbb{I}^s:\boldsymbol{A}^s$


$\text{dev}(\boldsymbol{A})=\boldsymbol{A}-\displaystyle\frac{1}{3}\text{tr}(\boldsymbol{A})\boldsymbol{I}=\boldsymbol{A}-\frac{1}{3}(\boldsymbol{I}:\boldsymbol{A})\boldsymbol{I}=\underbrace{(\mathbb{I}-\frac{1}{3}\boldsymbol{I}\otimes\boldsymbol{I})}_{\text{projection tensor}\,\mathbb{P}}:\boldsymbol{A}$

### Tensor-derivatives

#### For trace

$\displaystyle\frac{\partial\text{tr}(\boldsymbol{A})}{\partial\boldsymbol{A}}=\boldsymbol{I}=\delta_{ij}\boldsymbol{e}_i\otimes\boldsymbol{e}_j$

$\displaystyle\frac{\partial\text{tr}(\boldsymbol{A}^2)}{\partial\boldsymbol{A}}=2\boldsymbol{A}^T$

$\displaystyle\frac{\partial\text{tr}(\boldsymbol{A}^3)}{\partial\boldsymbol{A}}=\boldsymbol{I}=3(\boldsymbol{A}^T)^2$

#### For determinant
$\displaystyle\frac{\partial\text{det}(\boldsymbol{A})}{\partial\boldsymbol{A}}=\text{det}(\boldsymbol{A})\boldsymbol{A}^{-T}$

> Example: $J=\text{det}(\boldsymbol{F})$, $\boldsymbol{C}=\boldsymbol{F}^T\boldsymbol{F},\displaystyle\frac{\partial J}{\partial\boldsymbol{C}}=?$
> 
> Answer:
> 
> $\displaystyle\frac{\partial\text{det}(\boldsymbol{C})}{\partial\boldsymbol{C}}=\text{det}(\boldsymbol{C})\boldsymbol{C}^{-1}=J^2\boldsymbol{C}^{-1}$
>
> $\displaystyle\frac{\partial\text{det}(\boldsymbol{C})}{\partial\boldsymbol{C}}=\displaystyle\frac{\partial J^2}{\partial\boldsymbol{C}}=2J\frac{\partial J}{\partial \boldsymbol{C}}$
>
> So, 
> $\displaystyle\frac{\partial J}{\partial\boldsymbol{C}}=\frac{J}{2}\boldsymbol{C}^{-1}$
>

$\displaystyle\frac{\partial(\boldsymbol{A})^{-1}}{\partial\boldsymbol{A}}=-\frac{1}{2}(A_{ik}^{-1}A_{lj}^{-1}+A_{il}^{-1}A_{kj}^{-1})\boldsymbol{e}_i\otimes\boldsymbol{e}_j\otimes\boldsymbol{e}_k\otimes\boldsymbol{e}_l=-\boldsymbol{A}^{-1}\odot\boldsymbol{A}^{-1}$


## Linear elasticity

**Elastic free energy functional**

$\psi=\displaystyle\frac{1}{2}\boldsymbol{\sigma}:\boldsymbol{\varepsilon}=\frac{1}{2}(\mathbb{C}:\boldsymbol{\varepsilon}):\boldsymbol{\varepsilon}$

**Stress**

$\boldsymbol{\sigma}=\displaystyle\frac{\partial\psi}{\partial\boldsymbol{\varepsilon}}=\frac{1}{2}(\frac{\partial(\mathbb{C}:\boldsymbol{\varepsilon})}{\partial\boldsymbol{\varepsilon}}:\boldsymbol{\varepsilon}+(\mathbb{C}:\boldsymbol{\varepsilon}):\frac{\partial\boldsymbol{\varepsilon}}{\partial\boldsymbol{\varepsilon}})=\frac{1}{2}(2\mathbb{C}:\boldsymbol{\varepsilon})=\mathbb{C}:\boldsymbol{\varepsilon}$

**Jacobian/Constitutive tensor**

$\mathbb{J}=\mathbb{C}=\displaystyle\frac{\partial\boldsymbol{\sigma}}{\partial\boldsymbol{\varepsilon}}$

**Voigt notation**

...




## Hyperelasticity

**Strain**

Green-Lagrange strain tensor: $\boldsymbol{E}=\displaystyle\frac{1}{2}(\boldsymbol{F}^T\boldsymbol{F}-\boldsymbol{I})$

Right Cauchy-Green tensor: $\boldsymbol{C}=\boldsymbol{F}^T\boldsymbol{F}$

Left Cauchy-Green tensor: $\boldsymbol{C}=\boldsymbol{F}\boldsymbol{F}^T$

**Stress**

1st Piola-kirchhoff stress: $\boldsymbol{P}=\displaystyle\frac{\partial\psi}{\partial\boldsymbol{F}}=2\boldsymbol{F}\frac{\partial\psi}{\partial\boldsymbol{C}}$

2nd Piola-kirchhoff stress: $\boldsymbol{S}=\displaystyle\frac{\partial\psi}{\partial\boldsymbol{E}}=2\frac{\partial\psi}{\partial\boldsymbol{C}}$

$\boldsymbol{P}=\boldsymbol{F}\boldsymbol{S}$

> Prove $\displaystyle\frac{\partial}{\partial\boldsymbol{F}}=2\boldsymbol{F}\frac{\partial}{\partial\boldsymbol{C}}$
>
> $\displaystyle\frac{\partial}{\partial\boldsymbol{F}}=\frac{\partial}{\partial\boldsymbol{C}}\frac{\partial\boldsymbol{C}}{\partial\boldsymbol{F}}=\frac{\partial}{\partial C_{kl}}\frac{\partial F_{pk}F_{pl}}{\partial F_{ij}}=\frac{\partial}{\partial C_{kl}}(\delta_{pi}\delta_{kj}F_{pl}+F_{pk}\delta_{pi}\delta_{lj})=\frac{\partial }{\partial C_{jl}}F_{il}+\frac{\partial }{\partial C_{kj}}F_{ik}=2F_{il}\frac{\partial}{\partial C_{lj}}$
>
> $\displaystyle\frac{\partial\boldsymbol{C}}{\partial\boldsymbol{F}}=\boldsymbol{I}\underline{\otimes}\boldsymbol{F}^T+\boldsymbol{F}^T\overline{\otimes}\boldsymbol{I}$
>

**Jacobian**

$\mathbb{J}=\displaystyle\frac{\partial\boldsymbol{P}}{\partial\boldsymbol{F}}=\frac{\partial}{\partial\boldsymbol{F}}(\frac{\partial\psi}{\partial\boldsymbol{F}})=2\boldsymbol{F}\frac{\partial}{\partial\boldsymbol{C}}(2\boldsymbol{F}\frac{\partial\psi}{\partial\boldsymbol{C}})=4\boldsymbol{F}\frac{\partial^2\psi}{\partial\boldsymbol{C}^2}\boldsymbol{F}^T$

$\mathbb{J}\approx\displaystyle\frac{\partial\boldsymbol{S}}{\partial\boldsymbol{C}}=4\frac{\partial^2\psi}{\partial\boldsymbol{C}^2}$

### Saint Venant-Kirchhoff model

**Elastic free energy**

$\psi(\boldsymbol{E})=\displaystyle\frac{1}{2}\lambda(\text{tr}(\boldsymbol{E}))^2+\mu\text{tr}(\boldsymbol{E}^2)$

**Stress**

$\boldsymbol{S}=\displaystyle\frac{\partial\psi}{\partial\boldsymbol{E}}=\frac{1}{2}\lambda2\text{tr}(\boldsymbol{E})\frac{\partial\text{tr}(\boldsymbol{E})}{\partial\boldsymbol{E}}+\mu\frac{\partial\text{tr}(\boldsymbol{E}^2)}{\partial\boldsymbol{E}}=\lambda\text{tr}(\boldsymbol{E})+2\mu\boldsymbol{E}^T$

**Jacobian**

$\mathbb{J}=\displaystyle\frac{\partial\boldsymbol{S}}{\partial\boldsymbol{E}}=\lambda\boldsymbol{I}\otimes\frac{\partial\text{tr}(\boldsymbol{E})}{\partial\boldsymbol{E}}+2\mu\frac{\partial\boldsymbol{E}^T}{\partial\boldsymbol{E}}=\lambda\boldsymbol{I}\otimes\boldsymbol{I}+2\mu\mathbb{I}^s=\lambda\delta_{ij}\delta_{kl}+\mu(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})$


### neo-Hookean model

#### Form 1

**Elastic free energy**

$\psi(\boldsymbol{C})=\displaystyle\frac{1}{2}\lambda(\text{ln}J)^2-\mu\text{ln}J+\frac{1}{2}\mu(\text{tr}(\boldsymbol{C})-3)$

**Stress**

$\boldsymbol{S}=2\displaystyle\frac{\partial\psi}{\partial\boldsymbol{C}}=\lambda(2\text{ln}J\frac{1}{J})\frac{\partial J}{\partial\boldsymbol{C}}-2\mu\frac{1}{J}\frac{\partial J}{\partial\boldsymbol{C}}+\mu\frac{\partial\text{tr}(\boldsymbol{C})}{\partial\boldsymbol{C}}=\frac{2\lambda}{J}\text{ln}J\frac{J}{2}\boldsymbol{C}^{-1}-\mu\boldsymbol{C}^{-1}+\mu\boldsymbol{I}=\lambda\text{ln}J\boldsymbol{C}^{-1}-\mu\boldsymbol{C}^{-1}+\mu\boldsymbol{I}$

**Jacobian**

$\mathbb{J}=4\displaystyle\frac{\partial^2\psi}{\partial\boldsymbol{C}^2}=2\frac{\partial\boldsymbol{S}}{\partial\boldsymbol{C}}=2\lambda\boldsymbol{C}^{-1}\otimes\frac{\partial\text{ln}J}{\partial\boldsymbol{C}}+2\lambda\text{ln}J\frac{\partial\boldsymbol{C}^{-1}}{\partial\boldsymbol{C}}-2\mu\frac{\partial\boldsymbol{C}^{-1}}{\partial\boldsymbol{C}}=\lambda\boldsymbol{C}^{-1}\otimes\boldsymbol{C}^{-1}-2\lambda\text{ln}J\boldsymbol{C}^{-1}\odot\boldsymbol{C}^{-1}+2\mu\boldsymbol{C}^{-1}\odot\boldsymbol{C}^{-1}$