# Chapter 1 Preliminary Concepts

> Last updated: 09 May 2026

## 1.4 Mechanics of Continuous Bodies

Three fundamental laws of mechanics:

1. **Conservation of Mass** （can be easily satisfied for a Lagrangian description）
2. **Conservation of Angular Momentum** $\rightarrow$ Symmetry of Cauchy stress tensor $\sigma$.
3. **Conservation of Linear Momentum** $\rightarrow$ a differential equation of force equilibrium

### 1.4.1 Boundary-Valued Problem

The **balance of linear momentum** at each point in the interested domain $\Omega$ can be expressed as:

$$
\nabla \cdot \boldsymbol{\sigma} + \boldsymbol{f}^b = \boldsymbol{0}, \quad \boldsymbol{x}\in \Omega
$$

The **boundary-valued problem** is to find $\boldsymbol{u}$ such that

$$\begin{aligned}
\nabla \cdot \boldsymbol{\sigma}(\boldsymbol{u}) + \boldsymbol{f}^b = \boldsymbol{0}, & \quad \boldsymbol{x}\in \Omega \\
\boldsymbol{u} = \boldsymbol{0}, & \quad\boldsymbol{x}\in \Gamma^h \\
\boldsymbol{\sigma} \cdot \boldsymbol{n} = \boldsymbol{t}, & \quad\boldsymbol{x}\in \Gamma^s
\end{aligned}
$$

where $\Gamma^h$ is the essential boundary and $\Gamma^s$ is the natural boundary. $\Gamma = \Gamma^h \cup \Gamma^s$ and $\Gamma^h \cap \Gamma^s = \emptyset$. The above problem is also called the *strong form* of the BVP, because the differential equation is satisfied at every point in the domain.

> Generally, if the order of the differential equation is $2m$, the BCs that contains derivatives of order $m-1$ or lower are called *essential* BCs, and those that contains derivatives of a higher order than $m-1$ are called *natural* BCs.

### 1.4.2 Principle of Minimum Potential Energy

The **principle of minimum potential energy** states that: for all **kinematically admissible displacements**, those that satisfy the above BVP make the total potential energy:

$$\Pi(\boldsymbol{u}) = \underbrace{\frac{1}{2} \int_\Omega \boldsymbol{\sigma}(\boldsymbol{u}) : \boldsymbol{\varepsilon}(\boldsymbol{u}) \, d\Omega}_{\text{Strain Energy } U(\boldsymbol{u})} - \underbrace{\int_\Omega \boldsymbol{u} \cdot \boldsymbol{f}^b\, d\Omega - \int_{\Gamma^s} \boldsymbol{u} \cdot \boldsymbol{t} \, d\Gamma}_{\text{External Work } W(\boldsymbol{u})}
$$

stationary on the solution space:

$$
\mathbb{Z} = \{\boldsymbol{u} \in  [H^1(\Omega)]^3 | \boldsymbol{u} = \boldsymbol{0} \text{ on } \boldsymbol{x} \in \Gamma^h\}
$$

where $H^1(\Omega)$ is the Sobolev space of order 1.

The **virtual displacement** or **variation of** $\boldsymbol{u}$, denoted as $\overline{\boldsymbol{u}}$, is defined by considering an perturbation $\boldsymbol{\eta}(\boldsymbol{x})$ in the solution space $\mathbb{Z}$ and a small scalar $\tau$:

$$
\overline{\boldsymbol{u}} = \lim_{\tau \to 0} \frac{1}{\tau} [(\boldsymbol{u} + \tau \boldsymbol{\eta}) - (\boldsymbol{u})] = 
\frac{d}{d\tau} \left. (\boldsymbol{u} + \tau \boldsymbol{\eta}) \right|_{\tau = 0} =\boldsymbol{\eta}
$$

An important property of $\overline{\boldsymbol{u}}$ is that it is independent of differentiation w.r.t. spatial coordinates:

$$
\frac{\partial \overline{\boldsymbol{u}}}{\partial x_i} = \overline{\left( \frac{\partial \boldsymbol{u}}{\partial x_i} \right)}
$$

The **principle of minimum potential energy** states that the true displacement field $\boldsymbol{u}$ uniquely minimizes the potential energy functional $\Pi(\boldsymbol{u})$. To find this minimum, we seek a stationary condition.

For any kinematically admissible virtual displacement $\overline{\boldsymbol{u}}$, we consider a perturbed configuration $\boldsymbol{u} + \tau \overline{\boldsymbol{u}}$. If $\boldsymbol{u}$ is the true minimizer, then the real-valued function $g(\tau) = \Pi(\boldsymbol{u} + \tau \overline{\boldsymbol{u}})$ must achieve its minimum at $\tau = 0$.

Consequently, the **first variation** of $\Pi$ at $\boldsymbol{u}$ in the direction of $\overline{\boldsymbol{u}}$, defined as the derivative of this function:

$$
\delta \Pi(\boldsymbol{u}; \overline{\boldsymbol{u}}) = \frac{d}{d\tau} \Pi(\boldsymbol{u} + \tau \overline{\boldsymbol{u}}) \big|_{\tau=0}
$$

must vanish for all arbitrary $\overline{\boldsymbol{u}}$. This leads to the **variational equation**:

$$
\delta \Pi(\boldsymbol{u}; \overline{\boldsymbol{u}}) = 0, \quad \forall \overline{\boldsymbol{u}} \in \mathbb{Z}
$$

This equation serves as the necessary condition for a minimum of $\Pi$ at $\boldsymbol{u}$. The variational equation can be further written as:

$$
\begin{aligned}
\delta\Pi(\boldsymbol{u}; \overline{\boldsymbol{u}}) &= \delta U(\boldsymbol{u}; \overline{\boldsymbol{u}}) - \delta W(\boldsymbol{u}; \overline{\boldsymbol{u}})\\
&= \underbrace{\int_\Omega \boldsymbol{\varepsilon}(\overline{\boldsymbol{u}}):\mathbb{D}:\boldsymbol{\varepsilon}(\boldsymbol{u}) \, d\Omega}_{a(\boldsymbol{u}, \overline{\boldsymbol{u}})} - \underbrace{(\int_\Omega \overline{\boldsymbol{u}} \cdot \boldsymbol{f}^b\, d\Omega + \int_{\Gamma^s} \overline{\boldsymbol{u}} \cdot \boldsymbol{t} \, d\Gamma)}_{l(\overline{\boldsymbol{u}})}\\
&= a(\boldsymbol{u}, \overline{\boldsymbol{u}}) - l(\overline{\boldsymbol{u}})=0
\end{aligned}
$$

where $a(\boldsymbol{u}, \overline{\boldsymbol{u}})$ is called **the energy bilinear form** and $l(\overline{\boldsymbol{u}})$ is called **the load linear form** (only **conservative loads** are considered here). The variational equation can be rewritten as:

$$
a(\boldsymbol{u}, \overline{\boldsymbol{u}}) = l(\overline{\boldsymbol{u}}), \quad \forall \overline{\boldsymbol{u}} \in \mathbb{Z}
$$

> For some specific conditions, the above variational equation has a unique solution. (See revelant contents and other useful discussion on P45).

### 1.4.3 Principle of Virtual Work

The **principle of minimum potential energy** only works for conservative systems, usually the elastic problmes. A more general principle is the **principle of virtual work**, which states that: for all kinematically admissible virtual displacements, the internal virtual work equals to the external virtual work. The equation remains the same as the variational equation but it is **unnecessary to require the point-wise satisfaction of the differential equation**.
