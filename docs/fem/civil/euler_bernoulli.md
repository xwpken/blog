# Euler-Bernoulli Beam Element

> Last updated: 01 July 2026

## 1. Hypothesis
The Euler-Bernoulli beam theory states that **cross-sections remain plane and perpendicular to the neutral axis after deformation**. This implies that transverse shear deformation is zero. The theory directly describes axial deformation and bending; torsion is not part of the original kinematic hypothesis but can be superimposed independently via Saint-Venant's torsion theory. The following derivations assume **linear elasticity** and **small deformations**. Extension to large deformations will be discussed later.

## 2. Kinematics

This section establishes the relationship between the nodal degrees of freedom and the strain field within the element. The fundamental strategy is to treat axial, bending, and torsion as **decoupled deformation modes**, apply their respective kinematic assumptions, and then superimpose the results using the principle of linear superposition.

### 2.1 Nodal Degrees of Freedom

The element possesses 12 degrees of freedom (DOFs), with 6 DOFs assigned to each of the two nodes:

$$
\mathbf{u}^{(e)} = [u_1, v_1, w_1, \theta_{x1}, \theta_{y1}, \theta_{z1}, \ u_2, v_2, w_2, \theta_{x2}, \theta_{y2}, \theta_{z2}]^{\text{T}} \in \mathbb{R}^{12}
$$

The physical interpretations are:
- $u$: Axial displacement along the local $x$-axis.
- $v, w$: Transverse displacements along the local $y$- and $z$-axes.
- $\theta_x$: Rotational angle about the local $x$-axis (torsion).
- $\theta_y, \theta_z$: Rotational angles about the local $y$- and $z$-axes (bending slopes).

### 2.2 Kinematic Assumptions

To construct the displacement field, we define the governing assumptions for each deformation mode independently.

#### Axial Deformation
The axial displacement $u(x)$ is uniform over the cross-section and varies only along the beam axis.

#### Bending (xy-plane and xz-plane)
The Euler-Bernoulli beam theory states that plane cross-sections remain plane and **perpendicular to the neutral axis** after bending deformation. This implies that transverse shear deformation due to bending is zero.

For the $xy$-plane (bending about $z$), the slope of the neutral axis equals the cross-sectional rotation:

$$
\theta_z(x) = \frac{dv(x)}{dx}
$$

For the $xz$-plane (bending about $y$), following the right-hand rule, the rotation relates to the slope by:

$$
\theta_y(x) = -\frac{dw(x)}{dx}
$$

> The negative sign in $\theta_y$ is a consequence of the right-hand rule: a positive rotation about the $y$-axis corresponds to a negative slope of the transverse displacement $w$.

#### Torsion
The torsion deformation is superimposed independently. Adopting the **Saint-Venant torsion theory**, we assume:
- The cross-section rotates as a rigid body about the $x$-axis by an angle $\theta_x(x)$.
- The warping (out-of-plane axial displacement) is uniform along the beam axis, meaning it does not produce any axial normal strain $\varepsilon_{xx}$.
- The only strain components introduced by torsion are the shear strains $\gamma_{xy}$ and $\gamma_{xz}$ in the cross-section.

### 2.3 Displacement Field

The total displacement of a point $(x, y, z)$ is the superposition of contributions from each deformation mode. Cross-sectional rotations about the three axes produce additional displacements for points off the neutral axis ($y \neq 0$ or $z \neq 0$).

$$
\boxed{
\begin{aligned}
U_x(x,y,z) &= u(x) - y\,\theta_z(x) + z\,\theta_y(x) \\
U_y(x,y,z) &= v(x) - z\,\theta_x(x) \\
U_z(x,y,z) &= w(x) + y\,\theta_x(x)
\end{aligned}
}
$$

| Term | Origin |
| :--- | :--- |
| $u$ | Axial translation of the neutral axis |
| $v$ | Transverse translation in $y$ |
| $w$ | Transverse translation in $z$ |
| $-y\theta_z$ | Rotation $\theta_z$ about $z$ (xy-bending) moves $y$-offset points in $x$ |
| $z\theta_y$ | Rotation $\theta_y$ about $y$ (xz-bending) moves $z$-offset points in $x$ |
| $-z\theta_x$ | Torsion $\theta_x$ moves points tangentially in $y$ |
| $y\theta_x$ | Torsion $\theta_x$ moves points tangentially in $z$ |

The signs follow from the right-hand rule.

### 2.4 Strain

With the displacement field established, we now derive the strain components. The infinitesimal strain tensor is defined as $\varepsilon_{ij} = \frac{1}{2}(\frac{\partial U_i}{\partial x_j} + \frac{\partial U_j}{\partial x_i})$.

#### Normal Strain
The axial normal strain is obtained by differentiating $U_x$ with respect to $x$:

$$
\varepsilon_{xx} = \frac{\partial U_x}{\partial x}
= \frac{du}{dx} - y\frac{d\theta_z}{dx} + z\frac{d\theta_y}{dx}
$$

Substituting the Euler-Bernoulli slope relations $\theta_z = v'$ and $\theta_y = -w'$:

$$
\boxed{\varepsilon_{xx} = \frac{du}{dx} - y\frac{d^2v}{dx^2} - z\frac{d^2w}{dx^2}}
$$

#### Shear Strains (Critical Distinction)
We explicitly compute the full shear strain expressions before applying the bending assumptions. 

For the $xy$-plane:

$$
\gamma_{xy} = \frac{\partial U_x}{\partial y} + \frac{\partial U_y}{\partial x}
= \left(-\theta_z + \frac{dv}{dx}\right) - z\frac{d\theta_x}{dx}
$$

Notice the bracket $\left(-\theta_z + \frac{dv}{dx}\right)$. According to the Euler-Bernoulli hypothesis in Section 2.2, $\theta_z = dv/dx$. Therefore, the **transverse shear deformation due to bending** is identically zero:

$$
-\theta_z + \frac{dv}{dx} = 0
$$

Thus, the shear strain in the $xy$-plane reduces to:

$$
\boxed{\gamma_{xy} = -z\frac{d\theta_x}{dx}}
$$

Similarly, for the $xz$-plane:

$$
\gamma_{xz} = \frac{\partial U_x}{\partial z} + \frac{\partial U_z}{\partial x}
= \left(\theta_y + \frac{dw}{dx}\right) + y\frac{d\theta_x}{dx}
$$

Again, the bracket $\left(\theta_y + \frac{dw}{dx}\right)$ represents the transverse shear deformation due to bending in the $xz$-plane. Since $\theta_y = -dw/dx$, it vanishes:

$$
\theta_y + \frac{dw}{dx} = 0
$$

Leaving only the torsional contribution:

$$
\boxed{\gamma_{xz} = y\frac{d\theta_x}{dx}}
$$

#### Summary of Strains
The final strain field utilized for the element formulation is:

| Component | Expression | Source |
| :--- | :--- | :--- |
| $\varepsilon_{xx}$ | $u' - y v'' - z w''$ | Axial + Bending |
| $\gamma_{xy}$ | $-z\,\theta_x'$ | Torsion only |
| $\gamma_{xz}$ | $y\,\theta_x'$ | Torsion only |

> Remarks: The transverse shear strains generated by bending forces are completely omitted in this theory. The only shear strains present in the cross-section are entirely due to the Saint-Venant torsion. This confirms that the axial/bending and torsion modes are fully decoupled at the strain energy level.



## 4. Strain Energy and Weak Form

### 4.1 Strain Energy Functional

Total strain energy $U = \frac12 \int_V \boldsymbol{\sigma}^T \boldsymbol{\varepsilon}\,dV$. With $\sigma_{xx} = E\varepsilon_{xx}$ and $\tau = G\gamma$:

$$
U = \frac12 \int_V \left[E\varepsilon_{xx}^2 + G(\gamma_{xy}^2 + \gamma_{xz}^2)\right] dV
$$

Integrating term by term over the cross-section:

**Axial** ($\int_A dA = A$):
$$
\frac12\int_0^L EA\,(u')^2 dx
$$

**xy-bending** ($\int_A y^2 dA = I_z$):
$$
\frac12\int_0^L EI_z\,(v'')^2 dx
$$

**xz-bending** ($\int_A z^2 dA = I_y$):
$$
\frac12\int_0^L EI_y\,(w'')^2 dx
$$

**Torsion** ($\int_A r^2 dA = J$, the torsion constant):
$$
\frac12\int_0^L GJ\,(\theta_x')^2 dx
$$

> **Important**: For circular cross-sections, $J = I_p = I_y + I_z$. For non-circular sections, $J < I_p$ (determined by the Saint-Venant torsion solution). The stiffness matrix uses $J$, while the mass matrix uses $I_p$ — this distinction is critical and is detailed in Section 7.3.

The four modes are fully decoupled:

$$
\boxed{U = \frac12\int_0^L\Big[EA(u')^2 + EI_z(v'')^2 + EI_y(w'')^2 + GJ(\theta_x')^2\Big] dx}
$$

### 4.2 External Virtual Work

$$
\delta W = \int_0^L (p\,\delta u + q_y\,\delta v + q_z\,\delta w + m_x\,\delta\theta_x) dx + \text{boundary nodal force terms}
$$

### 4.3 Weak Form

Applying the principle of virtual work $\delta U - \delta W = 0$ and integrating by parts yields the symmetric weak form:

$$
\boxed{
\int_0^L EA\,u'\delta u'\,dx + \int_0^L EI_z\,v''\delta v''\,dx + \int_0^L EI_y\,w''\delta w''\,dx + \int_0^L GJ\,\theta_x'\delta\theta_x'\,dx = \delta W
}
$$

---

## 5. Shape Functions

Introduce the natural coordinate $\xi = x/L \in [0, 1]$, so $dx = L d\xi$ and $d/dx = (1/L)\,d/d\xi$.

### 5.1 Axial and Torsion — $C^0$ Linear Lagrange Interpolation

Both $u(x)$ and $\theta_x(x)$ require only $C^0$ continuity (first derivative not needed in the weak form):

$$
\begin{aligned}
u(\xi) &= (1-\xi) u_1 + \xi u_2 \equiv N_1(\xi) u_1 + N_2(\xi) u_2 \\
\theta_x(\xi) &= (1-\xi) \theta_{x1} + \xi \theta_{x2}
\end{aligned}
$$

$$
\boxed{N_1(\xi) = 1 - \xi, \qquad N_2(\xi) = \xi}
$$

Properties: $N_1(0)=1, N_1(1)=0$; $N_2(0)=0, N_2(1)=1$.

### 5.2 Bending — $C^1$ Hermite Cubic Interpolation

The bending weak form contains $v''$, requiring $C^1$ continuity. A cubic polynomial $v(\xi) = a_0 + a_1\xi + a_2\xi^2 + a_3\xi^3$ is determined by four boundary conditions:

$$
\begin{aligned}
v(0) &= v_1, \quad v'(0)/L = \theta_{z1} \\
v(1) &= v_2, \quad v'(1)/L = \theta_{z2}
\end{aligned}
$$

Solving for $a_0, a_1, a_2, a_3$ and grouping by DOFs yields:

$$
v(\xi) = H_1(\xi) v_1 + H_2(\xi) \theta_{z1} + H_3(\xi) v_2 + H_4(\xi) \theta_{z2}
$$

$$
\boxed{
\begin{aligned}
H_1(\xi) &= 2\xi^3 - 3\xi^2 + 1 \\
H_2(\xi) &= L(\xi^3 - 2\xi^2 + \xi) \\
H_3(\xi) &= -2\xi^3 + 3\xi^2 \\
H_4(\xi) &= L(\xi^3 - \xi^2)
\end{aligned}}
$$

**Verification** (only $v_1=1$, all other DOFs zero at $\xi=0$):
- $H_1(0) = 1$, $H_1'(0) = 0$ $\Rightarrow$ $v=1, \theta_z=0$ at node 1 $\checkmark$
- $H_2(0) = 0$, $H_2'(0)/L = 1$ $\Rightarrow$ $v=0, \theta_z=1$ at node 1 $\checkmark$

For xz-plane bending, since $\theta_y = -dw/dx$, the shape functions for $\theta_y$ are $-H_2$ and $-H_4$.

---

## 6. Element Stiffness Matrix

### 6.1 Axial Stiffness

Strain-displacement matrix $\mathbf{B}_a$:

$$
\varepsilon_{xx}^a = \frac{du}{dx} = \frac{1}{L}\frac{d}{d\xi}[N_1, N_2]\begin{Bmatrix}u_1\\u_2\end{Bmatrix}
= \frac{1}{L}[-1, 1]\begin{Bmatrix}u_1\\u_2\end{Bmatrix}
$$

$$
\boxed{\mathbf{B}_a = \frac{1}{L}[-1, \; 1]}
$$

Element stiffness matrix:

$$
\mathbf{k}_a = \int_0^L \mathbf{B}_a^T EA\,\mathbf{B}_a\,dx
= EA\int_0^1 \frac{1}{L^2}\begin{bmatrix}-1\\1\end{bmatrix}[-1, 1]\,L d\xi
= \frac{EA}{L}\int_0^1\begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix}d\xi
$$

$$
\boxed{\mathbf{k}_a = \frac{EA}{L}\begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix}}
$$

### 6.2 Torsional Stiffness

Identical form, replacing $EA$ with $GJ$:

$$
\boxed{\mathbf{k}_t = \frac{GJ}{L}\begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix}}
$$

### 6.3 xy-Plane Bending Stiffness (about z-axis)

Curvature-displacement matrix $\mathbf{B}_{bz}$:

$$
\kappa_z = \frac{d^2v}{dx^2} = \frac{1}{L^2}\frac{d^2}{d\xi^2}[H_1, H_2, H_3, H_4]\begin{Bmatrix}v_1\\\theta_{z1}\\v_2\\\theta_{z2}\end{Bmatrix}
$$

Second derivatives of $H_i$ with respect to $\xi$:

$$
\begin{aligned}
H_1''(\xi) &= 12\xi - 6 \\
H_2''(\xi) &= L(6\xi - 4) \\
H_3''(\xi) &= -12\xi + 6 \\
H_4''(\xi) &= L(6\xi - 2)
\end{aligned}
$$

Therefore:

$$
\boxed{\mathbf{B}_{bz} = \left[\frac{12\xi-6}{L^2},\; \frac{6\xi-4}{L},\; \frac{-12\xi+6}{L^2},\; \frac{6\xi-2}{L}\right]}
$$

The stiffness matrix is obtained by integration:

$$
\mathbf{k}_{bz} = \int_0^L \mathbf{B}_{bz}^T EI_z\,\mathbf{B}_{bz}\,dx
= EI_z L \int_0^1 \mathbf{B}_{bz}^T \mathbf{B}_{bz}\,d\xi
$$

Example entry $k_{11}$:

$$
k_{11} = EI_z L \int_0^1 \frac{(12\xi-6)^2}{L^4}d\xi
= \frac{EI_z}{L^3}\int_0^1(144\xi^2 - 144\xi + 36)d\xi
= \frac{EI_z}{L^3}\left[\frac{144}{3} - \frac{144}{2} + 36\right] = \frac{12EI_z}{L^3}
$$

All entries:

$$
\boxed{\mathbf{k}_{bz} = \begin{bmatrix}
\dfrac{12EI_z}{L^3} & \dfrac{6EI_z}{L^2} & -\dfrac{12EI_z}{L^3} & \dfrac{6EI_z}{L^2} \\[10pt]
\dfrac{6EI_z}{L^2} & \dfrac{4EI_z}{L} & -\dfrac{6EI_z}{L^2} & \dfrac{2EI_z}{L} \\[10pt]
-\dfrac{12EI_z}{L^3} & -\dfrac{6EI_z}{L^2} & \dfrac{12EI_z}{L^3} & -\dfrac{6EI_z}{L^2} \\[10pt]
\dfrac{6EI_z}{L^2} & \dfrac{2EI_z}{L} & -\dfrac{6EI_z}{L^2} & \dfrac{4EI_z}{L}
\end{bmatrix}}
$$

### 6.4 xz-Plane Bending Stiffness (about y-axis)

Since $\theta_y = -dw/dx$, the shape functions for $\theta_y$ are $-H_2$ and $-H_4$:

$$
w(\xi) = H_1(\xi) w_1 + [-H_2(\xi)]\theta_{y1} + H_3(\xi) w_2 + [-H_4(\xi)]\theta_{y2}
$$

This sign change propagates to the stiffness matrix:

$$
\boxed{\mathbf{k}_{by} = \begin{bmatrix}
\dfrac{12EI_y}{L^3} & -\dfrac{6EI_y}{L^2} & -\dfrac{12EI_y}{L^3} & -\dfrac{6EI_y}{L^2} \\[10pt]
-\dfrac{6EI_y}{L^2} & \dfrac{4EI_y}{L} & \dfrac{6EI_y}{L^2} & \dfrac{2EI_y}{L} \\[10pt]
-\dfrac{12EI_y}{L^3} & \dfrac{6EI_y}{L^2} & \dfrac{12EI_y}{L^3} & \dfrac{6EI_y}{L^2} \\[10pt]
-\dfrac{6EI_y}{L^2} & \dfrac{2EI_y}{L} & \dfrac{6EI_y}{L^2} & \dfrac{4EI_y}{L}
\end{bmatrix}}
$$

Note the sign difference from $\mathbf{k}_{bz}$: $v$-$\theta_z$ coupling is positive ($+6EI_z/L^2$), $w$-$\theta_y$ coupling is negative ($-6EI_y/L^2$).

### 6.5 Complete 12×12 Stiffness Matrix

With DOF ordering $[u_1, v_1, w_1, \theta_{x1}, \theta_{y1}, \theta_{z1}, u_2, v_2, w_2, \theta_{x2}, \theta_{y2}, \theta_{z2}]$:

$$
\mathbf{k}_{loc} = \begin{bmatrix}
\mathbf{K}_{11} & \mathbf{K}_{12} \\[4pt]
\mathbf{K}_{12}^T & \mathbf{K}_{22}
\end{bmatrix}_{12\times12}
$$

**$\mathbf{K}_{11}$ (Node 1 × Node 1):**

$$
\mathbf{K}_{11} = \begin{bmatrix}
\frac{EA}{L} & 0 & 0 & 0 & 0 & 0 \\[6pt]
0 & \frac{12EI_z}{L^3} & 0 & 0 & 0 & \frac{6EI_z}{L^2} \\[6pt]
0 & 0 & \frac{12EI_y}{L^3} & 0 & -\frac{6EI_y}{L^2} & 0 \\[6pt]
0 & 0 & 0 & \frac{GJ}{L} & 0 & 0 \\[6pt]
0 & 0 & -\frac{6EI_y}{L^2} & 0 & \frac{4EI_y}{L} & 0 \\[6pt]
0 & \frac{6EI_z}{L^2} & 0 & 0 & 0 & \frac{4EI_z}{L}
\end{bmatrix}
$$

**$\mathbf{K}_{12}$ (Node 1 × Node 2):**

$$
\mathbf{K}_{12} = \begin{bmatrix}
-\frac{EA}{L} & 0 & 0 & 0 & 0 & 0 \\[6pt]
0 & -\frac{12EI_z}{L^3} & 0 & 0 & 0 & \frac{6EI_z}{L^2} \\[6pt]
0 & 0 & -\frac{12EI_y}{L^3} & 0 & -\frac{6EI_y}{L^2} & 0 \\[6pt]
0 & 0 & 0 & -\frac{GJ}{L} & 0 & 0 \\[6pt]
0 & 0 & \frac{6EI_y}{L^2} & 0 & \frac{2EI_y}{L} & 0 \\[6pt]
0 & -\frac{6EI_z}{L^2} & 0 & 0 & 0 & \frac{2EI_z}{L}
\end{bmatrix}
$$

**$\mathbf{K}_{22}$ (Node 2 × Node 2):**

$$
\mathbf{K}_{22} = \begin{bmatrix}
\frac{EA}{L} & 0 & 0 & 0 & 0 & 0 \\[6pt]
0 & \frac{12EI_z}{L^3} & 0 & 0 & 0 & -\frac{6EI_z}{L^2} \\[6pt]
0 & 0 & \frac{12EI_y}{L^3} & 0 & \frac{6EI_y}{L^2} & 0 \\[6pt]
0 & 0 & 0 & \frac{GJ}{L} & 0 & 0 \\[6pt]
0 & 0 & \frac{6EI_y}{L^2} & 0 & \frac{4EI_y}{L} & 0 \\[6pt]
0 & -\frac{6EI_z}{L^2} & 0 & 0 & 0 & \frac{4EI_z}{L}
\end{bmatrix}
$$

#### Rigid-Body Verification

For $y$-direction rigid translation $[0,1,0,0,0,0,0,1,0,0,0,0]^T$:

- $v$-force at node 1: $\frac{12EI_z}{L^3}\cdot1 + (-\frac{12EI_z}{L^3})\cdot1 = 0$ $\checkmark$
- Moment about $z$ at node 1: $\frac{6EI_z}{L^2}\cdot1 + (-\frac{6EI_z}{L^2})\cdot1 = 0$ $\checkmark$

All rigid-body modes produce zero internal forces. The matrix is symmetric ($\mathbf{K}_{11}^T = \mathbf{K}_{11}$, $\mathbf{K}_{22}^T = \mathbf{K}_{22}$).

---

## 7. Element Consistent Mass Matrix

### 7.1 Kinetic Energy Formulation

The consistent mass matrix is derived from the kinetic energy using the **same** shape functions as the stiffness matrix. The kinetic energy is:

$$
T = \frac12\int_V \rho(\dot{U}_x^2 + \dot{U}_y^2 + \dot{U}_z^2)\,dV
$$

With $\mathbf{U} = \mathbf{N}\dot{\mathbf{u}}^{(e)}$:

$$
T = \frac12 \dot{\mathbf{u}}^{(e)T} \underbrace{\left(\int_V \rho \mathbf{N}^T\mathbf{N}\,dV\right)}_{\mathbf{m}^{(e)}} \dot{\mathbf{u}}^{(e)}
$$

### 7.2 Axial Mass

$u(\xi) = N_1 u_1 + N_2 u_2$, kinetic energy:

$$
T_a = \frac12\int_0^L \rho A\,\dot{u}^2 dx = \frac12\rho AL\int_0^1 \bigl[(1-\xi)\dot{u}_1 + \xi\dot{u}_2\bigr]^2 d\xi
$$

Mass matrix:

$$
\mathbf{m}_a = \rho AL\int_0^1\begin{bmatrix}(1-\xi)^2 & (1-\xi)\xi \\ (1-\xi)\xi & \xi^2\end{bmatrix}d\xi
$$

Using $\int_0^1 (1-\xi)^2 d\xi = \frac13$, $\int_0^1 \xi(1-\xi)d\xi = \frac16$, $\int_0^1 \xi^2 d\xi = \frac13$:

$$
\boxed{\mathbf{m}_a = \rho AL\begin{bmatrix}\frac13 & \frac16 \\[4pt] \frac16 & \frac13\end{bmatrix}}
$$

### 7.3 Torsional Mass — Fundamental Derivation

The kinetic energy of torsion comes from the rotational motion of the cross-section about the $x$-axis. The velocity field is $\dot{U}_y = -z\,\dot{\theta}_x$, $\dot{U}_z = y\,\dot{\theta}_x$. The squared velocity magnitude is:

$$
\dot{U}_y^2 + \dot{U}_z^2 = (y^2 + z^2)\,\dot{\theta}_x^2 = r^2\,\dot{\theta}_x^2
$$

Integrating over the cross-section:

$$
T_t = \frac12\int_0^L \rho\left[\int_A (y^2 + z^2)dA\right] \dot{\theta}_x^2 dx
= \frac12\int_0^L \rho I_p\,\dot{\theta}_x^2 dx
$$

$$
\boxed{I_p = \int_A (y^2+z^2)dA = I_y + I_z}
$$

> **Critical distinction**:
> - For **stiffness** ($GJ$): $J$ is the **torsion constant**, determined by the Saint-Venant torsion problem (stress-based). For non-circular sections, $J < I_p$.
> - For **mass** ($\rho I_p$): $I_p = I_y + I_z$ is the **polar moment of inertia** (geometry-only, independent of stress). It is always simply $\int(y^2+z^2)dA$.
> - For circular sections: $J = I_p$ (they coincide).
> - For non-circular sections: $J \neq I_p$. Example: a rectangular section $b \times h$ has $I_p = b^3h/12 + bh^3/12$, while $J \approx bh^3\left[\frac13 - 0.21\frac{h}{b}(1 - \frac{h^4}{12b^4})\right]$ — always less than $I_p$.

Using the linear shape functions $N_1, N_2$ for $\theta_x(\xi) = (1-\xi)\theta_{x1} + \xi\theta_{x2}$:

$$
T_t = \frac12\rho I_p L \int_0^1 \bigl[(1-\xi)\dot{\theta}_{x1} + \xi\dot{\theta}_{x2}\bigr]^2 d\xi
$$

Therefore:

$$
\boxed{\mathbf{m}_t = \rho I_p L \begin{bmatrix}\frac13 & \frac16 \\[4pt] \frac16 & \frac13\end{bmatrix}}
$$

**The torsional off-diagonal is $+\rho I_p L/6$**, identical in sign to the axial off-diagonal since both use the same linear shape functions.

### 7.4 xy-Plane Bending Mass

$v(\xi) = H_1 v_1 + H_2 \theta_{z1} + H_3 v_2 + H_4 \theta_{z2}$:

$$
T_{bz} = \frac12\int_0^L \rho A\,\dot{v}^2 dx = \frac12\rho AL\int_0^1 \dot{v}(\xi)^2 d\xi
$$

Mass matrix entries $m_{ij} = \rho AL\int_0^1 H_i(\xi)H_j(\xi)\,d\xi$. The Hermite integrals are:

$$
\begin{aligned}
\int_0^1 H_1^2 d\xi &= \frac{13}{35}, &
\int_0^1 H_2^2 d\xi &= \frac{L^2}{105}, &
\int_0^1 H_3^2 d\xi &= \frac{13}{35}, &
\int_0^1 H_4^2 d\xi &= \frac{L^2}{105} \\[4pt]
\int_0^1 H_1 H_2 d\xi &= \frac{11L}{210}, &
\int_0^1 H_1 H_3 d\xi &= \frac{9}{70}, &
\int_0^1 H_1 H_4 d\xi &= -\frac{13L}{420} \\[4pt]
\int_0^1 H_2 H_3 d\xi &= \frac{13L}{420}, &
\int_0^1 H_2 H_4 d\xi &= -\frac{L^2}{140}, &
\int_0^1 H_3 H_4 d\xi &= -\frac{11L}{210}
\end{aligned}
$$

Assembled matrix:

$$
\boxed{\mathbf{m}_{bz} = \rho AL\begin{bmatrix}
\frac{13}{35} & \frac{11L}{210} & \frac{9}{70} & -\frac{13L}{420} \\[6pt]
\frac{11L}{210} & \frac{L^2}{105} & \frac{13L}{420} & -\frac{L^2}{140} \\[6pt]
\frac{9}{70} & \frac{13L}{420} & \frac{13}{35} & -\frac{11L}{210} \\[6pt]
-\frac{13L}{420} & -\frac{L^2}{140} & -\frac{11L}{210} & \frac{L^2}{105}
\end{bmatrix}}
$$

### 7.5 xz-Plane Bending Mass

Since $\theta_y = -dw/dx$, the shape functions for $\theta_y$ are $-H_2$ and $-H_4$, flipping the signs of all $w$-$\theta_y$ coupling terms relative to the xy-plane:

$$
\boxed{\mathbf{m}_{by} = \rho AL\begin{bmatrix}
\frac{13}{35} & -\frac{11L}{210} & \frac{9}{70} & \frac{13L}{420} \\[6pt]
-\frac{11L}{210} & \frac{L^2}{105} & -\frac{13L}{420} & -\frac{L^2}{140} \\[6pt]
\frac{9}{70} & -\frac{13L}{420} & \frac{13}{35} & \frac{11L}{210} \\[6pt]
\frac{13L}{420} & -\frac{L^2}{140} & \frac{11L}{210} & \frac{L^2}{105}
\end{bmatrix}}
$$

### 7.6 Complete 12×12 Mass Matrix

Assembled in block form with $m_{total} = \rho A L$ and $r_x^2 = I_p/A = (I_y + I_z)/A$:

$$
\mathbf{m}_{loc} = \begin{bmatrix}
\mathbf{M}_a & \mathbf{M}_b \\[4pt]
\mathbf{M}_b^T & \mathbf{M}_c
\end{bmatrix}_{12\times12} \times (\rho A L)
$$

**$\mathbf{M}_a$ (Node 1 × Node 1):**

$$
\mathbf{M}_a = \begin{bmatrix}
\frac13 & 0 & 0 & 0 & 0 & 0 \\[6pt]
0 & \frac{13}{35} & 0 & 0 & 0 & \frac{11L}{210} \\[6pt]
0 & 0 & \frac{13}{35} & 0 & -\frac{11L}{210} & 0 \\[6pt]
0 & 0 & 0 & \frac{r_x^2}{3} & 0 & 0 \\[6pt]
0 & 0 & -\frac{11L}{210} & 0 & \frac{L^2}{105} & 0 \\[6pt]
0 & \frac{11L}{210} & 0 & 0 & 0 & \frac{L^2}{105}
\end{bmatrix}
$$

**$\mathbf{M}_b$ (Node 1 × Node 2 coupling):**

$$
\mathbf{M}_b = \begin{bmatrix}
\frac16 & 0 & 0 & 0 & 0 & 0 \\[6pt]
0 & \frac{9}{70} & 0 & 0 & 0 & -\frac{13L}{420} \\[6pt]
0 & 0 & \frac{9}{70} & 0 & \frac{13L}{420} & 0 \\[6pt]
0 & 0 & 0 & \frac{r_x^2}{6} & 0 & 0 \\[6pt]
0 & 0 & -\frac{13L}{420} & 0 & -\frac{L^2}{140} & 0 \\[6pt]
0 & \frac{13L}{420} & 0 & 0 & 0 & -\frac{L^2}{140}
\end{bmatrix}
$$

**$\mathbf{M}_c$ (Node 2 × Node 2):**

$$
\mathbf{M}_c = \begin{bmatrix}
\frac13 & 0 & 0 & 0 & 0 & 0 \\[6pt]
0 & \frac{13}{35} & 0 & 0 & 0 & -\frac{11L}{210} \\[6pt]
0 & 0 & \frac{13}{35} & 0 & \frac{11L}{210} & 0 \\[6pt]
0 & 0 & 0 & \frac{r_x^2}{3} & 0 & 0 \\[6pt]
0 & 0 & \frac{11L}{210} & 0 & \frac{L^2}{105} & 0 \\[6pt]
0 & -\frac{11L}{210} & 0 & 0 & 0 & \frac{L^2}{105}
\end{bmatrix}
$$

> **$\mathbf{M}_b[3,3] = +r_x^2/6$**, not $-r_x^2/6$. Both axial and torsional off-diagonal coupling masses are positive, since both use the same linear Lagrange shape functions with identical sign conventions.

#### Rigid-Body Verification

**Torsion** ($\dot{\theta}_{x1} = \dot{\theta}_{x2} = \omega$):

$$
T_t = \frac12\begin{bmatrix}\omega & \omega\end{bmatrix}
\rho I_p L\begin{bmatrix}\frac13 & \frac16 \\ \frac16 & \frac13\end{bmatrix}
\begin{bmatrix}\omega \\ \omega\end{bmatrix}
= \frac12\,\rho I_p L\,\omega^2
$$

Continuum: $T = \frac12\int_0^L \rho I_p \omega^2 dx = \frac12 \rho I_p L \omega^2$ $\checkmark$

**Axial** ($\dot{u}_1 = \dot{u}_2 = \omega$):

$$
T_a = \frac12\begin{bmatrix}\omega & \omega\end{bmatrix}
\rho A L\begin{bmatrix}\frac13 & \frac16 \\ \frac16 & \frac13\end{bmatrix}
\begin{bmatrix}\omega \\ \omega\end{bmatrix}
= \frac12\,\rho A L\,\omega^2
$$

Continuum: $T = \frac12 \rho A L \omega^2$ $\checkmark$

**y-direction translation** ($\dot{v}_1 = \dot{v}_2 = \omega$, $\dot{\theta}_{z1} = \dot{\theta}_{z2} = 0$):

$$
T_{bz} = \frac12\rho AL \Bigl[\frac{13}{35}\omega^2 + \frac{9}{70}\omega^2 + \frac{9}{70}\omega^2 + \frac{13}{35}\omega^2\Bigr]
= \frac12\rho AL\,\omega^2\left(\frac{26}{35} + \frac{18}{70}\right) = \frac12\rho AL\,\omega^2
$$

where $\frac{26}{35} + \frac{18}{70} = \frac{52}{70} + \frac{18}{70} = 1$ $\checkmark$

---

## 8. 3D Coordinate Transformation

### 8.1 Local Coordinate System

The local $x$-axis is along the element:

$$
\mathbf{e}_1 = \frac{\mathbf{p}_2 - \mathbf{p}_1}{L}, \qquad
L = \|\mathbf{p}_2 - \mathbf{p}_1\|
$$

The local $z$-axis is perpendicular to $\mathbf{e}_1$ and lies in the $\mathbf{e}_1$-$\mathbf{v}$ plane, where $\mathbf{v}$ is a reference vector (typically the global $Z$-axis $[0,0,1]^T$):

$$
\mathbf{e}_3 = \frac{\mathbf{v} \times \mathbf{e}_1}{\|\mathbf{v} \times \mathbf{e}_1\|}, \qquad
\mathbf{e}_2 = \mathbf{e}_3 \times \mathbf{e}_1
$$

### 8.2 Vertical Element Singularity

When $\mathbf{e}_1 \parallel \mathbf{v}$ (vertical element), $\mathbf{v} \times \mathbf{e}_1 = \mathbf{0}$. Use the global $X$-axis $[1,0,0]^T$ as the reference instead:

$$
\mathbf{e}_2 = \frac{\mathbf{e}_1 \times [1,0,0]^T}{\|\mathbf{e}_1 \times [1,0,0]^T\|}, \qquad
\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2
$$

### 8.3 Transformation Matrix

The 3×3 rotation matrix from global to local coordinates:

$$
\mathbf{T}_i = \begin{bmatrix} e_{1x} & e_{1y} & e_{1z} \\ e_{2x} & e_{2y} & e_{2z} \\ e_{3x} & e_{3y} & e_{3z} \end{bmatrix}
$$

Properties: $\mathbf{T}_i^T = \mathbf{T}_i^{-1}$, $\det(\mathbf{T}_i) = 1$.

The 12×12 block-diagonal transformation matrix:

$$
\boxed{\mathbf{T} = \begin{bmatrix}
\mathbf{T}_i & & & \\
& \mathbf{T}_i & & \\
& & \mathbf{T}_i & \\
& & & \mathbf{T}_i
\end{bmatrix}_{12\times12}}
$$

Matrix transformation:

$$
\mathbf{k}_{glob} = \mathbf{T}^T \mathbf{k}_{loc} \mathbf{T},\qquad
\mathbf{m}_{glob} = \mathbf{T}^T \mathbf{m}_{loc} \mathbf{T}
$$

---

## 9. Global Assembly

Using the direct stiffness method, element matrices are summed into the global system via DOF index mapping:

$$
\mathbf{K}_{glob} = \sum_{e=1}^{N_e} \mathbf{A}_e^T \mathbf{k}_{glob}^{(e)} \mathbf{A}_e
$$

Numerical implementation using precomputed index arrays:

```python
global_mat.at[I, J].add(local_mat.flatten())
```

Dirichlet boundary conditions are applied by zeroing rows/columns and setting diagonal entries to 1:

$$
\tilde{K}_{ii} = \begin{cases}
1 & i \in \text{BC} \\
K_{ii} & \text{otherwise}
\end{cases},\quad
\tilde{K}_{ij} = \begin{cases}
0 & i \in \text{BC} \text{ or } j \in \text{BC}, i \neq j \\
K_{ij} & \text{otherwise}
\end{cases}
$$

---

## 10. Common Implementation Errors

| Location | Incorrect | Correct | Reason |
|:---:|:---:|:---:|:---:|
| $r_x^2$ definition | $J / A$ | $(I_y + I_z)\,/\,A$ | Mass matrix uses the polar moment $I_p = I_y + I_z$ (kinetic energy), not the torsion constant $J$ (stiffness). For circular sections $J = I_p$, but for non-circular sections $J \neq I_p$. |
| $\mathbf{M}_b[3,3]$ | $-\,r_x^2/6$ | $+\,r_x^2/6$ | Torsional off-diagonal uses the same linear shape functions as axial, where the coupling is positive. A negative sign would under-report the rigid-body torsional kinetic energy by a factor of 3. |

The **stiffness matrix** is correct in both cases — it correctly uses $GJ$.
