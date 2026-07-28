# Chapter 2 Nonlinear Finite Element Analysis Procedure

> Last updated: 09 Jun 2026 

## 2.1 Introduction to Nonlinear Systems in Solid Mechanics

~

## 2.2 Solution Procedures for Nonlinear Algebraic Equations

~

### 2.2.1 Newton-Raphson Method

The Newton-Raphson method converges quadratically (**when the initial estimate is close to the solution**) when there exists a constant $c$ such that

$$
\| \boldsymbol{u}_{\text{exact}} - \boldsymbol{u}_{n+1} \| \leq c \| \boldsymbol{u}_{\text{exact}} - \boldsymbol{u}_n \|^2
$$

In practice, $\boldsymbol{u}_{\text{exact}}$ is unknown, and the solution at the convergered iteration is used as an approximation. 
To show the quadratic convergence, it is required to show the follwoing ratio approaches a constant $c$ as $n \to \infty$:

$$
\lim_{n \to \infty} \frac{\| \boldsymbol{u}_{\text{exact}} - \boldsymbol{u}_{n+1} \|}{\| \boldsymbol{u}_{\text{exact}} - \boldsymbol{u}_n \|^2} = c
$$

The Newton-Raphson method does not always guarantee convergence to the accurate solution. First, it **assumes that solution increment is relatively small**, but this is violated when the Jacobian matrix is singular or nearly singular. Second, the method may diverge or oscillate if **the initial guess is too far away from the exact solution**, or **the curvature of the internal force change its sign between two consecutive iterations**.

### 2.2.2 Modified Newton-Raphson Method
The initial tangent stiffness matrix is repeatedly used for all iterations, which can be combined with **LU factorization** ($\mathcal{O}(n^3)$ for factorization and $\mathcal{O}(n^2)$ for forward/backward substitution) to reduce the computational cost. This method is a little more stable and is not prone to divergence, but it dose not provide quadratic convergence.

### 2.2.3 Incremental Secant Method
The core idea of the incremental secant method is to use the secant stiffness matrix to approximate the tangent stiffness matrix. For the single variable case, the convergence ratio is the golden ratio $0.618$. This **quasi-Newton method** is less expensive but converges slower than the Newton-Raphson method. 

For the multi-variable case, the secant stiffness matrix can be updated using the **Broyden's method**. The solution increment is obtained by:

$$
\boldsymbol{K}_s^i \Delta \boldsymbol{u}^i = -\boldsymbol{R}^i
$$

where $\boldsymbol{K}_s^i$ is the secant stiffness matrix at iteration $i$. The idea is that the Jacobian matrix is calculated at the first iteration, and then updated using the **rank-one update**:

$$
\boldsymbol{K}_s^i = \boldsymbol{K}_s^{i-1} + \frac{(\Delta \boldsymbol{R}^i - \boldsymbol{K}_s^{i-1} \Delta \boldsymbol{u}^i)}{\| \Delta \boldsymbol{u}^i \|^2}\left(\Delta \boldsymbol{u}^{i}\right)^{\text{T}}
$$

where $\Delta \boldsymbol{R}^i = \boldsymbol{R}^i - \boldsymbol{R}^{i-1}$ and $\Delta \boldsymbol{u}^i = \boldsymbol{u}^i - \boldsymbol{u}^{i-1}$.

Another Broyden's method is to update the inverse of the secant stiffness matrix $\boldsymbol{H}_s^i$:

$$
\boldsymbol{H}_s^i = \boldsymbol{H}_s^{i-1} + \frac{(\Delta \boldsymbol{u}^i - \boldsymbol{H}_s^{i-1} \Delta \boldsymbol{R}^i)}{\left(\Delta\boldsymbol{u}^i\right)^{\text{T}}\boldsymbol{H}_s^{i-1} \Delta \boldsymbol{R}^i}\left(\left(\Delta\boldsymbol{u}^i\right)^{\text{T}} \boldsymbol{H}_s^{i-1}\right)
$$

It should be noted that the above update formula cannot guarantee the symmetry and positive definiteness. To address this issue, the **BFGS method** is usually adopted, but can become unstable when the iteration number increases. In practice, the secant stiffness matrix is reset to the stiffness matrix of the Newton-Raphson method after a certain number of iterations.

For most quasi-Newton methods, the convergence rate is superlinear, with an order between $1$ and $2$. 

### 2.2.4 Incremental Force Method
The idea of the incremental force method is to **apply the load in increments** and use the solution from the previous load step as the **initial guess** for the next load step.

---

The best way to check if the load step is too small or too large is to **count the number of iterations**. For the standard Newton-Raphson method, the load step is considered to be appropriate if the solution converges around **5 or 6 iterations**. Otherwise, the load step should be reduced or increased accordingly. 

For complex nonlinear problems, it is possible to go back to the previous converged solution and **reduce the load step by half** if the solution does not converge after a certain number of iterations. 

These strategies are called **adaptive time stepping**.

---

In the displacement-controlled procedure, the solution can be converged in a broader range of displacements.


## 2.3 Steps in the Solution of Nonlinear Finite Element Analysis

State determination --> Residual calculation --> Convergence check --> Linearization --> Solution