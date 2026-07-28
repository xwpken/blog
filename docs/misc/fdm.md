# Finite difference method

> Reference:
> 1. https://en.wikipedia.org/wiki/Finite_difference_method


## Approximation of first-order derivatives

From the Taylor series expansion, we can get

$$f(x_0+h) = f(x_0) + \frac{f^{\prime}(x_0)}{1!}h+ \frac{f^{(2)}(x_0)}{2!}h^2+...+\frac{f^{(n)}(x_0)}{n!}h^n+R_n(x)$$

where $R_n(x)$ is the remainder term representing the difference of the approximation.

Considering the expansion at $f(x_0+h)$ and $f(x_0-h)$, respectively, we can get three types of difference scheme, that is

* Forward-differnce scheme (FDS)
$$
(\frac{\partial f}{\partial x})_{x_0}=\frac{f(x_0+h)-f(x_0)}{h}
$$

* Backward-differnce scheme (BDS)
$$
(\frac{\partial f}{\partial x})_{x_0}=\frac{f(x_0)-f(x_0-h)}{h}
$$

* Central-differnce scheme (CDS)
$$
(\frac{\partial f}{\partial x})_{x_0}=\frac{f(x_0+h)-f(x_0-h)}{2h}
$$


### An example - The heat equation

$$
\begin{cases}u_t=u_{xx}\\u(0,t)=u(1,t)=0&\text{(boundary condition)}\\u(x,0)=u_0(x)&\text{(initial condition)}\end{cases}
$$

#### Explicit method - FTCS

With the forwaed difference in time and central difference in space, we can get：

$$
\frac{u^{n+1}_j-u_j^n}{\Delta t} = \frac{u_{j+1}^n-2u_{j}^n+u_{j-1}^n}{h^2}
$$

which leads to the solution scheme:

$$
u_j^{n+1}=(1-2r)u_j^n + ru_{j-1}^n + ru_{j+1}^n
$$

where $r = \Delta t/h^2$

This explicit scheme is numerically stable when $r\leq1/2$. The numerical errors are:

$$\Delta u = O(\Delta t)+O(h^2)$$


#### Implicit method - BTCS

If we use the backward difference at time $t_{n+1}$, the FTCS scheme can be modified to BTCS, i.e.

$$
\frac{u^{n+1}_j-u_j^n}{\Delta t} = \frac{u_{j+1}^{n+1}-2u_{j}^{n+1}+u_{j-1}^{n+1}}{h^2}
$$

The solution scheme can be stated as:

$$
(1+2r)u_j^{n+1}-ru_{j-1}^{n+1}-ru_{j+1}^{n+1}=u_j^n
$$

This scheme is always numerical stable but requires solver a system of numerical equations. The error are:

$$
\Delta u = O(\Delta t)+O(h^2)
$$
