# Chapter 2 Strain analysis

## Concept of deformation and strain

### 位移

物体内物质点A的位置矢量 $\pmb{r}=x\pmb{e}_x+y\pmb{e}_y+z\pmb{e}_z$，外力作用下A的位置矢量变为 $\pmb{R}=x'\pmb{e}_x+y'\pmb{e}_y+z'\pmb{e}_z$

> $x,y,z$ 称为物质坐标

A点的位移表示为两位置矢量之差 $\pmb{u}(\pmb{r})=\pmb{R}-\pmb{r}$

> 各点位移矢量的集合确定了物体的位移场，弹塑性力学中，通常假定位移场足够光滑，存在三阶以上连续导数

### 变形

物体经过位移后大小形状发生改变，称为变形

变形包括体积改变和形状畸变，位移远小于物体最小尺寸时称为小变形

### 应变

通过过物体内一点的任意微小线段即线元，在变形前后长度相对改变和方向相对改变，来描述物体内一点的变形

正应变： $\varepsilon=\frac{l-l_0}{l_0}$ （变形前后线元长度相对改变，伸长为正，缩短为负）

剪应变：$\gamma=\frac{\pi}{2}-\alpha$ (变形前后线元方向相对改变，夹角锐化为正，钝化为负)

> $\alpha$为变形后两线元（线元及与其在变形前垂直的辅助线元）之间的夹角

## Strain tensor & Geometric equation

考察线元$AB$的变形情况，A的空间位置坐标$(x,y,z)$，B的空间位置坐标$(x+dx,y+dy,z+dz)$，变形后到达新位置$A'B'$

B和A之间的相对位移矢量定义为$d\pmb{u}=\overrightarrow{BB'}-\overrightarrow{AA'}=\pmb{u}_B-\pmb{u}_A$

利用Talyor级数将$B$点位移相对$A$点展开，并略去二阶以上高阶项，相对位移分量为：

$du_i=(u_i)_B-(u_i)_A=u_{i,j}dx_j$

其中$u_{i,j}$为位移梯度张量（一般不对称）

$[u_{i,j}]=[\frac{\partial u_i}{\partial x_j}]=
\left[
\begin{matrix}
\frac{\partial u_x}{\partial x} & \frac{\partial u_x}{\partial y} & \frac{\partial u_x}{\partial z}\\
\frac{\partial u_y}{\partial x} & \frac{\partial u_y}{\partial y} & \frac{\partial u_y}{\partial z}\\
\frac{\partial u_z}{\partial x} & \frac{\partial u_z}{\partial y} & \frac{\partial u_z}{\partial z}
\end{matrix}
\right]$

### 正应变

变形前后线元的长度改变为

$\Vert d\pmb{R} \Vert^2-\Vert d\pmb{r} \Vert^2=(du_i+dx_i)(du_i+dx_i)-dx_idx_i$

在**小变形假定**下，位移梯度张量的分量均为小量，满足 $\vert \frac{\partial u_i}{\partial x_j} \vert \ll 1$，相应的乘积项可忽略

因此有

$\frac{1}{2}(\Vert d\pmb{R} \Vert^2-\Vert d\pmb{r} \Vert^2)=du_i dx_i=u_{i,j}dx_i dx_j=\frac{1}{2}(u_{i,j}+u_{j,i})dx_i dx_j$

其中$n_i=\frac{dx_i}{\Vert d\pmb{r} \Vert}$为方向余弦（同上）

考虑**小变形假定**，$\Vert d\pmb{R} \Vert-\Vert d\pmb{r} \Vert \ll \Vert d\pmb{r} \Vert$

$\frac{1}{2}(\Vert d\pmb{R} \Vert^2-\Vert d\pmb{r} \Vert^2)=\frac{1}{2}(\Vert d\pmb{R} \Vert-\Vert d\pmb{r} \Vert)(\Vert d\pmb{R} \Vert+\Vert d\pmb{r} \Vert)\approx\frac{1}{2}(\Vert d\pmb{R} \Vert-\Vert d\pmb{r} \Vert)2\Vert d\pmb{r} \Vert=(\Vert d\pmb{R} \Vert-\Vert d\pmb{r} \Vert)\Vert d\pmb{r} \Vert$

上式除以$\Vert d\pmb{r} \Vert^2$ ，再结合线元$AB$的方向余弦 $n_i=\frac{dx_i}{\Vert d\pmb{r} \Vert}$, 即得正应变表达式

$\varepsilon_n=\frac{1}{2}(u_{i,j}+u_{j,i})n_i n_j$

### 剪应变
线元$AB$的单位方向矢量为$\pmb{n}$，与$AB$垂直的线元为$AC$，单位方向矢量为$\pmb{s}$，变形后分别为和$A'C'$

定义应变矢量$\pmb{E}(\pmb{n})=\frac{du_i}{\Vert d\pmb{r} \Vert}\pmb{e}_i=\frac{u_{i,j}dx_j}{\Vert d\pmb{r} \Vert}=u_{i,j}n_j\pmb{e}_i$

考虑小变形假定，线元$AB$变形到$A'B'$产生的转角是

$\alpha\approx\frac{d\pmb{u}\cdot\pmb{s}}{\Vert d\pmb{R} \Vert}\approx\frac{d\pmb{u}\cdot\pmb{s}}{\Vert d\pmb{r} \Vert}=\pmb{E}(\pmb{n})\cdot\pmb{s}$

同理可得线元$AC$变形到$A'C'$产生的转角是

$\beta=\pmb{E}(\pmb{s})\cdot\pmb{n}$

最终可得剪应变

$\gamma_{ns}=\alpha+\beta=\pmb{E}(\pmb{n})\cdot\pmb{s}+\pmb{E}(\pmb{s})\cdot\pmb{n}=u_{i,j}n_j\pmb{e}_i\cdot s_k\pmb{e}_k+u_{i,j}s_j\pmb{e}_i\cdot n_k\pmb{e}_k=u_{i,j}n_j s_i+u_{i,j}s_j n_i=(u_{i,j}+u_{j,i})s_i n_j$

和正应变统一形式可得剪应变为：

$\frac{1}{2}\gamma_{ns}=\frac{1}{2}(u_{i,j}+u_{j,i})s_i n_j$

### 应变张量 & 几何方程

$\frac{1}{2}(u_{i,j}+u_{j,i})$定义了任意方向线元的应变，决定了一点的应变状态，是位移梯度张量对称化的结果，构成了应变张量

$\varepsilon_{i,j}=\frac{1}{2}(u_{i,j}+u_{j,i})$

上式定义了应变张量分量与位移分量之间的关系，称为**几何方程**（小应变、小转动）

> 几何方程的6个关系式(应变张量对称)是线性时，称为几何线性

$[\varepsilon_{ij}]=
\left[
\begin{matrix}
\frac{\partial u_x}{\partial x} & \frac{1}{2}(\frac{\partial u_x}{\partial y}+\frac{\partial u_y}{\partial x}) & \frac{1}{2}(\frac{\partial u_x}{\partial z}+\frac{\partial u_z}{\partial x})\\
\frac{1}{2}(\frac{\partial u_y}{\partial x}+\frac{\partial u_x}{\partial y}) & \frac{\partial u_y}{\partial y} & \frac{1}{2}(\frac{\partial u_y}{\partial z}+\frac{\partial u_z}{\partial y})\\
\frac{1}{2}(\frac{\partial u_z}{\partial x}+\frac{\partial u_x}{\partial z}) & \frac{1}{2}(\frac{\partial u_z}{\partial y}+\frac{\partial u_y}{\partial z}) & \frac{\partial u_z}{\partial z}
\end{matrix}
\right]$

> 应变张量的非对角分量$\varepsilon_{ij}(i\neq j)$为$i$轴方向和$j$轴方向之间的剪应变的一半

从而任意线元的应变用应变张量表示为：

$\varepsilon_n=\varepsilon_{ij}n_i n_j$

$\frac{1}{2}\gamma_{ns}=\varepsilon_{ij}s_i n_j$

## Rigid body rotation & Rotation tensor

### 转动张量

当物体仅产生刚体转动时，线元长度应保持不变，因此对于刚体转动，位移梯度张量必须是反对称的，即

$u_{i,j}=-u_{j,i}$

位移梯度张量可以分解为应变张量$\varepsilon_{i,j}$和转动张量$\Omega_{i,j}$之和

> 任意二阶张量可以分解为一个对称张量和反对称张量之和

$u_{i,j}=\varepsilon_{ij}+\Omega_{ij}$

其中 $\varepsilon_{ij}=\frac{1}{2}(u_{i,j}+u_{j,i})$，$\Omega_{ij}=\frac{1}{2}(u_{i,j}-u_{j,i})$

转动张量展开表示为：

$[\Omega_{ij}]=
\left[
\begin{matrix}
0 & \frac{1}{2}(\frac{\partial u_x}{\partial y}-\frac{\partial u_y}{\partial x}) & \frac{1}{2}(\frac{\partial u_x}{\partial z}-\frac{\partial u_z}{\partial x})\\
\frac{1}{2}(\frac{\partial u_y}{\partial x}-\frac{\partial u_x}{\partial y}) & 0 & \frac{1}{2}(\frac{\partial u_y}{\partial z}-\frac{\partial u_z}{\partial y}) \\
\frac{1}{2}(\frac{\partial u_z}{\partial x}-\frac{\partial u_x}{\partial z}) & \frac{1}{2}(\frac{\partial u_z}{\partial y}-\frac{\partial u_y}{\partial z}) & 0
\end{matrix}
\right]$

转动张量有三个独立的分量 

$\omega_x=\Omega_{32}=-\Omega_{23}$，$\omega_y=\Omega_{13}=-\Omega_{31}$，$\omega_z=\Omega_{21}=-\Omega_{12}$

将上述位移梯度表达式代入相对位移分量表达式，即 $du_i=(u_i)_B-(u_i)_A=u_{i,j}dx_j$

可得 $du_i=\Omega_{ij}dx_j+\varepsilon_{ij}dx_j$，使用矩阵形式可表示为：

$\left[
\begin{matrix}
u_x\\
u_y\\
u_z
\end{matrix}
\right]_B=
\left[
\begin{matrix}
u_x\\
u_y\\
u_z
\end{matrix}
\right]_A+
\left[
\begin{matrix}
0 & -\omega_z & \omega_y\\
\omega_z & 0 & -\omega_x\\
-\omega_y & \omega_x & 0
\end{matrix}
\right]
\left[
\begin{matrix}
dx\\
dy\\
dz
\end{matrix}
\right]
+
\left[
\begin{matrix}
\varepsilon_x & \frac{1}{2}\gamma_{xy} & \frac{1}{2}\gamma_{xz}\\
\frac{1}{2}\gamma_{yx} & \varepsilon_y & \frac{1}{2}\gamma_{yz}\\
\frac{1}{2}\gamma_{zx} & \frac{1}{2}\gamma_{zy} & \varepsilon_z
\end{matrix}
\right]
\left[
\begin{matrix}
dx\\
dy\\
dz
\end{matrix}
\right]$

上式表面线元$AB$的位移由三部分组成：A点的**平动**+转动张量引起的**刚体转动**+应变张量引起的**纯变形**

### 转动矢量

转动矢量/转动张量$\Omega_{ij}$的反偶矢量： $\pmb{\omega}=\omega_i\pmb{e}_i$

转动张量引起的转动可以看作是转动矢量$\pmb{\omega}$和线元矢量$d\pmb{r}$的矢量积

$\Omega_{ij}dx_j=(\pmb{\omega}\times d\pmb{r})_i$

几何含义：线元的末端点$B$，以$\pmb{\omega}$方向的直线为转轴，绕$A$点的刚体转动，转动角度即为转动矢量的模$\Vert \pmb{\omega} \Vert$

刚体运动时（$\pmb{\omega}$为常数，$\pmb{\varepsilon}=0$）任意一点的位移矢量表示： $\pmb{u}'=\pmb{u}+\pmb{\omega}\times\pmb{r}$

## Volume strain

设微六面体的边长分别是$dx,dy,dz$，以原点$M$为起点的三条线元沿坐标轴投影分别为 

$\overrightarrow{MA}=[dx,0,0]$，$\overrightarrow{MB}=[0,dy,0]$，$\overrightarrow{MC}=[0,0,dz]$

变形前体积为$V_0=\overrightarrow{MA}\times \overrightarrow{MB}\cdot \overrightarrow{MC}=dxdydz$

变形后线元沿坐标轴投影为

$dR_i=dr_i+du_i=dr_i+du_{i,j}dx_j$

进而可得

$\overrightarrow{M'A'}=[(1+\frac{\partial u_x}{\partial x})dx,\frac{\partial u_y}{\partial x}dx,\frac{\partial u_z}{\partial x}dx]$

$\overrightarrow{M'B'}=[\frac{\partial u_x}{\partial y}dy,(1+\frac{\partial u_y}{\partial y})dy,\frac{\partial u_z}{\partial y}dy]$

$\overrightarrow{M'C'}=[\frac{\partial u_x}{\partial z}dz,\frac{\partial u_y}{\partial z}dz,(1+\frac{\partial u_z}{\partial z})dz]$

微六面体变形后的体积是

$V=\overrightarrow{M'A'}\times \overrightarrow{M'B'}\cdot \overrightarrow{M'C'}=
\left[
\begin{matrix}
(1+\frac{\partial u_x}{\partial x})dx & \frac{\partial u_y}{\partial x}dx & \frac{\partial u_z}{\partial x}dx\\
\frac{\partial u_x}{\partial y}dy & (1+\frac{\partial u_y}{\partial y})dy & \frac{\partial u_z}{\partial y}dy\\
\frac{\partial u_x}{\partial z}dz & \frac{\partial u_y}{\partial z}dz & (1+\frac{\partial u_z}{\partial z})dz
\end{matrix}
\right]$

在小变形假定下，位移梯度张量的分量都是小量，乘积项可以略去，可得

$V=(1+\frac{\partial u_x}{\partial x}+\frac{\partial u_y}{\partial y}+\frac{\partial u_z}{\partial z})dxdydz=(1+\varepsilon_x+\varepsilon_y+\varepsilon_z)dxdydz$

体积应变为：

$\varepsilon_v=\frac{V-V_0}{V_0}=\varepsilon_x+\varepsilon_y+\varepsilon_z$

体积应变与剪切应变分量无关，即剪切应变不改变物体体积

## Properties of strain tensor

类似于应力张量，应变张量的坐标转换：
$\varepsilon_{m'n'}=\beta_{m'i}\beta_{n'i}\varepsilon_{ij}$

其中$\beta_{m'i}$和$\beta_{n'j}$代表新坐标轴$m',n'$在旧坐标系下的方向余弦

### 主应变和应变不变量

类似于应力张量

应变主方向/应变主轴：在此方向上只有正应变没有剪应变，其应变值称为主值或主应变

> 应变主轴相互垂直

设应变主方向为$\pmb{n}$，主值为$\varepsilon$，沿主应变方向取线元，只考虑线元的纯变形

线元的相对位移方向与$\pmb{n}$相同，因此，线元的应变矢量方向与$\pmb{n}$相同，即

$\pmb{E}(\pmb{n})=\varepsilon\pmb{n}$

$E_i=\varepsilon_i n_i$

纯变形中，转动张量$\Omega_{ij}=0$，位移梯度张量$u_{i,j}$与应变张量$\varepsilon_{ij}$相等

$E_i=u_{i,j}n_j=\varepsilon_{ij}n_j=
\left[
\begin{matrix}
\varepsilon_x & \frac{1}{2}\gamma_{xy} & \frac{1}{2}\gamma_{xz}\\
\frac{1}{2}\gamma_{yx} & \varepsilon_y & \frac{1}{2}\gamma_{yz}\\
\frac{1}{2}\gamma_{zx} & \frac{1}{2}\gamma_{zy} & \varepsilon_z
\end{matrix}
\right]
\left[
\begin{matrix}
l\\
m\\
n
\end{matrix}
\right]$

由上述两个$E_i$的表达式可以建立$n_i$非零解条件的特征方程

$\varepsilon^3-D_1\varepsilon^2+D_2\varepsilon-D_3=0$

应变张量的第一坐标不变量：$D_1=\varepsilon_{kk}$ (体积应变)

应变张量的第二坐标不变量：$D_2=\frac{1}{2}(D_1^2-\varepsilon_{ij}\varepsilon_{ij})$

应变张量的第三坐标不变量：$D_3=\frac{1}{3}(3D_1D_2-D_1^3+\varepsilon_{ij}\varepsilon_{jk}\varepsilon_{ki})$

> 解特征方程求得主值，再回代求出主方向，与主应力求解过程相似，此处略~

### 应变张量的分解

类似于应力张量，应变张量可分解为

$\varepsilon_{ij}=\varepsilon_0\delta_{ij}+e_{ij}$

其中$\varepsilon_0=\frac{1}{3}(\varepsilon_x+\varepsilon_y+\varepsilon_z)$为平均应变，$e_{ij}$为偏应变张量  

$\varepsilon_{ij}=\varepsilon_0\delta_{ij}$ 对应体积的等向膨胀或收缩，没有形状畸变

$\varepsilon_{ij}=e_{ij}$ 对应的应变状态为只有形状畸变而没有体积改变

偏应变主值求解：

$e^3-D'_1e^2+D'_2e-D'_3=0$

偏应变不变量：

$D'_1=e_{kk}=0$

$D'_2=\frac{1}{2}e_{ij}e_{ij}$

$D'_3=\frac{1}{3}e_{ij}e_{jk}e_{ki}$

> 偏应变主值，应变$\varepsilon$的Lode角与偏应力相关概念类似，此处略~

> 偏应变张量与应变张量的主方向一致，主值相差平均应变：$e_i=\varepsilon_i-\varepsilon_0$

等效应变：

$\overline{\varepsilon}=\sqrt{\frac{2}{3}e_{ij}e_{ij}}=\sqrt{\frac{2}{9}[(\varepsilon_1-\varepsilon_2)^2+(\varepsilon_2-\varepsilon_3)^2+(\varepsilon_3-\varepsilon_1)^2]}$

## Deformation compatibility equation

### 变形协调方程/相容方程

$\varepsilon_{ij,kl}+\varepsilon_{kl,ij}-\varepsilon_{ik,jl}-\varepsilon_{jl,ik}=0$

其中 $\varepsilon_{ij,kl}=\frac{\partial^2 \varepsilon_{ij}}{\partial x_k \partial x_l}$

对于单连通体来说，变形协调方程是位移单值连续的充分必要条件

对于多连通体来说，除满足协调方程外还应保证切口处（多连通体可以在适当切口剪开变为单连通体）位移单值连续

> 若位移函数已知，变形协调方程自然满足（由位移求应变）
>
> 反之，由应变求位移函数时，应变分量之间需满足变形协调方程

## Strain rate & Strain increment
### 构型
任意时刻$t$物体所占的区域称为构型

未变形状态（$t=0$）所占据的区域称为初始构型，建立固定的笛卡尔坐标系，物质点的位置矢量表示为

$\pmb{r}=x\pmb{e}_x+y\pmb{e}_y+z\pmb{e}_z$，其中$x,y,z$称为物质坐标/$Lagrangian$坐标

变形后所占的区域称为当前即时构型或当前构型，建立与初始构型相同的笛卡尔坐标系，物质点的变形后的位置矢量为

$\pmb{R}=x'\pmb{e}_x+y'\pmb{e}_y+z'\pmb{e}_z$，其中$x',y',z'$称为空间坐标/$Euler$坐标

> 在声明体积元、面积元、线元、应力、应变等物理量时，应当声明所相对的参考构型，不同参考构型使得物理量有不同定义

### 变形描述

$Lagrangian$描述/物质描述：以物质坐标为基本变量，始终追踪每一个物质点

$Euler$描述/空间描述：始终着眼于固定的空间点，占据空间点的物质点在不断变化

> 小变形假设下，$Lagrangian$坐标和$Euler$坐标之间的差别可以忽略，某些初始构型上的物理量可以近似当作即时构型上的物理量使用

### 变形率
对于位移场$\pmb{u}(x,y,z,t)$，物质点的位移相对时间的变化率即为物质点的运动速度

$v_i=\frac{\partial u_i}{\partial t}=\dot{u}_i$

已知时刻$t$物体的即时构型，在微小的时间间隔$dt$内，物质点的位移为$v_idt$

以即时构型为参考构型计算应变张量分量并除以$dt$得到单位时间产生的应变，称为变形率，用矩阵表示为

$[d_{ij}]=
\left[
\begin{matrix}
\frac{\partial v_x}{\partial x'} & \frac{1}{2}(\frac{\partial v_x}{\partial y'}+\frac{\partial v_y}{\partial x'}) & \frac{1}{2}(\frac{\partial v_x}{\partial z'}+\frac{\partial v_z}{\partial x'})\\
\frac{1}{2}(\frac{\partial v_y}{\partial x'}+\frac{\partial v_x}{\partial y'}) & \frac{\partial v_y}{\partial y'} & \frac{1}{2}(\frac{\partial v_y}{\partial z'}+\frac{\partial v_z}{\partial y'})\\
\frac{1}{2}(\frac{\partial v_z}{\partial x'}+\frac{\partial v_x}{\partial z'}) & \frac{1}{2}(\frac{\partial v_z}{\partial y'}+\frac{\partial v_y}{\partial z'}) & \frac{\partial v_z}{\partial z'}
\end{matrix}
\right]$

> 变形率是相对于即时构型的物理量，大变形情况下仍然适用

### 应变率张量

对应变张量求物质时间导数，得到应变率张量，可表示为

$[\dot{\varepsilon}_{ij}]=
\left[
\begin{matrix}
\frac{\partial v_x}{\partial x} & \frac{1}{2}(\frac{\partial v_x}{\partial y}+\frac{\partial v_y}{\partial x}) & \frac{1}{2}(\frac{\partial v_x}{\partial z}+\frac{\partial v_z}{\partial x})\\
\frac{1}{2}(\frac{\partial v_y}{\partial x}+\frac{\partial v_x}{\partial y}) & \frac{\partial v_y}{\partial y} & \frac{1}{2}(\frac{\partial v_y}{\partial z}+\frac{\partial v_z}{\partial y})\\
\frac{1}{2}(\frac{\partial v_z}{\partial x}+\frac{\partial v_x}{\partial z}) & \frac{1}{2}(\frac{\partial v_z}{\partial y'}+\frac{\partial v_y}{\partial z}) & \frac{\partial v_z}{\partial z}
\end{matrix}
\right]$

简记为 $\dot{\varepsilon}_{ij}=\frac{1}{2}(v_{i,j}+v_{j,i})$

应变率张量是相对初始构型而言的

小变形假设下，$Lagrangian$坐标和$Euler$坐标之间的差别可以忽略，变形率张量近似等于应变率张量

$d_{ij}=\dot{\varepsilon}_{ij}=\frac{1}{2}(v_{i,j}+v_{j,i})$

类似于应变张量$\varepsilon_{ij}$，应变率张量$\dot{\varepsilon}_{ij}$也可以求主方向、主应变率和不变量以及张量分解

应变率张量不变量

$\dot{\overline{\varepsilon}}=\sqrt{\frac{2}{3}\dot{e}_{ij}\dot{e}_{ij}}=\sqrt{\frac{2}{9}[(\dot{\varepsilon}_1-\dot{\varepsilon}_2)^2+(\dot{\varepsilon}_2-\dot{\varepsilon}_3)^2+(\dot{\varepsilon}_3-\dot{\varepsilon}_1)^2]}$

应变率张量和应变张量主方向一般不重合，其不变量和主应变率不等于应变张量的不变量和主应变求时间率,即

$\dot{\overline{\varepsilon}}\neq\frac{\partial}{\partial t}(\overline{\varepsilon})$

$\dot{\varepsilon}_i\neq\frac{\partial}{\partial t}(\varepsilon_i)$

> 只有应变各分量之间的比例在整个变形过程中始终保持不变时，上述等式关系才成立

### 应变增量
对于率无关材料（力学性质与应变率关系不大），可以用应变增量$d\varepsilon_{ij}$代替应变率

$d\varepsilon_{ij}=\dot{\varepsilon}_{ij}dt=\frac{1}{2}(\frac{\partial}{\partial x_j}(du_i)+\frac{\partial}{\partial x_i}(du_j))$

表示加载过程中的应变改变量
