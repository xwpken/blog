# Chapter 1 Stress analysis

## Stress vector

$\pmb{T}(\pmb{n})=\lim\limits_{\Delta s\to0}\frac{\Delta \pmb{F}}{\Delta S}$

$\pmb{T}(\pmb{n})=T_ie_i$

## Stress tensor

$[\sigma_{ij}]=
\left[
\begin{matrix}
\sigma_{11} & \sigma_{12} & \sigma_{13}\\
\sigma_{21} & \sigma_{22} & \sigma_{23} \\
\sigma_{31} & \sigma_{32} & \sigma_{33}
\end{matrix}
\right]=
\left[
\begin{matrix}
\sigma_{xx} & \tau_{xy} & \tau_{xz}\\
\tau_{yx} & \sigma_{yy} & \tau_{yz} \\
\tau_{zx} & \tau_{zy} & \sigma_{zz}
\end{matrix}
\right]$


斜六面体(Oblique hexahedron)3个正面上的应力矢量：
$\pmb{T}(\pmb{e_i})=\sigma_{ik}\pmb{e_k}$

>一点的应力张量$\sigma_{ij}$完全决定了该点的应力状态，即若已知3个相互垂直面上的应力矢量（表示为应力张量$\sigma_{ij}$）其他任意一斜面上的应力矢量可根据该点的平衡条件导出  $\Downarrow$


## Cauchy formula (Oblique stress formula)

考虑由三个负面和一个斜面组成的微四面体，斜面的外法线单位矢量为$n$，

与3个坐标轴的投影分别为$l,m,n$(即与各坐标轴的夹角余弦)，由微四面体的平衡条件并且忽略高阶小量（体力项）可得

斜面的应力矢量为：

$\pmb{T}(\pmb{n})=\pmb{T}(\pmb{e_x})l+\pmb{T}(\pmb{e_y})m+\pmb{T}(\pmb{e_z})n=n_i\pmb{T}(\pmb{e_i})$ 

$T_j=n_i\sigma_{ij}$

> $\Uparrow$  Cauchy公式，斜面应力公式

斜面的正应力分量:
$\sigma_n=\pmb{T}(\pmb{n})\cdot \pmb{n}=T_jn_j=\sigma_{ij}n_in_j$

斜面的剪应力分量：
$\tau_n=\sqrt{\Vert \pmb{T}(\pmb{n}) \Vert^2-\sigma_n^2}$

## Equilibrium differential equation
使用微六面体代表物体内的一点,作用在微六面体上的所有力应满足平衡条件
分别考虑微六面体在三个方向上的力平衡可得

**平衡微分方程**：
$\sigma_{ij,i}+F_j=0$
> $\pmb{F}$ 为体力

考虑对坐标轴的三个力矩平衡可得

**剪应力互等定理**：
$\sigma_{ij}=\sigma_{ji}$

## Boundary conditions for force
力的边界条件指力边界上各点的**应力**与已知**表面力**应满足的关系

$n_i\sigma_{ij}=\bar{T}_j$

本质上是物体边界点的平衡条件

> $\pmb{\bar{T}}$ 为该点上作用的表面力矢量


## Coordinate transformation of stress components
旧坐标系的基矢量：$\pmb{e}_i$

新坐标系的基矢量：$\pmb{e^{'}}_i$，在旧坐标系中的投影为 
$(l_i,m_i,n_i)$

定义新旧坐标的转换矩阵为
$[\beta]=
\left[
\begin{matrix}
l_{1} & m_{1} & n_{1}\\
l_{2} & m_{2} & n_{2} \\
l_{3} & m_{3} & n_{3}
\end{matrix}
\right]$

基矢量的坐标变换
$\pmb{e^{'}_i}=\beta_{ij}\pmb{e}_j$

由Cauchy斜面应力公式可知

$\pmb{T}(\pmb{e^{'}}_i)=\beta_{ij}\pmb{T}(\pmb{e}_j)=\beta_{ij}\sigma_{jk}\pmb{e}_k$

新坐标系下的应力分量为：

$\sigma^{'}_{mn}=\pmb{T}(\pmb{e^{'}}_m)\cdot\pmb{e^{'}}_n=\beta_{mi}\sigma_{ik}\pmb{e}_k \cdot \beta_{nj}\pmb{e}_j=\beta_{mi}\beta_{nj}\sigma_{ik}\delta_{kj}=\beta_{mi}\beta_{nj}\sigma_{ij}$
> $\delta_{kj}$为Kronecker $\delta$ 符号，又称二阶单位张量

对应的矩阵形式为

$[\sigma']=[\beta][\sigma][\beta]^T$

应力分量在坐标变换时满足上述变换准则，因此应力为二阶张量

## Principle stress & Stress tensor invariants
主平面上只有正应力的作用，剪应力为零

主平面的外法线方向称为主方向，沿三个主方向的直线称为主轴

$\pmb{T}(\pmb{n})=\sigma\pmb{n}$

$T_i=\sigma n_i$

由Cauchy公式可得关于$n_i$的齐次方程

$\begin{cases}
(\sigma_x-\sigma)l+\tau_{yx}m+\tau_{zx}n=0\\
\tau_{xy}l+(\sigma_y-\sigma)m+\tau_{zy}n=0\\
\tau_{xz}l+\tau_{yz}m+(\sigma_z-\sigma)n=0\\
\end{cases}$

由于 $l^2+m^2+n^2=1$ 故该方程组应有非零解，因此系数矩阵行列式应为零

结合剪应力互等定理可得

$\begin{vmatrix}
\sigma_x-\sigma & \tau_{xy} & \tau_{xz} \\
\tau_{yx} & \sigma_y-\sigma & \tau_{yz} \\
\tau_{zx} & \tau_{zy} & \sigma_z-\sigma \\
\end{vmatrix}=0$

展开可得一元三次方程

$\sigma^3-I_1\sigma^2+I_2\sigma-I_3=0$

其中

$I_1=\sigma_{kk}$

$I_2=\frac{1}{2}(I_1^2-\sigma_{ij}\sigma_{ij})$

$I_3=\frac{1}{3}(3I_1I_2-I_1^3+\sigma_{ij}\sigma_{jk}\sigma_{ki})$

解出三个特征根 $\sigma_i$ 即为主应力，回代求得三组 $n_i$ 可得三个主方向

**主应力的特性**：极值性、主方向相互垂直、$I_1,I_2,I_3$ 的坐标不变性

> $I_1,I_2,I_3$ 的坐标不变性指的是，由其他坐标系下的应力分量求主应力时，由于主应力与坐标系的选择无关，因此待解方程的未知系数应保持一致，即$I_1,I_2,I_3$为坐标不变量

## Maximum shear stress & Mohr cricle   
在以主方向为坐标轴的坐标系中，以 $l,m,n$ 为投影的外法线所指示的任一斜面上的正应力 $\sigma_n$ 和剪应力 $\tau_n$应满足：

$\begin{cases}
\tau_n^2+\sigma_n^2=\Vert \pmb{T}\Vert^2=(l\sigma_1)^2+(m\sigma_2)^2+(n\sigma_3)^2\\
\sigma_n=l^2\sigma_1+m^2\sigma_2+n^2\sigma_3\\
l^2+m^2+n^2=1\\
\end{cases}$

进而有

$l^2=\frac{\tau_n^2+(\sigma_n-\sigma_2)(\sigma_n-\sigma_3)}{(\sigma_1-\sigma_2)(\sigma_1-\sigma_3)}\geq 0$

$m^2=\frac{\tau_n^2+(\sigma_n-\sigma_3)(\sigma_n-\sigma_1)}{(\sigma_2-\sigma_3)(\sigma_2-\sigma_1)}\geq 0$

$n^2=\frac{\tau_n^2+(\sigma_n-\sigma_1)(\sigma_n-\sigma_2)}{(\sigma_3-\sigma_1)(\sigma_3-\sigma_2)}\geq 0$

设 $\sigma_1 \geq \sigma_2 \geq \sigma_3$，有

$\tau_n^2+(\sigma_n-\frac{\sigma_2+\sigma_3}{2})^2 \geq \frac{(\sigma_2-\sigma_3)}{2}^2$

$\tau_n^2+(\sigma_n-\frac{\sigma_3+\sigma_1}{2})^2 \leq \frac{(\sigma_3-\sigma_1)}{2}^2$

$\tau_n^2+(\sigma_n-\frac{\sigma_1+\sigma_2}{2})^2 \geq \frac{(\sigma_1-\sigma_2)}{2}^2$

不同外法线方向的斜平面上 $\sigma_n,\tau_n$ 符合上述规律，并在$\sigma \sim \tau$ 坐标系中表示为Mohr图（Mohr圆，应力圆）

Mohr圆描述了一点的应力状态及其主应力、最大应力的情况

Mohr圆上各点的坐标代表与某个主应力方向平行面上的应力


**平面应力状态的Mohr圆**

对于微单元体中与xy坐标平面垂直且与x轴夹角为$\theta$的任意平面，外法线可表示为

 $\pmb{n}=\cos\theta\pmb{e_x}+\sin\theta\pmb{e_y}$

 xy平面内与外法线垂直的矢量表示为

 $\pmb{s}=-\sin\theta\pmb{e_x}+\cos\theta\pmb{e_y}$

 斜面的应力分量可表示为

$\sigma_n=\pmb{T}(\pmb{n})\cdot\pmb{n}=T_k\pmb{e}_k\cdot n_j\pmb{e}_j=n_i\sigma_{ik}n_j\delta_{kj}=n_in_j\sigma_{ij}$


$\tau_n=\pmb{T}(\pmb{n})\cdot\pmb{s}=T_k\pmb{e}_k\cdot s_j\pmb{e}_j=n_i\sigma_{ik}s_j\delta_{kj}=n_is_j\sigma_{ij}$

代入可得

$\sigma_n=\sigma_x\cos^2\theta+\sigma_y\sin^2\theta+2\tau_{xy}\sin\theta\cos\theta=\frac{1}{2}(\sigma_x+\sigma_y)+\frac{1}{2}(\sigma_x-\sigma_y)\cos 2\theta+\tau_{xy}\sin 2\theta$

$\tau_n=-(\sigma_x-\sigma_y)\sin\theta\cos\theta+\tau_{xy}(cos^2\theta-sin^2\theta)=-\frac{1}{2}(\sigma_x-\sigma_y)\sin 2\theta+\tau_{xy}\cos 2\theta$

$\sigma_n$最右边第一项移到左端，两等式两端各取平方并相加，最终得：

$(\sigma_n-\frac{\sigma_x+\sigma_y}{2})^2+\tau_n^2=(\frac{\sigma_x-\sigma_y}{2})^2+\tau_{xy}^2$

主应力为 $\sigma_{1,2}=\frac{\sigma_x+\sigma_y}{2}\pm\sqrt{(\frac{\sigma_x-\sigma_y}{2})^2+\tau_{xy}^2}$

> 剪应力$\tau_n$使微单元体逆时针旋转为正
> 
> 微单元体上斜面的外法线矢量$\pmb{n}$逆时针旋转$\theta$，在Mohr圆中对应点应顺时针旋转$2\theta$
>
> 最大剪应力方向所在的平面与中主应力平行（$\sigma_1\geq\sigma_2\geq\sigma_3$，$\sigma_2$为中主应力）与最大和最小主应力主方向的夹角为$45^{\circ}$，大小为Mohr圆半径

## Stress deviator and its invariants
一点的应力状态可以分解为：静水压力状态和偏应力状态

$\sigma_{ij}=\sigma_0\delta_{ij}+s_{ij}$

球形张量：$\sigma_0\delta_{ij}$，
其中
$\sigma_0=\frac{1}{3}\sigma_{ii}$

偏应力张量：$s_{ij}$

类似于应力不变量的推导过程，偏应力不变量为

$J_1=s_{kk}=tr[s_{ij}]=0$

$J_2=\frac{1}{2}s_{ij}s_{ij}=-I_2+\frac{1}{3}I_1^2$

$J_3=\frac{1}{3}s_{ij}s_{jk}s_{ki}=det[s_{ij}]=I_3-\frac{1}{3}I_1 I_2+\frac{2}{27}I_1^3$

## Stress and equivalent stress on Octahedron
考虑物体中一点，过该点作一外法线$\pmb{n}$与三个主应力方向有相同角度的斜面，称为等斜面，共计8个

方向余弦为
$(l,m,n)=(\pm\frac{1}{\sqrt{3}},\pm\frac{1}{\sqrt{3}},\pm\frac{1}{\sqrt{3}})$

8个等斜面组成的微单元体称为八面体

等斜面上的剪应力和正应力分别表示为

$\tau_0=\frac{1}{3}\sqrt{(\sigma_1-\sigma_2)^2+(\sigma_2-\sigma_3)^2+(\sigma_3-\sigma_1)^2}=\sqrt{\frac{2}{3}J_2}=\sqrt{\frac{1}{3}s_{ij}s_{ij}}$

$\sigma_0=\frac{1}{3}(\sigma_1+\sigma_2+\sigma_3)$

等效应力/Von Mises应力：

$\overline{\sigma}=\sqrt{\frac{3}{2}s_{ij}s_{ij}}=\sqrt{3J_2}=\frac{1}{\sqrt{2}}\sqrt{(\sigma_1-\sigma_2)^2+(\sigma_2-\sigma_3)^2+(\sigma_3-\sigma_1)^2}$



## Principle stress space & $\pi$ plane

建立由主应力$\sigma_i$为坐标轴的直角坐标系，称为主应力空间

主应力空间中任意一点代表物体一点的应力状态

$\overrightarrow{OP}=\sigma_i\pmb{e}_i$

静水压力轴：过原点$O$且与三个坐标轴具有相同夹角的直线

$\pi$平面：过原点$O$并以静水压力轴为法线的平面

任一应力状态$\overrightarrow{OP}$可分解为静水压力轴和$\pi$平面上投影的矢量和

$\overrightarrow{OP}=\sigma_o\pmb{e}_i+s_i\pmb{e}_i$

根据相应的几何关系建立$\pi$平面中任意一点的平面坐标$(x,y)$与主应力空间坐标$(\sigma_1,\sigma_2,\sigma_3)$之间的关系

主偏应力矢量的模为

$r_{\sigma}=\sqrt{2J_2}$

主偏应力矢量与$\pi$平面中$x$轴的夹角为Lode角，记为$\theta_{\sigma}$

$\tan\theta_{\sigma}=\frac{1}{\sqrt{3}}\mu_{\sigma}$

其中$\mu_{\sigma}=\frac{2\sigma_2-\sigma_1-\sigma_3}{\sigma_1-\sigma_3}$为Lode参数，表示主应力之间的相对比值关系

> 偏应力张量的三个主值也可由$J_2$和Lode角确定，此处略

**PS：材料失效强度理论**

延性材料的失效由屈服产生，仅取决于偏应力分量

+ Rankine (Maximum principle stress theory)
  
    屈服发生在最大(小)主应力等于材料的屈服强度
  
  $\sigma_{max}=\sigma_y$

+ Tresca (Maximum shear stress theory)

    屈服发生在最大剪应力等于单轴拉伸试验中屈服时剪应力

    $\tau_{max}=\frac{\sigma_1-\sigma_3}{2}=\tau_y$

+ Von Mises (Maximum distortion theory)

    屈服发生材料的畸变能等于单轴拉伸试验中屈服时材料的畸变能

    $u_d=u_{d,y}$

    畸变能$u_d=\frac{1+\nu}{6E}[(\sigma_1-\sigma_2)^2+(\sigma_2-\sigma_3)^2+(\sigma_3-\sigma_1)^2]$

    单轴拉伸试验中屈服畸变能$u_{d,y}=\frac{1+\nu}{3E}\sigma_y^2$

    进而可得屈服时

    $\sigma_y=\sqrt{\frac{1}{2}(\sigma_1-\sigma_2)^2+(\sigma_2-\sigma_3)^2+(\sigma_3-\sigma_1)^2}$

    上式右端即为等效应力$\sigma_{eq}$

脆性材料失效由于断裂产生，受到静水压力和偏应力的共同影响

> 脆性材料一般抗压强度大于抗拉强度
>
> 
Coulomb-Mohr理论（失效包络Coulomb-Mohr理论）

绘制单轴压缩试验和拉伸试验的两个相切的Mohr圆，作两圆的公切线，与该切线相切的Mohr圆代表的应力状态表示失效



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