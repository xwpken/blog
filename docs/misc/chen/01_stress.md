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
