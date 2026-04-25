# Tensors for begineers
> from 'eigenchris' on Youtube
> 
> Last modified: 20 November, 2022 
> 
## 1. Defination of tensor

+ **Tensor** : an object is ```invariant``` under a change of coordinates, and ...\
has ```componets``` that change in a ```special```, ```predictable``` way under  a change of coordinates.


+ **Tensor** : a collection of ```vectors``` and ```covectors``` combined  together using the ```tensor product```.


+ **Tensor** : as ```partial derivatives``` and ```gradients``` that transform with ```jacobian Matrix```


## 2. Forward and Backward Transforms

**Forward transforms:**
$\widetilde{\vec {e}_i}=\sum_{j=1}^{n}F_{ji}\vec{e}_j$
 

**Backward transforms:**
$\vec{e}_i=\sum_{j=1}^{n}B_{ji}\widetilde{\vec {e}_j}$

$\sum_{j}F_{kj}B_{ji}=\delta_{ik}$ 
```(Kronecker delta)```

## 3. Vectors & Covectors

### 3.1. Defination
+ ***Vector*** : a member of  ```vector space```-->
***(V,S,+,.)***\
***v*** : Set of Vectors\
***S*** : Set of Scalars\
***+*** :  Vector addition rule\
***.*** : Vector scaling

+ ***Covector*** : 
  + Fucntions $\alpha:V-\mathbb{R}$ that map a vector to a number and also obey the following rules:\
$\alpha({\vec {v}}+{\vec {w}})=\alpha({\vec {v}})+\alpha({\vec {w}})$\
$\alpha(n{\vec {v}})=n\alpha({\vec {v}})$
  + Elements of $V^{*}$
```dual vector space```-->
***($V^{*}$,S,+,.)***\
$(n\cdot\alpha){\vec {v}}=n\alpha({\vec {v}})$\
$(\beta+\gamma){\vec {v}}=\beta({\vec {v}})+\gamma({\vec {v}})$
### 3.2. Transforms Rules

***Vector*** :
${\vec {v}}=\sum_{i=1}^{n}v^{i}\vec{e}_i=\sum_{i=1}^{n}\widetilde{v^{i}}\vec{e}_i$

***Covector*** :
${{\alpha}}=\sum_{i=1}^{n}\alpha_{i}{\epsilon}^i=\sum_{i=1}^{n}\widetilde{\alpha_{i}}\widetilde{{\epsilon}^i}$

+ **Covarient**
  + **Basic vectors:**\
$\widetilde{\vec {e}_j}=\sum_{i=1}^{n}F_{ij}\vec{e}_i$\
$\vec{e}_j=\sum_{i=1}^{n}B_{ij}\widetilde{\vec {e}_i}$

  + **Covector components:**\
$\widetilde{{\alpha}_j}=\sum_{j=1}^{n}F_{ij}{\alpha}_i$
 \
${\alpha}_j=\sum_{j=1}^{n}B_{ij}\widetilde{{\alpha}_i}$

+ **Contravarient**

  + **Basic covectors:**\
$\widetilde{\epsilon^i}=\sum_{i=1}^{n}B_{ij}\epsilon^j$\
$\epsilon^i=\sum_{i=1}^{n}F_{ij}\widetilde{\epsilon^j}$

  + **Vector components:**\
$\widetilde{{v}^i}=\sum_{j=1}^{n}B_{ij}{v}^j$
 \
${v}^i=\sum_{j=1}^{n}F_{ij}\widetilde{{v}^j}$


## 4. Linear Maps
