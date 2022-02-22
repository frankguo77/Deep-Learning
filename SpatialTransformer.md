# Why?
CNNs are not invariant to rotation and scale and more general affine transformation. Spatial transformer networks(STN) allow a nerual network to learn how to perform spatial transformation on input image in order to enhance the geometric invariance of the model. For example, crop, scale, skew, tranlate.
# What & How?
General affine transformation:  
 $$ 
\begin{pmatrix}
x_i^s \\
y_i^s
\end{pmatrix}
= \tau_\theta(G_i) =
A_theta
\begin{pmatrix}
x_i^t \\
y_i^t \\
1
\end{pmatrix}
=
\begin{bmatrix}
\theta_{11} & \theta_{12}  & \theta_{13} \\
\theta_{21} & \theta_{22}  & \theta_{23} 
\end{bmatrix}
\begin{pmatrix}
x_i^t \\
y_i^t \\
1
\end{pmatrix}
$$

The archtecture of STN
<p align="center">
![logo](https://pytorch.org/tutorials/_images/stn-arch.png "STN")
</p>
*The ocalization Network(regular CNN)
*The grid generator
*The sampler

Let's see the three parts one by one.
## Localization network
find affine matrix A_\theta. \theta_{11} ~ \theta_{23}.
 

The input of the localisation net is a feature map: $U \in R^{HXWXC}$ 。经过若干卷积或全链接操作后接一个回归层回归输出变换参数θ。θ的维度取决于网络选择的具体变换类型,如选择仿射变换则 [公式] 。如选择投影变换则 [公式] 。θ的值决定了网络选择的空间变换的”幅度大小”。

## Grid generator
Do the matrix product to do affine transformation.

## Interpolation
After we do the affine transformation, by the matrix production. we gone get the decimal coordinates. we can not simply rund them  t integer because it will make  the process differentiable, we need interpolate. 
<p align="center">
![logo](https://i.imgur.com/wd0Xd2T.png "Interpolation")
</p>


# How?
# Implement in Pytorch
