# Why?
CNNs are not invariant to rotation and scale and more general affine transformation. Spatial transformer networks(STN) allow a nerual network to learn how to perform spatial transformation on input image in order to enhance the geometric invariance of the model. For example, crop, scale, skew, tranlate.
# What?
 general affine transformation:  
 $$ 
\begin{pmatrix}
x_i^s \\
y_i^s
\end{pmatrix}
= \tau_\theta(G_i) =
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}
$$
# How?
# Implement in Pytorch
