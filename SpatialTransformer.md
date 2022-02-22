# Why?
CNNs are not invariant to rotation and scale and more general affine transformation. Spatial transformer networks(STN) allow a nerual network to learn how to perform spatial transformation on input image in order to enhance the geometric invariance of the model. For example, crop, scale, skew, tranlate.
# What and How?
General affine transformation:  
$$
\begin{pmatrix}
x_i^s \\
y_i^s
\end{pmatrix}
= \tau_\theta(G_i) =
A_\theta
\begin{pmatrix}
x_i^t \\
y_i^t \\
1
\end{pmatrix}=
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
  <img src="https://pytorch.org/tutorials/_images/stn-arch.png" alt="STN"/>
</p> 

- The ocalization Network(regular CNN)
- The grid generator
- The sampler

Let's see the three parts one by one.
## Localization network
Find affine matrix $$A_\theta$$: $$\theta_{11}$$ ~ $$\theta_{23}$$.
 
Input: $$U \in R^{H \times W \times C}$$  
Output: $$\theta \in R^{2 \times 3}$$ (2D) or $$\theta \in R^{3 \times 4}$$ (3D)


## Grid generator
Do the matrix product to do affine transformation.

## Interpolation
After we do the affine transformation by the matrix production. we gonna get the decimal coordinates. we can not simply rund them  to integer because it will make the opteration not differentiable, we need interpolate. 
<p align="center">
  <img src="https://i.imgur.com/wd0Xd2T.png" alt="Interpolation"/>
</p> 

# Implement in Pytorch
The first part of localization network is to extracts features of input image 
```python
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
   ```
 The second part is to do the $\theta$ regression
```python        
       # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
```

STN operation:
```python
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
```
F.affine_grid do affine transform and outputs: $(N \times H \times W \times 2)$.
F.grid_sample do interpolation.
