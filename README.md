# Quaternion Dynamical System Learning

Quaternion-DS



---

### Update
9/2
- scale the result angular velcotiy
- extend to multiple DS learning

9/1
- ~~propose a feasible dimension reduction to tackle singular covariance matrix resulting from unit vectors~~
- ~~construct double DS learning~~
- start on quaternion clustering


8/31
- ~~verify if system stable in tangent space~~
- ~~verify the equivalence between tan and quat~~


8/29 
- ~~finshi plot_tools.py~~
- ~~optimize_single_system~~
- ~~retrieve mean and covariance~~
- ~~construct quat normal dist~~
- plot sequence of rotation and show clusters in time series

---
thoughts
- angular velocity directly acted on current orientation(world frame vs. local frame)
- is cannocal quaternion always the right option? maybe should choose the one closest to the att out of two