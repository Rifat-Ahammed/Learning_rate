# Learning_rate

The learning rate in machine learning and deep learning is a hyperparameter that controls how much
we are adjusting the weights of our network with respect to the loss gradient. The lower the value, the
slower we travel along the downward slope. While this might be a good approach for not missing any
local minima, it could also mean that we’ll be taking a very long time to reach the bottom, if we ever
do. On the other hand, a high learning rate might allow us to converge quickly, but there's a risk of
overshooting the lowest point if the step size is too large.

