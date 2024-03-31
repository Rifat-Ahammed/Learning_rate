# Learning_rate

The learning rate in machine learning and deep learning is a hyperparameter that controls how much
we are adjusting the weights of our network with respect to the loss gradient. The lower the value, the
slower we travel along the downward slope. While this might be a good approach for not missing any
local minima, it could also mean that we’ll be taking a very long time to reach the bottom, if we ever
do. On the other hand, a high learning rate might allow us to converge quickly, but there's a risk of
overshooting the lowest point if the step size is too large.

Minimize a loss function L(w) with respect to the weight(w) . The update rule for using gradient
descent is:

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \nabla L(w_{\text{old}})$$

## where:<br/>

$$w_{\text{new}} is the current weight,
is the updated weight,
is the learning rate, and
is the gradient of the loss function with respect to at the old weight.

Let's generate a simple quadratic loss function = , where the minimum loss is achieved
when (w = 4), and simulate gradient descent with different learning rates to see how it impacts the
convergence towards the minimum loss.



