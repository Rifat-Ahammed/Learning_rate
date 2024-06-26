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

**where:** <br/>

- $w_{\text{old}}$ is the current weight,<br/>
- $w_{\text{new}}$is the updated weight,<br/>
- $\alpha$ is the learning rate, and<br/>
- $\nabla L(w_{\text{old}})$ is the gradient of the loss function with respect to at the old weight.<br/>

Let's generate a simple quadratic loss function $L(w) = (w - 4)^2$, where the minimum loss is achieved
when (w = 4), and simulate gradient descent with different learning rates to see how it impacts the
convergence towards the minimum loss.

```Python
# plotting

plt.figure(figsize=(10, 6))

for i, lr in enumerate(learning_rate):
    #print(i)
    plt.plot(paths[i], label=f'LR={lr}')
    
plt.xlabel('Iteration')
plt.ylabel('Weight Value')
plt.title('Weight update paths for DIfferent learning Rates')
plt.legend()
plt.grid(True)
plt.show()
```

![image](https://github.com/Rifat-Ahammed/Learning_rate/assets/96107279/1e7a7dd3-77f8-419e-a711-51a57377e2e8)

The plot above illustrates how the weight updates over iterations for different learning rates
when minimizing the loss function $L(w) = (w - 4)^2$, starting from an initial weight of (w = 0).<br/>

- For a low learning rate (LR=0.01), we see a very gradual approach towards the optimal weight of
  4. This demonstrates how a small learning rate may lead to slow convergence.<br/>
- With a higher learning rate (LR=0.1), the path towards the optimal weight becomes significantly
  faster, showing an improvement in convergence speed without overshooting.<br/>
- As the learning rate increases further (LR=0.5), the updates become more aggressive, and it still
  converges towards the optimal weight but with larger steps. This is closer to the edge of what
  might cause overshooting, but in this case, it effectively reaches near the optimal point.<br/>
- At an even higher learning rate (LR=1.0), the weight update oscillates around the optimal value.
  This illustrates the risk of choosing a too high learning rate, where the model may fail to
  converge properly and instead oscillate around or overshoot the minimum.<br/>

#
To delve deeper into the mathematics of learning rates in the context of gradient descent, let's start
by defining a simple but concrete example. I'll use a quadratic function as our loss function, (L(w)),
and illustrate how the learning rate $\alpha$ influences the gradient descent process mathematically.

# Loss Function

Consider a simple quadratic loss function:<br/>
$L(w) = (w - 4)^2$<br/>

This function has its minimum when $w = 4$, where $L(w) = 0$. Our goal is to adjust $w$ to minimize $L(w)$
using gradient descent. <br/>

# Gradient Descent Update Rule

The update rule in gradient descent adjusts $w$ by moving it in the opposite direction of the gradient
of the loss function $\nabla L(w)$ with respect to $w$ , scaled by the learning rate $\alpha$:

$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \nabla L(w_{\text{old}})$<br/>


# Gradient of Loss Function

The gradient of$L(w)$ with respect to $w$ is:<br/>

$\nabla L(w) = \frac{d}{dw} (w - 4)^2 = 2(w - 4)$ <br/>

This gradient tells us the slope of $L(w)$ at any point $w$, guiding how to adjust $w$ to decrease $L(w)$.

# Applying the Update Rule

$w_{\text{new}} = w_{\text{old}} - \alpha \cdot 2(w_{\text{old}} - 4)$<br/>

This equation will iteratively adjust $w$ towards 4, depending on the value of $\alpha$.

# Effect of Different Learning Rates <br/>

- **Small**  $\alpha$(e.g., 0.01): Makes small updates to $w$, leading to slow convergence towards the
  minimum of $L(w)$. It may require many iterations to get close to $w = 4$.<br/>
- **Moderate** $\alpha$(e.g., 0.1): Balances the rate of convergence and stability. It moves $w$ towards the
  minimum efficiently without risking overshooting.
- **Large** $\alpha$(e.g., 1.0): Makes large updates, which can either quickly converge to the minimum or
  overshoot and diverge, depending on the function and initial conditions.

# Example Calculation<br/>

Let's calculate one step of the update for two different learning rates, starting from $w_{\text{new}} = 0$ :

1. For $\alpha = 0.01$:<br/>
   $w_{\text{new}} = 0 - 0.01 \cdot 2(0 - 4) = 0.08$

2. For $\alpha = 1.0$<br/>
    $w_{\text{new}} = 0 - 1.0 \cdot 2(0 - 4) = 8$

In the first case, $w$ moves slightly towards the optimal value. In the second, $w$ overshoots the
minimum significantly. This illustrates how the choice of learning rate impacts the convergence
behavior in gradient descent.
