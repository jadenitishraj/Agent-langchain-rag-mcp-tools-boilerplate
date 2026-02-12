# Study Material: Linear Regression with PyTorch

## Architecture Overview

Linear Regression is the "Hello World" of Deep Learning. While often treated as a simple statistical method, in the context of PyTorch, it represents a **Single-Layer Neural Network** with no activation function.
Mathematically, the model attempts to fit a hyperplane to the data:
$$Y = XW^T + b$$
where $X$ is the input features, $W$ is the weight matrix, and $b$ is the bias vector.
The learning process involves minimizing a cost function (typically **Mean Squared Error**) using an iterative optimization algorithm called **Gradient Descent**.
This notebook progresses from a "Manual" implementation—explicitly defining parameter tensors and computing gradients via matrix derivatives—to a "PyTorch Style" implementation using high-level abstractions:

- `torch.nn.Linear`: A predefined layer that manages weights and biases.
- `torch.utils.data.DataLoader`: A utility for automatic mini-batching and shuffling.
- `torch.optim.SGD`: A pre-built optimizer that handles the weight update logic.

## Exhaustive Q&A

### 1. Mathematically, how does a Linear Regression model relate to a single fully-connected neuron?

A Linear Regression model with $N$ input features and 1 output is mathematically identical to a single artificial neuron with a **Linear Activation Function** (or Identity activation).
The neuron computes the weighted sum of inputs plus a bias ($z = \sum w_i x_i + b$).
In Linear Regression, this $z$ is the final prediction ($y_{pred} = z$).
In a Multi-Layer Perceptron (MLP), this $z$ would be passed through a non-linearity (like ReLU). Thus, Linear Regression is effectively a "neural network" with 1 layer and rank-1 non-linearity. This implies that no matter how many neurons you put in a hidden layer, if they don't have non-linear activations, the entire deep network collapses mathematically into a single Linear Regression model due to the associativity of matrix multiplication ($W_2(W_1x) = (W_2W_1)x = W_{new} x$).

### 2. Why do we initialize weights randomly instead of setting them all to zero?

In Linear Regression, initializing weights to zero is actually "safe" (unlike in Multi-Layer Perceptrons) because the gradients depend on the Input $X$ ($dL/dW \propto X$). Since different features in $X$ have different values, the gradients for different weights will be different, allowing them to diverge and learn.
**However**, random initialization is the standard practice because it breaks symmetry in deeper networks. If neurons in a hidden layer all start at 0, they all receive the same gradient and update identically, acting as a single neuron. PyTorch's `nn.Linear` uses Kaiming or Xavier initialization (depending on version) by default to ensure the signal variance is preserved through the layer, preventing vanishing/exploding gradients at the very start of training.

### 3. Explain the dimensionality of the Weight Matrix `W` and Bias Vector `b` for input size $D_{in}$ and output size $D_{out}$.

In PyTorch, `nn.Linear` stores weights in the shape `[out_features, in_features]`.

- Input $X$: Shape `[Batch_Size, D_in]`
- Weight $W$: Shape `[D_out, D_in]`
- Bias $b$: Shape `[D_out]`
  The linear operation is $Y = X W^T + b$.
  Note the transpose $W^T$. If $W$ were stored as `[D_in, D_out]`, we would do $X W$. PyTorch stores it effectively "transposed" to optimize memory layout for the back-end linear algebra libraries (BLAS/CuBLAS) which often prefer row-major access for the output dimension.
  If you manually initialize $W$ as `randn(D_in, D_out)`, you must perform `x @ w`. If you use `nn.Linear`, the internal weight is `(D_out, D_in)` and it performs `x @ w.t()`.

### 4. What is the derivative of the Mean Squared Error (MSE) loss w.r.t our prediction $y_{pred}$?

MSE Loss: $L = \frac{1}{N} \sum (y_{pred} - y_{true})^2$
Derivative w.r.t. prediction:
$$\frac{dL}{dy_{pred}} = \frac{2}{N} (y_{pred} - y_{true})$$
This derivative is intuitively satisfying: the "direction" we need to move is simply the **Residual** (the difference between what we guessed and the truth). If we guessed too high, the term is positive, so the negative gradient pulls us down. The magnitude is proportional to the error—large errors result in large updates, small errors result in fine-tuning. The factor $\frac{2}{N}$ is just a scaling constant that is absorbed into the Learning Rate during optimization.

### 5. Why is "Gradient Descent" preferred over the "Normal Equation" (Analytical Solution) for finding Linear Regression parameters?

The Normal Equation for Linear Regression is $\theta = (X^T X)^{-1} X^T y$. This gives the perfect global minimum instantly (for convex MSE).
However, it requires computing the **Inverse** of the matrix $(X^T X)$, which is an $O(D^3)$ operation where $D$ is the number of features.

1.  **Computational Cost**: If you have 100,000 features, inverting a $100k \times 100k$ matrix is computationally impossible. Gradient Descent scales with $O(N \cdot D)$, which is much faster for large $D$.
2.  **Memory**: The matrix might not fit in RAM. Gradient Descent works with mini-batches.
3.  **Generalization**: The analytical solution tries to fit the training noise perfectly (overfitting). Iterative Gradient Descent acts as a regularizer (Early Stopping), often producing models that generalize better to unseen data.

### 6. Explain the concept of "Learning Rate" and its impact on convergence.

The Learning Rate ($\alpha$) controls the size of the step we take downhill.
New Weight = Old Weight $- \alpha \cdot Gradient$.

- **Too Small**: The model learns excruciatingly slowly. It might get stuck in a local minimum (in non-convex landscapes) or simply take days to reach the solution.
- **Too Large**: The model "Overshoots" the minimum. It might bounce back and forth across the valley, failing to settle. In extreme cases, the steps increase the energy, causing the loss to explode to Infinity (Divergence).
- **Just Right**: The model steadily descends. Modern optimizers (Adam, Scheduler) adapt the learning rate: starting large for fast traversal and decaying it over time to settle precisely into the minimum.

### 7. What is the difference between "Batch Gradient Descent", "Stochastic Gradient Descent" (SGD), and "Mini-batch SGD"?

1.  **Batch GD**: Computes gradient using the **Entire Dataset**. It gives a stable, accurate gradient but is slow and memory-heavy.
2.  **Stochastic GD**: Computes gradient using a **Single Example**. Highly divergent, noisy path, but very fast updates. The noise can help escape local minima.
3.  **Mini-batch SGD**: Computes gradient using a small chunk (e.g., 64 samples). This is the best of both worlds. It vectorizes the computation (GPU friendly), provides a reasonably stable gradient estimate, and fits in memory. In PyTorch, `DataLoader` facilitates Mini-batch SGD, which is the standard industry practice.

### 8. Describe the role of `torch.no_grad()` in the manual weight update step.

When we manipulate weights manually (`w -= w.grad * lr`), this involves an arithmetic operation (subtraction) on a tensor.
If we didn't wrap this in `torch.no_grad()`, Autograd would track this subtraction and add it to the computational graph.

1.  **Memory Leak**: The graph history would grow indefinitely with every epoch.
2.  **Incorrect Gradients**: PyTorch would try to backpropagate through the _weight update itself_ in the next step, which is mathematically wrong for standard optimization.
    By using `width torch.no_grad():`, we tell PyTorch: "This is an in-place update to the variable itself, not a mathematical operation that is part of the model's prediction logic."

### 9. Why do we need to specify `requires_grad=True` for weights?

`requires_grad=True` signals the Autograd engine to allocate memory to store gradients for this tensor.
Standard data tensors (Inputs/Labels) have `requires_grad=False` by default because we don't want to update the data; we only want to update the parameters.
If you forget this flag, `w.grad` will be `None` after `loss.backward()`, and your optimizer will throw an error because it has no gradient information to work with. It essentially marks the tensor as a "Learnable Parameter" within the dynamic graph.

### 10. How does `torch.nn.Linear` simplify model creation compared to manual matrix multiplication?

Manual:

1.  Initialize `w` and `b` tensors (remembering shapes and `requires_grad`).
2.  Write `def model(x): return x @ w.t() + b`.
3.  Register these parameters with the optimizer manually.
    `nn.Linear(in, out)`:
4.  Automatically creates `weight` and `bias` parameters with correct shapes and optimized initialization schemes (Kaiming/He).
5.  Packages them into a Module that can be passed to an optimizer cleanly (`optimizer(model.parameters())`).
6.  Implements the forward pass using optimized C++ backend calls (`F.linear`), which handles edge cases and broadcasting more robustly than raw Python operators.

### 11. What is a `TensorDataset` and why is it useful?

`TensorDataset` is a wrapper class that wraps input tensors and target tensors into a single dataset object.
It implements `__getitem__` and `__len__`.
`__getitem__(index)` returns the tuple `(input[index], target[index])`.
This allows it to be passed to a `DataLoader`. Without it, you would have to manually slice your input and target arrays inside your training loop to create batches. `TensorDataset` standardizes the interface between your raw data tensors and the iterating logic of the DataLoader.

### 12. Explain the function of `DataLoader` specifically the `shuffle=True` argument.

`DataLoader` handles the complexity of:

1.  Batching: Slicing the dataset into chunks of size `batch_size`.
2.  Shuffling: Randomizing the order of samples.
3.  Parallel Loading: Using multiprocessing (`num_workers`) to load data.
    `shuffle=True` is critical during **Training**. If you don't shuffle, the model sees data in the same order every epoch.

- **Bias**: If the data is sorted by class (all Cats then all Dogs), the model will oscillate wildly, learning "All Cats" then forgetting it to learn "All Dogs."
- **Cyclical gradients**: Even if mixed, fixed patterns can lead to cyclical gradient updates that hinder convergence. Shuffling (IID assumption) ensures the gradient estimate is unbiased and diverse at every step.

### 13. What is the difference between `nn.MSELoss` and `nn.L1Loss` (MAE)?

- **MSE (L2 Loss)**: Squaring the error ($e^2$). It penalizes large errors heavily (quadratic). It is sensitive to **Outliers**. If one sample is wrong by 10, the gradient is proportional to 20.
- **L1 Loss (MAE)**: Absolute value ($|e|$). It penalizes all errors linearly. If a sample is wrong by 10, the gradient is constant (1 or -1). It is **Robust to Outliers**.
  For Linear Regression on clean data, MSE is preferred because it is differentiable everywhere and converges faster to the mean. If the data has crazy outliers (e.g., sensor glitches), L1 might be better because the model won't be "pulled" as drastically toward the outlier.

### 14. Why is "Feature Scaling" (Normalization) important for Linear Regression?

If Feature A (e.g., "House Area") ranges from 0-5000 and Feature B (e.g., "Bedrooms") ranges from 1-5.
The gradients for Feature A will be massive compared to Feature B. The loss landscape becomes an elongated "ellipse" or "taco" shape rather than a nice circle.
Gradient descent struggles on elongated surfaces: it oscillates back and forth across the steep dimension while moving slowly along the shallow dimension.
Scaling (e.g., Standardizing to mean 0, std 1) makes the loss landscape spherical. The gradient points directly towards the minimum, allowing for a larger learning rate and much faster convergence.

### 15. Interpret the weights of a trained Linear Regression model.

In Linear Regression, the weights are directly interpretable.
If $y = w_1 x_1 + w_2 x_2 + b$.
$w_1$ represents the change in $y$ for a one-unit increase in $x_1$, assuming $x_2$ is held constant.

- Positive $w$: Positive correlation.
- Negative $w$: Negative correlation.
- Magnitude $|w|$: Importance (assuming features are scaled!).
  This interpretability is why Linear Regression remains the dominant model in fields like Finance and Medicine, where explaining "Why" the prediction was made is as important as the prediction itself.

### 16. What is "Overfitting" in the context of Linear Regression, and how can we detect it?

Overfitting in Linear models usually happens when you have **more features than samples** ($D > N$) or highly correlated features (Multicollinearity). The model learns to fit the noise in the training data perfectly but fails on new data.
Detection:

- **Training Loss** is very low.
- **Validation Loss** is high.
- **Weights** are typically huge (e.g., one weight is 1,000,000 and another is -999,999 to cancel it out).
  We mitigate this using **Regularization** (L2 Ridge or L1 Lasso), which penalizes large weights, forcing the model to learn smoother, simpler patterns.

### 17. How does the `optimizer.step()` method work internally?

`optimizer.step()` iterates over all the parameter groups registered with the optimizer.
For standard SGD:
For each parameter `p`:

1.  Check if `p.grad` exists.
2.  If Momentum is enabled, update the velocity buffer.
3.  Compute the update: `delta = learning_rate * p.grad`.
4.  Apply update: `p.data.add_(-delta)`.
    This abstraction separates the "Learning Logic" (Adam, SGD, RMSProp) from the "Model Architecture." You can swap optimizers by changing one line of code, without rewriting the update loop math manually.

### 18. Why is the Validation Set crucial, even for a simple Linear Regression?

Even with a simple model, we need to know how well it generalizes.
If we test on the Training Set, we are just measuring "Memorization." A validation set (held-out data) estimates "Generalization Performance."
It is also used for **Hyperparameter Tuning**. We might try different Learning Rates or Batch Sizes. We pick the one that performs best on the Validation Set. If we picked based on Training set, we'd just pick the model that overfits the most.
Finally, for Linear Regression, it helps detect **Data Drift**. If validation loss is consistently worse than training loss even with a simple model, it suggests the validation distribution is fundamentally different from the training distribution.

### 19. What happens if the Learning Rate is too high?

If $\alpha$ is too high, the update step takes the weights from one side of the valley to the other, potentially landing _higher_ up the slope than before.
Next step, the gradient is even larger (steeper slope), so the step is even larger.
The weights diverge towards infinity. In PyTorch, you will see the Loss turn to `NaN` (Not a Number) very quickly because floating point values exceed their maximum capacity ($> 10^{38}$). This is a signature sign of a bad Learning Rate configuration.

### 20. Can Linear Regression solve non-linear problems?

Natively, no. It fits a straight line/plane.
However, through **Feature Engineering**, it can.
We can create "Polynomial Features": inputs $x, x^2, x^3$.
The model is still linear w.r.t. the weights ($y = w_1 x + w_2 x^2 + b$), but non-linear w.r.t. the input $x$. This is called "Polynomial Regression" but is solved using the exact same Linear Regression architecture.
This demonstrates a core ML concept: A linear model in a high-dimensional engineered feature space can represent complex non-linear boundaries in the original lower-dimensional space.

### 21. Describe the difference between `model.train()` and `model.eval()` modes.

For a pure Linear Regression (`nn.Linear`), these modes do nothing.
However, in general PyTorch practice, it is mandatory to switch.

- `model.train()`: Enables layers like Dropout and BatchNorm (updates running stats).
- `model.eval()`: Disables Dropout, locks BatchNorm to use saved stats.
  Even if your simple model doesn't use these layers now, getting into the habit prevents catastrophic bugs later when you add a BatchNorm layer and wonder why your validation accuracy is garbage (because you forgot `model.eval()`).

### 22. What is the role of the bias term 'b'? Why can't we just use 'Wx'?

The bias term acts as an **Intercept**. It allows the hyperplane to not pass through the origin $(0,0)$.
If you have data where $x=0$ implies $y=50$ (e.g., base temperature), a model $y=wx$ forces $y$ to be 0 when $x$ is 0. It biases the line to pivot continuously around the origin, likely resulting in a terrible fit.
The bias provides an "Offset" that shifts the activation function left or right. In high dimensions, it allows the decision boundary to be placed anywhere in the space, not just tethered to the center.

### 23. Explain "Huber Loss" and when it is preferable to MSE.

Huber Loss (Smooth L1) combines the best of MSE and L1.

- For small errors ($|e| < \delta$): It is quadratic ($e^2$). Differentiable and smooth near 0 (unlike L1 which has a kink).
- For large errors ($|e| > \delta$): It is linear ($|e|$). Robust to outliers (unlike MSE which explodes).
  It effectively clips the gradient of the loss preventing it from becoming too large when the prediction is very wrong. This is the default regression loss in many robust systems (like Object Detection bounding box regression) where bad training examples exist but shouldn't destroy the training stability.

### 24. How does `torch.optim.SGD` implement "Momentum"?

Standard SGD can oscillate in ravines (steep in one direction, shallow in another).
Momentum adds a "Velocity" term.
$v_{t} = \gamma v_{t-1} + \eta \nabla J(\theta)$
$\theta = \theta - v_{t}$
It accumulates previous gradients. If the gradient keeps pointing in the same direction, velocity increases (accelerates), speeding up convergence. If gradients oscillate (zig-zag), they cancel each other out in the velocity term, smoothing the path.
This physical analogy (a heavy ball rolling down a hill) helps the optimizer power through flat local minima and dampen oscillations.

### 25. What is the "Computational Graph" for a Linear Regression model?

Graph Nodes: Inputs ($x$), Weights ($W$), Bias ($b$).
Operation 1: `MatMul` ($x, W^T$) -> Output $A$.
Operation 2: `Add` ($A, b$) -> Output $Y_{pred}$.
Operation 3: `Sub` ($Y_{pred}, Y_{true}$) -> Diff $D$.
Operation 4: `Pow` ($D, 2$) -> Output $S$.
Operation 5: `Mean` ($S$) -> Output Loss $L$.
Autograd builds this graph dynamically. When `L.backward()` is called, it traverses 5 -> 4 -> 3 -> 2 -> 1, computing partial derivatives at each step to populate `W.grad` and `b.grad`.

### 26. Why do we divide the accumulated loss by `diff.numel()` in the manual MSE implementation?

`torch.sum(diff * diff)` computes the **Total Squared Error** (SSE).
This value depends on the Batch Size. If you double the batch size, SSE doubles.
This means your Gradient also doubles. You would have to constantly adjust your Learning Rate based on your Batch Size.
By dividing by `numel` (taking the Mean), the Loss acts as an "Average error per sample." The scale of the gradient remains roughly constant regardless of whether you process 1 sample or 1 million samples. This decouples the Hyperparameters (Learning Rate) from the Batch Size configuration.

### 27. What is "Convexity" and why is it important for Linear Regression?

The Loss function of Linear Regression (MSE) is **Convex**. It looks like a perfect bowl.
Properties:

1.  It has exactly one Global Minimum.
2.  It has no Local Minima.
3.  Every tangent line lies below the curve.
    This guarantees that Gradient Descent will **Always** converge to the global optimal solution (given a proper learning rate). You don't need to worry about initialization or getting stuck in a bad spot. This is a unique luxury; Deep Neural Networks are non-convex and have many local minima (though they are usually "good enough").

### 28. How does `inputs @ w.t()` work dimensionally?

Inputs: $[N, F]$ (N samples, F features).
Weights: $[O, F]$ (O outputs, F input features).
`w.t()`: Transposes weights to $[F, O]$.
Multiplication: $[N, F] \times [F, O] = [N, O]$.
Result: A matrix where each row corresponds to a sample and each column corresponds to an output neuron.
This is the fundamental operation of all Dense layers. The dimensionality check (matching inner dimensions $F$) is the first step in debugging any shape mismatch error.

### 29. What is the impact of "Multicollinearity" on training stability?

Multicollinearity occurs when two features are highly correlated (e.g., "Weight in Kg" and "Weight in Lbs").
The information is redundant. This makes the matrix $X^T X$ close to singular (non-invertible) or "Ill-conditioned".
In the Loss Landscape, this creates a "valley" that is perfectly flat along the bottom. Infinite combinations of weights ($w_1=2, w_2=0$ vs $w_1=0, w_2=2.2$) yield the same prediction.
While SGD can handle this better than matrix inversion, it causes the weights to drift randomly or become excessively large, making the model unstable and hard to interpret. L2 Regularization solves this by forcing the solution to pick the smaller weights.

### 30. Why is `optimizer.zero_grad()` often placed inside the training loop?

It is placed _before_ `loss.backward()`.
PyTorch accumulates gradients. If you don't zero them, the gradient for epoch 2 will be $Grad_{epoch1} + Grad_{epoch2}$.
The update would be: $W = W - \alpha (G_1 + G_2)$.
Effectively, the momentum of previous epochs would exist purely due to an implementation detail, not mathematical intent. This essentially increases the effective learning rate and step size constantly, leading to divergence.
The standard pattern is: `zero` -> `forward` -> `backward` -> `step`.

### 31. What is the difference between Parameters and Hyperparameters in this context?

- **Parameters**: Variables learned by the model from data ($W$, $b$). Internal to the model.
- **Hyperparameters**: Configuration variables set by the human before training (Learning Rate, Batch Size, Number of Epochs, Optimizer choice). External to the model.
  The "Learning" process finds parameters. The "Tuning" process finds hyperparameters. There is no gradient for hyperparameters; we find them via Grid Search, Random Search, or intuition.

### 32. Explain "Early Stopping" as a regularization technique.

We track the Validation Loss every epoch.
Initially, both Training and Validation loss decrease.
Eventually, Training loss keeps dropping, but Validation loss flattens or starts increasing. This is the inflection point where **Overfitting** begins (learning noise).
Early Stopping means monitoring this metric and terminating training immediately when Validation loss stops improving for $K$ epochs ("patience"). It explicitly prevents the model from minimizing training loss at the expense of generalization.

### 33. How does PyTorch handle floating point precision in gradients?

Gradients are often very small numbers ($10^{-5}$). Repeatedly multiplying small numbers (backprop) leads to Underflow (becoming pure zero).
PyTorch uses `float32` by default which has ~7 decimal digits of precision.
For very deep networks or small learning rates, this isn't enough.
**Mixed Precision** (AMP) uses dynamic loss scaling to keep gradients in the representable range of `float16`.
For standard Linear Regression, `float32` is sufficient. `float64` is rarely used unless debugging numerical instability or doing precise scientific modeling.

### 34. What is the "Identity Function" in the context of activation?

Linear Regression can be viewed as $Activation(Linear(x))$.
Here, $Activation(z) = z$. This is the Identity function.
Derivative: $f'(z) = 1$.
Because the derivative is constant (1), gradients flow perfectly through this layer without diminishing or exploding (unlike Sigmoid). This is why linear layers are efficient but cannot stack to correct complex patterns—the "Identity" nature preserves the linearity of the transformation.

### 35. Can we use GPU for Linear Regression? Is it worth it?

Yes, just move model and data: `model.to('cuda')`, `data.to('cuda')`.
Is it worth it?
For small datasets (like the tutorial's 15 rows), **No**. The overhead of transferring data to GPU (PCIe latency) and launching the CUDA kernel exceeds the computation time. CPU is faster for tiny matrices.
For large datasets (1M rows, 1000 features), **Yes**. The massive parallelism of the GPU matrix multiplication core will outperform the CPU by orders of magnitude. The "breakeven point" typically happens at matrix sizes around 500x500 or 1000x1000.

### 36. What is "Weight Decay" in the generic sense of the SGD optimizer?

User passes `weight_decay=1e-5` to `optim.SGD`.
This performs **L2 Regularization**.
Conceptually, it modifies the loss: $Loss_{total} = Loss_{MSE} + \frac{\lambda}{2} ||W||^2$.
Gradient update rule changes: $w \leftarrow w - \alpha (\frac{dL}{dw} + \lambda w)$.
Or: $w \leftarrow (1 - \alpha\lambda)w - \alpha \frac{dL}{dw}$.
Before applying the gradient, we shrink the weight slightly (multiply by $<1$). This constantly pushes "useless" weights towards zero, keeping the model simple and reducing overfitting.

### 37. Describe the "Data Leakage" risk in Linear Regression preprocessing.

If you Normalize your data (subtract mean, divide by std), you must calculate mean/std using **Only the Training Set**.
If you use the whole dataset (Train + Test) to calculate mean/std, information about the Test distribution "leaks" into the Training set.
The model learns a bias based on what it will see in the future. This leads to overly optimistic testing results that fail in production.
Always: `scaler.fit(X_train)`, `scaler.transform(X_train)`, `scaler.transform(X_test)`.

### 38. How does `torch.utils.data.random_split` helper function work?

It splits a dataset into non-overlapping new datasets of given lengths.
Crucial for creating a Validation set.
`train_ds, val_ds = random_split(dataset, [10000, 2000])`.
It randomizes indices correctly. One gotcha: If strict reproducibility is needed, you must set the manual seed generator passed to the function, otherwise your train/val split changes every run, making hyperparameter comparison invalid.

### 39. What is a "Parameter Group" in PyTorch optimizers?

Optimizers take a list of parameters. But they can also take a list of dicts: `[{'params': layer1}, {'params': layer2, 'lr': 1e-3}]`.
This allows **Differential Learning Rates**.
You can train the weights of the Linear layer with a small LR and the Bias with a large LR, or (more commonly) fine-tune a pre-trained feature extractor with a tiny LR while training the new Linear classifier head with a normal LR. This granular control is essential for Transfer Learning.

### 40. Explain the "Chain Rule" application in the Linear Regression Backward pass.

Forward: $z = Input \cdot Weight$. Loss $L = (z - y)^2$.
We want $dL/dW$.
Chain Rule: $\frac{dL}{dW} = \frac{dL}{dz} \cdot \frac{dz}{dW}$.

1. $\frac{dL}{dz} = 2(z - y)$ (Gradient of Loss w.r.t Output).
2. $\frac{dz}{dW} = Input$ (Gradient of Linear func w.r.t Weight).
   Result: $\frac{dL}{dW} = 2(z - y) \cdot Input$.
   Autograd handles this composition automatically. If we added an activation $a = \sigma(z)$, the chain rule simply extends: $\frac{dL}{da} \cdot \frac{da}{dz} \cdot \frac{dz}{dW}$.

### 41. Why does PyTorch use `float32` by default instead of `float64`?

Double precision `float64` uses 64 bits (8 bytes) per number. `float32` uses 4 bytes.

1.  **Memory**: `float64` halves the max batch size you can fit in VRAM.
2.  **Bandwidth**: Moves twice as much data across memory bus.
3.  **Compute**: FP32 ALUs are more common and faster on GPUs. FP64 units are often physically rare on consumer GPUs (GeForce), artificially capped at 1/32 speed to upsell professional (Quadro/Tesla) cards.
    Since ML requires stochastic noise anyway, the extra precision of FP64 doesn't usually improve accuracy, making FP32 the optimal engineering choice.

### 42. What is "Model Serialization" in the context of this simple model?

Saving the trained model to disk.
`torch.save(model.state_dict(), 'linear.pth')`.
The `state_dict` is just a Python dictionary: `{'weight': tensor([[...]]), 'bias': tensor([...])}`.
It does not save the class definition `nn.Linear`.
To load:

1.  Instantiate `model = nn.Linear(...)`.
2.  `model.load_state_dict(torch.load('linear.pth'))`.
    This ensures you are loading weights into the correct architecture.

### 43. Explain "Residual Analysis" for model validation.

After training, plot the Residuals ($y_{true} - y_{pred}$) vs $y_{pred}$.
Ideal: Random scatter around 0.
Bad patterns:

- **Funnel Shape**: Heteroscedasticity (Error variance increases with value).
- **Curve Shape**: Creating specific errors at specific ranges. Means the model is missing a non-linear term (e.g., trying to fit a parabola with a straight line).
  Residual analysis tells you _how_ your model is failing, often guiding Feature Engineering (e.g., "Add a log transform to the target").

### 44. What is the minimal PyTorch code to implement Linear Regression?

```python
model = nn.Linear(1, 1)
opt = optim.SGD(model.parameters(), lr=0.1)
for x, y in data:
    loss = ((model(x) - y)**2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
```

This 7-line snippet encapsulates the entire learning workflow. The power of PyTorch is compressing the math into these readable, semantic verbs (backward, step, zero_grad).

### 45. Why is `.t()` used in the manual model implementation?

Math convention: $y = Wx + b$ (where $x$ is column vector).
Data convention: $X$ is row-major matrix (Samples as rows).
So we do $Y = X W^T$.
In the manual implementation, `w` was initialized as `(2, 3)` (Out, In).
`inputs` is `(5, 3)` (Samples, In).
We need to multiply Input features by Weights.
`inputs @ w` fails (3 vs 2).
`inputs @ w.t()` aligns the inner dimension (3) so the dot product happens over the features.

### 46. What is the difference between `backward()` on a scalar vs a tensor?

`loss.backward()` works because Loss is a Scalar (Rank-0).
If you output a vector `y` and call `y.backward()`, PyTorch demands an argument: `gradient` (vector of same shape).
This is because a vector output w.r.t vector input produces a Jacobian Matrix. Autograd computes Vector-Jacobian Product.
It assumes you are starting the chain rule from a weighted sum of the outputs.
`loss.backward()` is syntactic sugar for `loss.backward(torch.tensor(1.0))`.

### 47. How does "StandardScaler" (Z-score normalization) affect the Bias term?

If you center inputs ($X - \mu$), the "Cloud" of data moves to the origin.
Before scaling: Bias had to project far out to intercept the cloud.
After scaling: Bias should be close to the Mean of the target $Y$.
If you also scale $Y$, Bias should be close to 0.
This drastically simplifies the search space for the Bias parameter. Without scaling, the Bias might need to traverse from 0 to 10,000, while weights are small. This scale mismatch hinders optimization.

### 48. What is the "Epoch" vs "Iteration" distinction?

- **Epoch**: One complete pass through the entire Training Dataset.
- **Iteration (Step)**: One update step using a single Batch.
  If Dataset = 1000, Batch Size = 100.
  1 Epoch = 10 Iterations.
  Model weights update 10 times per epoch.
  "Training for 100 epochs" means the model sees each example 100 times, but updates weights 1000 times total.

### 49. Describe the usage of `model.parameters()` generator.

`for param in model.parameters(): print(param.shape)`
It recursively yields all learnable tensors (weights + biases) from the module and all sub-modules.
Crucial for:

1.  Passing to Optimizer: `SGD(model.parameters())`.
2.  Counting parameters: `sum(p.numel() for p in model.parameters())`.
3.  Clipping gradients: `clip_grad_norm_(model.parameters())`.
    It abstracts away the internal hierarchy of the network layers.

### 50. What is "Gradient Accumulation" conceptually?

If your GPU can only fit batch size 16 (`batch_size=16`), but you want the stability of batch size 64.
You run 4 forward/backward passes.
You do NOT call `opt.step()` or `zero_grad()` between them.
PyTorch **adds** the gradients to `.grad`.
After 4 loops, `.grad` contains the sum of gradients from 64 samples.
Then you call `opt.step()`.
This simulates a larger batch size perfectly (except for BatchNorm statistics updates).

### 51. Why is `torch.randn` used for weight init but `torch.zeros` or `rand` is bad?

- `zeros`: Symmetry problem (in MLPs).
- `rand` (Uniform 0-1): All weights positive. In the backward pass, all gradients w.r.t weights will have the same sign as the input gradient (zigzagging updates). Also, mean is 0.5, shifting activations to positive regime.
- `randn` (Normal 0,1): Mean is 0. Weights are positive and negative. Allows outputs to be centered around 0. This is closer to the "Ideal" initialization for keeping signal variance stable.

### 52. How does `register_backward_hook` help in debugging gradients?

You can attach a function to a layer that executes during the backward pass.
`layer.register_backward_hook(print_grad)`
It receives `(module, grad_input, grad_output)`.
You can check "Is the gradient vanishing (becoming 0) at this layer?" or "Is it exploding (NaN)?".
This gives visibility into the "Black Box" of backpropagation, helping diagnose why a Linear Regression might be diverging.

### 53. What is the difference between `F.linear(x, w, b)` and `x @ w.t() + b`?

Mathematically: Identical.
Implementation:
`F.linear` calls the underlying C++ "addmm" (Add-Matrix-Multiply) function directly if possible, which is a fused kernel.
`x @ w.t() + b` might launch two separate kernels (MatMul, then Add).
Fused kernels are faster (less memory bandwidth overhead). Always prefer the functional API (`F.linear`) or Module API (`nn.Linear`) over raw arithmetic for performance.

### 54. Explain "Kaiming Initialization" (He Init) vs "Xavier Initialization" (Glorot).

Linear layers default to a specialized init (often Kaiming Uniform).

- **Xavier**: Keeps variance constant for Tanh/Sigmoid activations. Scales by $\frac{1}{\sqrt{n_{in} + n_{out}}}$.
- **Kaiming**: Keeps variance constant for ReLU activations. Scales by $\sqrt{\frac{2}{n_{in}}}$.
  Even though Linear Regression has no activation, PyTorch assumes you might put one after. Using these schemes is statistically safer than standard `randn` (std=1) which would cause variance to explode in value as it passes through matrix multiplication with many inputs.

### 55. What is the role of `torch.set_grad_enabled(False)`?

It is the function underlying the `with torch.no_grad():` context manager.
It globally disables the autograd engine's tracking.
Useful for writing a generic `evaluate()` function:

```python
def evaluate(model, loader):
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    # ... logic ...
    torch.set_grad_enabled(prev)
```

This ensures that evaluation code never leaks memory, regardless of where it is called.

### 56. Can a Linear Regression model "memorize" the training data?

If $N \le D$ (samples $\le$ features), yes.
A system of linear equations with equal equations and variables has a unique solution that satisfies all of them perfectly (Loss = 0).
The model essentially draws a hyperplane that touches every single point.
This is the definition of **Overfitting**. The model has "memorized" the data geometry rather than learning the statistical trend.

### 57. What is the derivative of $|x|$ (L1 Loss) at $x=0$?

Technically undefined (sharp corner).
Sub-gradient theory allows us to pick any value between [-1, 1].
In practice (PyTorch/Tensorflow), it is usually defined as 0.
This "Sparsity Inducing" property (gradient doesn't vanish as we approach 0, it stays constant -1 or +1) forces weights to become exactly 0, effectively performing Feature Selection. MSE approaches 0 smoothly, so weights get small but rarely hit absolute zero.

### 58. How do we export this PyTorch model to ONNX?

`torch.onnx.export(model, dummy_input, "linear.onnx")`.
PyTorch uses **Tracing**. It runs the `dummy_input` through the graph, records the operations (`MatMul`, `Add`), and writes them to the standard ONNX format.
This allows the trained Linear Regression to be deployed on Edge Devices, Browsers (ONNX.js), or optimized runtimes (TensorRT) which might not have Python installed.

### 59. What happens if input data contains `NaN`?

`NaN` propagates.
$NaN \times W = NaN$.
Loss = $NaN$.
Gradient = $NaN$.
Weights Update = $W - NaN = NaN$.
The entire model becomes corrupted instantly.
PyTorch does not check for NaNs by default (performance penalty). You must sanitize data (`torch.isnan(x).any()`) in your preprocessing pipeline or use `torch.autograd.set_detect_anomaly(True)` to find where the NaNs originate during debugging.

### 60. Final synthesis: Why learn Linear Regression in PyTorch if Scikit-Learn exists?

Scikit-Learn uses analytical solutions (LAPACK/OLS). It is CPU-bound and cannot handle "Big Data" that doesn't fit in RAM.
PyTorch Linear Regression:

1.  **Scalable**: Supports Mini-batch SGD (Infinite dataset size).
2.  **GPU**: Matrix ops run on CUDA.
3.  **Differentiable**: Can be just one piece of a larger system (e.g., the final layer of a CNN, or a control loop in RL).
    Learning it in PyTorch isn't about fitting lines; it's about mastering the "Differentiable Programming" workflow (Tensors $\to$ Autograd $\to$ Optimizer) in a controlled environment before tackling Deep Networks.
