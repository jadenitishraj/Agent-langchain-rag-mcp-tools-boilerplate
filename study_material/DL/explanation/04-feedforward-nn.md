# Study Material: Feedforward Neural Networks & GPU Training

## Architecture Overview

This module introduces the **Multi-Layer Perceptron (MLP)**, also known as a Feedforward Neural Network.
Unlike Logistic Regression, which directly maps inputs to outputs ($Input \to Output$), an MLP introduces **Hidden Layers** ($Input \to Hidden \to Output$).
The specific architecture used here is:

1.  **Input**: 784-dimensional vector (flattened image).
2.  **Hidden Layer**: 32 neurons with **ReLU** activation.
3.  **Output Layer**: 10 neurons (logits for digits).
    The introduction of the hidden layer and non-linear activation allows the model to learn complex, non-linear relationships.
    This module also covers the mechanics of **GPU Training**: moving tensors to VRAM (`.to(device)`), using a `DeviceDataLoader` wrapper, and the significant speedup obtained by parallelizing matrix operations on CUDA cores.

## Exhaustive Q&A

### 1. What is the fundamental difference between Logistic Regression and a Feedforward Neural Network?

Logistic Regression is a linear classifier. It separates classes using straight lines (hyperplanes).
A Feedforward Neural Network (with at least one hidden layer and non-linear activation) is a **Non-Linear Classifier**.
It involves composition of functions: $f_2(f_1(x))$.
This allows it to draw curved, complex decision boundaries (e.g., separating points inside a circle from points outside), which is impossible for a linear model.

### 2. Why is "ReLU" (Rectified Linear Unit) preferred over Sigmoid/Tanh in hidden layers?

Sigmoid ($0 \to 1$) and Tanh ($-1 \to 1$) saturate at the extremes.
If input is large (positive or negative), the gradient is near zero.
In deep networks, multiplying many small gradients causes the **Vanishing Gradient Problem**—early layers stop learning.
ReLU ($max(0, x)$) has a constant gradient of 1 for stable, positive inputs. It does not vanish (for positive values) and is computationally trivial (just a `max` check), speeding up training significantly.

### 3. What does the "Hidden Size" of 32 represent?

It represents the **Capacity** or "Memory" of the intermediate representation.
The model compresses the 784 pixel features into 32 abstract features.
These 32 features might represent strokes (loops, lines, curves) rather than raw pixels.

- **Too Small**: Bottleneck. The model cannot capture enough information to distinguish digits.
- **Too Large**: Overfitting. The model memorizes noise. Computional cost increases quadratically with layer size.

### 4. Explain the Universal Approximation Theorem.

It states that a Feedforward Network with a **single hidden layer** containing a finite number of neurons can approximate **any continuous function** to arbitrary precision, given appropriate activation functions (like ReLU/Sigmoid).
This is the theoretical foundation of Deep Learning: standard MLPs can theoretically solve any solvable problem, provided we have enough data and compute to find the parameters.

### 5. Why do we need to move both the Model and the Data to the GPU?

The GPU has its own dedicated memory (VRAM). The CPU has RAM.
They are physically separate chips connected by the PCIe bus.
A GPU core cannot access CPU RAM directly (efficiently).
To perform matrix multiplication on the GPU, the matrices ($W$, $X$) must reside in VRAM.
If the model is on GPU but data is on CPU, PyTorch will throw a `RuntimeError: Expected object of device type cuda but got device type cpu`.

### 6. What is the `non_blocking=True` argument in `.to(device)`?

Data transfer (Host to Device) is slow.
By default, Python waits for the transfer to finish before proceeding (Blocking).
`non_blocking=True` allows the CPU to continue executing subsequent Python code (e.g., preparing the next batch) while the DMA controller handles the transfer in the background.
This allows **Overlapping** of Compute (GPU) and Data Transfer (PCIe), significantly increasing throughput.

### 7. Describe the `DeviceDataLoader` pattern used in the notebook.

It is a Wrapper class (Decorator pattern).
It takes a standard `DataLoader`.
It intercepts the `__iter__` method.
When you loop through it, it fetches a batch from the CPU DataLoader, moves it to the target device (GPU), and yields it.
Benefit: This decouples the "Data Loading logic" from the "Device Management logic." Your training loop doesn't need to know about `.to(device)`; it just consumes clean, device-ready batches.

### 8. Why do we apply `view(-1, 784)` inside the `forward` method instead of the transform?

If we flatten in the `transform`, the DataLoader yields 1D vectors.
If we visualize the data (matplotlib), we have to un-flatten `view(28, 28)`.
By keeping data as 2D images `[1, 28, 28]` until the very last moment (inside the model), we preserve flexibility.
We can swap the MLP for a CNN (which expects 2D) without changing the data pipeline.
The `forward` method adapts the data to the model's specific needs using `xb.view(xb.size(0), -1)`.

### 9. What is the shape of the Weight Matrix in the hidden layer (`linear1`)?

Input: 784. Hidden: 32.
Shape: `[32, 784]`.
$32$ rows (neurons), $784$ columns (inputs).
Note: The notebook might define `nn.Linear(784, 32)`, but internal storage is transposed `[32, 784]`.

### 10. What is the shape of the Weight Matrix in the output layer (`linear2`)?

Input: 32 (from hidden layer). Output: 10.
Shape: `[10, 32]`.
Each output neuron looks at the 32 abstract features extracted by the previous layer to make a decision.

### 11. Why does the loss decrease faster on GPU compared to CPU for larger nets?

ML involves Matrix Multiplication ($O(N^3)$).
CPU: Few cores (4-64), optimized for serial tasks, branching, low latency.
GPU: Thousands of cores (2000-10000), optimized for parallel SIMD tasks, high throughput.
A matrix multiply allows thousands of independent dot products to happen simultaneously. The GPU crushes this workload.
For a tiny MLP (784->32), CPU might be faster due to transfer overhead, but as size grows, GPU wins exponentially.

### 12. What is "Dead ReLU" problem?

If a neuron learns a large negative bias, $Wx + b$ is always negative.
ReLU output is always 0.
Gradient of ReLU in negative region is 0.
Weights $W$ never update. The neuron "dies" and never activates again.
Solution: Leaky ReLU (small slope in negative region) or careful initialization/learning rates.

### 13. How does `torch.device` abstraction help code portability?

We define `device` once:
`device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')`.
All subsequent code uses `to(device)`.
If you run this on a MacBook (CPU only), it works.
If you run on a Colab (Nvidia Tesla), it uses GPU.
This prevents "hardcoded" cuda calls that crash on different hardware.

### 14. What are "Logits" in the context of this MLP?

The output of `self.linear2`.
They are raw scores.
They can be negative.
They are NOT probabilities.
We pass them to `F.cross_entropy`, which internally applies Softmax.
If we applied Softmax inside the model, we would strictly call them "Probabilities."

### 15. Why don't we use `nn.Sequential` for this model?

We _could_.
`nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))`.
The notebook uses a custom class to demonstrate explicit control flow.
Custom classes allow for: debugging prints in `forward`, skip connections (ResNet), multiple inputs/outputs, or reusing layers. `Sequential` is limited to a linear chain of layers.

### 16. What is the parameter count of this model?

Layer 1 (Weights): $784 \times 32 = 25,088$.
Layer 1 (Bias): $32$.
Layer 2 (Weights): $32 \times 10 = 320$.
Layer 2 (Bias): $10$.
Total: $25,088 + 32 + 320 + 10 = 25,450$.
Even a tiny "Deep" network has significantly more parameters than Logistic Regression ($7850$).

### 17. Why is 32 chosen as hidden size and not 3200?

Balance of Underfitting vs Overfitting.
32 is enough to get ~96% accuracy.
3200 would likely memorize the training set (99.9% train acc) but generalize poorly (95% val acc) unless heavily regularized.
32 is also faster to train on the specific hardware used in tutorial.

### 18. What is the role of `super().__init__()`?

Initializes the `nn.Module` base class.
Sets up the internal `self._modules` dictionary where layers like `linear1` are stored.
Without this, PyTorch's magic methods (`.parameters()`, `.to()`) would fail because the container structure wasn't initialized.

### 19. Can we use multiple hidden layers?

Yes. `784 -> 32 -> 32 -> 10`.
Deeper networks can theoretically represent more complex hierarchical features (Edges -> Shapes -> Digits) with fewer total parameters than a single massive wide layer.
This "Depth Efficiency" is a key driver of modern Deep Learning success.

### 20. How does the `DataLoader` interact with the GPU memory?

Standard `DataLoader` allocates tensors in Host RAM (CPU).
It uses multiprocessing workers.
If `pin_memory=True`, it allocates in Page-Locked RAM.
Then `to(device)` copies to Device VRAM.
The `DeviceDataLoader` we wrote handles the movement step during iteration.

### 21. What happens if we forget to zero gradients?

Gradients accumulate.
Epoch 1 grad: $g_1$.
Epoch 2 grad: $g_1 + g_2$.
The step size effectively grows with every batch.
The training usually diverges (Loss explodes to NaN) very quickly.

### 22. Why do we use `F.relu` functional API instead of `nn.ReLU` layer?

`nn.ReLU()` creates a layer object. It has no learnable parameters.
`F.relu()` is a pure function.
Since ReLU has no state (weights), using the function version in `forward` is slightly more concise and pythonic.
`nn.ReLU` is useful when constructing `nn.Sequential` (which only accepts layer objects, not functions).

### 23. What is the range of values output by ReLU?

$[0, \infty)$.
Negative inputs become 0.
Positive inputs remain unchanged.
This "Rectification" destroys information (negative values are lost), which is actually useful for creating sparse representations where neurons only fire for specific patterns.

### 24. Why is the Validation Accuracy higher than 95% here while Logistic Regression was ~92%?

Non-Linearity.
Digits like "5" and "6" or "8" and "3" have complex geometric differences that aren't linearly separable in pixel space.
The hidden layer transforms the space into a manifold where these classes become linearly separable.
The internal "bending" of the decision boundary solves the cases that confused the linear model.

### 25. Explain "Batch Processing" efficiency on GPU.

Scalar add: $A + B$. 1 cycle.
Vector add (1000 items): Loop 1000 times? No.
GPU essentially does `A[:] + B[:]` in 1 cycle (using 1000 cores).
If batch size is 1, we use 1 core, wasting 999.
If batch size is 1000, we utilize the hardware fully.
This is why increasing batch size (up to memory limits) usually increases training speed (images/second).

### 26. What is the constraint on `hidden_size` choice?

None, technically. It can be any integer.
Conventionally powers of 2 (32, 64, 128, 256).
Memory alignment on GPUs (warps of 32 threads) makes sizes divisible by 32/64 slightly more efficient.

### 27. What is the derivative of ReLU?

$x > 0 \implies f'(x) = 1$.
$x < 0 \implies f'(x) = 0$.
$x = 0 \implies$ Undefined (sub-gradient is usually set to 0 or 0.5 in implementation).
The gradient of 1 is excellent because it does not decay (unlike $0.25$ max gradient of Sigmoid).

### 28. How do we access the GPU if we have multiple GPUs?

`torch.device('cuda:0')` or `torch.device('cuda:1')`.
By default `cuda` means `cuda:0`.
DataParallel or DistributedDataParallel is needed to use multiple GPUs for a single model training.

### 29. What happens if input data is not normalized?

Model might still train, but slower.
Weights will adapt to the scale.
However, if inputs are huge `[0, 255]`, weights need to be tiny.
ReLU doesn't saturate, so it's more robust to scale than Sigmoid.
But large values can lead to large logits, which can cause numerical instability in Softmax calculation.

### 30. How does `yield` work in `DeviceDataLoader`?

It turns the class into a **Generator**.
It produces batches one by one only when requested.
It does not pre-load the entire dataset to GPU (which would crash VRAM).
It loads Batch 1 -> GPU, Yields, Deleted. Then Loads Batch 2.
This streaming approach allows training on datasets larger than VRAM.

### 31. What is "Model Capacity"?

The set of functions the model _can_ represent.
Determined by Depth (layers) and Width (neurons).
Logistic Regression: Low capacity (Linear functions only).
MLP (32 hidden): Medium capacity.
MLP (1024 hidden): High capacity.
Goal: Match model capacity to data complexity.

### 32. Why do we output 10 values, not 1 value (0-9)?

Regression (1 output) implies ordinal relationship ($4$ is "twice as much" as $2$, and close to $5$).
In digits, "4" is not mathematically close to "5" (shapes are totally different).
Classification uses One-Hot encoding because classes are nominal (independent).
We need 10 independent probability scores.

### 33. What is the "Bias" in the hidden layer?

It acts as a threshold for activation.
ReLU activates if $Wx + b > 0 \implies Wx > -b$.
The bias shifts the activation boundary. Without it, the neuron could only split the space with a plane passing through the origin.

### 34. How does `len(train_dl)` change with batch size?

$Len = \lceil TotalSamples / BatchSize \rceil$.
If batch size doubles, `len` halves.
Epoch takes fewer steps (iterations), but each step takes longer computationally.

### 35. What is the "state_dict" of this MLP?

OrderedDict containing:
`linear1.weight`
`linear1.bias`
`linear2.weight`
`linear2.bias`
It maps parameter names to tensors.
This is what is saved to disk (`.pth` file).

### 36. Why is Stochastic Gradient Descent (SGD) valid for non-convex MLP landscapes?

MLP loss surfaces are non-convex (hills and valleys).
Analytical solution is impossible.
SGD finds a _Local Minimum_.
In high-dimensional spaces (Deep Learning), local minima are usually "good enough" (close to global minimum loss).
SGD rarely gets stuck in bad local minima; saddle points are the bigger issue, which Momentum helps escape.

### 37. What is "Overfitting" behavior in loss curves?

Training Loss: Continues to go down (approaches 0).
Validation Loss: Goes down, then flattens, then starts Going UP (U-shape).
The point of divergence is where overfitting starts.
Validation metrics in the notebook show monotonic improvement, suggesting we haven't overfit yet (or model capacity is small enough).

### 38. Can we use `torch.mm` instead of `linear1`?

Yes. `out = torch.mm(xb, w1.t()) + b1`.
`nn.Linear` just wraps this math + parameter management.
Manual implementation is good for learning, `nn.Linear` for production.

### 39. What is the RAM usage of `DeviceDataLoader`?

Low.
It holds a reference to the CPU DataLoader.
It holds one batch on GPU at a time.
It does not duplicate the dataset.

### 40. Why do we need `torch.no_grad()` in evaluation?

To save memory.
Computing `y = model(x)` builds a graph connecting input to output to support backward pass.
This stores intermediate activations (feature maps).
In validation, we never backward pass. `no_grad` prevents graph building, reducing memory usuage by ~50% or more.

### 41. How does the learning rate `0.5` compare to standard defaults?

It is very high.
Standard defaults are often 0.1, 0.01, or 0.001 (for Adam).
For simple datasets (MNIST) and simple optimizers (SGD without momentum), high LR works because the surface is relatively smooth and convex-ish.
For complex CNNs/Transformers, 0.5 would cause immediate divergence.

### 42. What is `F.cross_entropy` doing with the labels?

It expects labels as class indices (LongTensor `[1, 5, 9]`).
It does not expect One-Hot vectors.
It internally selects the logit corresponding to the index.
This "Sparse" label format saves memory (storing 1 integer vs 10 floats per sample).

### 43. Why are "Saddle Points" an issue for MLPs?

In high dimensions, critical points (zero gradient) are rarely local minima.
They are usually Saddle Points (min in one dimension, max in another).
SGD can stall at saddle points.
Noise in SGD (from batches) and Momentum help kick the optimization off the saddle.

### 44. What is `tensor.view()`?

PyTorch's reshape operation.
It does not copy data (if contiguous). It just changes the metadata (strides/shape).
Very fast.
`view(-1)` flattens the tensor.

### 45. Explain "Gradient Descent" updates for Layer 1 weights.

Chain Rule (Backpropagation).
Error accumulates from Output to Layer 2 to ReLU to Layer 1.
$\frac{dL}{dW_1} = \frac{dL}{dY} \cdot \frac{dY}{dH} \cdot \frac{dH}{dZ} \cdot \frac{dZ}{dW_1}$.
The gradient depends on the downstream weights $W_2$ and the derivative of ReLU.
This verifies why dead ReLUs block learning (derivative term becomes 0).

### 46. What is the default data type of PyTorch tensors?

`float32`.
Images loaded via `ToTensor` are float32.
Labels are `int64` (Long).
Weights are float32.

### 47. Why do we import `torch.nn.functional as F`?

Convention.
Modules with state (`Linear`, `Conv2d`) are in `torch.nn`.
Pure functions (`relu`, `cross_entropy`, `softmax`) are in `torch.nn.functional`.
Keeping them separate clarifies code intent.

### 48. What is "Representation Learning"?

The hidden layer learns a new representation of the input.
Layer 1 converts Pixels $\to$ Edges/Shapes.
Layer 2 converts Shapes $\to$ Digits.
Deep Learning is essentially "Learning the best way to represent data such that it becomes linearly separable."

### 49. How to save this model for inference?

`torch.save(model.state_dict(), 'mnist-ffn.pth')`.
To load:
`model = MnistModel(...)`
`model.load_state_dict(torch.load('mnist-ffn.pth'))`
`model.eval()`
Note: You must define the class structure exactly as it was during saving.

### 50. What is "Inference Latency"?

Time taken for `model(x)`.
Matrix multiplication time.
For this small model, latency is microseconds.
Critical for real-time apps.

### 51. Why is PyTorch considered "Pythonic"?

It uses standard Python control flow (`for`, `if`) in `forward`.
It integrates with NumPy.
It uses standard OOP (`class`, `__init__`, `yield`).
Debugging is standard (can use `pdb` breakpoints inside `forward`).
TensorFlow 1.x (Static Graph) was not pythonic; PyTorch changed the industry standard.

### 52. What determines the output range of `linear1`?

Inputs are $[0, 1]$.
Weights are init around $[-0.1, 0.1]$.
Output is roughly sum of 784 small numbers.
Range is roughly $[-10, 10]$ or similar.
ReLU cuts off the negatives.
This range stability is improved by Batch Norm (not used here but relevant).

### 53. What is the impact of removing the hidden layer?

The model becomes `Linear(784, 10)`.
This is exactly Logistic Regression.
Accuracy drops to ~92%.
This proves the Hidden Layer (non-linearity) adds the extra ~4% accuracy value.

### 54. Why do we pass `model.parameters()` to the optimizer?

The optimizer needs to know _what_ to update.
It stores a reference to these tensors.
When `step()` is called, it iterates through this list and subtracts `lr * grad`.
This decouples the Model definition from the Optimization algorithm.

### 55. What is the relationship between VRAM size and Batch Size?

Batch size is limited by VRAM.
Each sample requires storing Activations for every layer.
If batch is too large -> `CUDA Out Of Memory`.
We tune batch size to fill VRAM (maximal parallelism) without crashing.
For this tiny model, batch size could be 50,000+ without issues, but 128 is used for convergence stability.

### 56. Can we use CPU for this training?

Yes.
The code checks `torch.cuda.is_available()`.
Training time will be slower (maybe 2x-5x slower for this size).
For massive Transformers, CPU training is effectively impossible (years vs days).

### 57. What is "Transfer Learning" potential here?

Low for MNIST.
But conceptually: we could take the first layer (Edges), freeze it, and retrain the second layer for a different dataset (e.g., Letters A-Z).
This reusing of "Feature Extractors" is the core of Transfer Learning.

### 58. How do we initialize weights?

`nn.Linear` uses **Kaiming Uniform** by default.
It considers the number of inputs (fan-in) to scale the random variance.
This keeps signal magnitude constant across layers.
Manual initialization `w.normal_(0, 1)` usually fails (signal explodes).

### 59. What is the "Forward Pass"?

Input `x` flows through the graph.
Values are computed.
Loss is computed.
This is the "Inference" direction.

### 60. What is the "Backward Pass"?

Gradients flow from Loss backwards to Inputs.
Calculates "Who is responsible for the error?".
Autograd engine handles this automatically based on the graph constructed during Forward Pass.
This is the "Learning" direction.
