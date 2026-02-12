# Study Material: Logistic Regression & Image Classification

## Architecture Overview

In this module, we transition from regression (predicting continuous values) to **Classification** (predicting discrete labels) using the MNIST dataset (handwritten digits using 28x28 grayscale images).
While called "Logistic Regression," in the context of Deep Learning with multiple classes, this is effectively a **Single-Layer Perceptron** followed by a **Softmax** activation.
Architecture:

1.  **Input**: Images are flattened from $1 \times 28 \times 28$ tensors into vectors of size $784$.
2.  **Linear Layer**: A dense matrix multiplication maps $784 \to 10$ (one score for each digit 0-9).
3.  **Activation**: The **Softmax** function converts raw scores (logits) into probabilities summing to 1.
4.  **Loss**: **Cross-Entropy Loss** minimizes the distance between predicted probabilities and the one-hot encoded ground truth.
    This module also introduces the object-oriented design pattern of PyTorch: creating custom models by inheriting from `nn.Module`, defining the `__init__` constructor for layers, and the `forward` method for data flow.

## Exhaustive Q&A

### 1. Why do we need to "Flatten" the input images for a Logistic Regression model?

Logistic Regression (implemented via `nn.Linear`) requires the input to be a 1D feature vector.
An MNIST image is a 2D grid ($28 \times 28$).
To feed it into a matrix multiplication $y = xW^T + b$, we must unroll the 2D spatial structure into a long 1D array of 784 pixels.
Note: Flattening **destroys spatial information**. The model no longer knows that pixel (0,0) is next to pixel (0,1). It treats pixel (0,0) and pixel (27,27) as just two independent features. This is the primary limitation of Dense networks for computer vision, motivating Convolutional Neural Networks (CNNs).

### 2. What is the role of `torchvision.transforms.ToTensor()`?

Raw images are typically loaded as PIL Images (integers 0-255, HWC format).
`ToTensor()` performs three critical operations:

1.  **Converts** the PIL Image/NumPy array to a PyTorch Tensor.
2.  **Reorders** dimensions from HWC (Height, Width, Channels) to CHW (Channels, Height, Width), which is PyTorch's native format.
3.  **Scales** the pixel values from the integer range [0, 255] to the floating-point range [0.0, 1.0].
    This scaling is a form of basic normalization that helps gradient descent converge faster.

### 3. Explain the "Softmax" function and why it is used for Multi-Class Classification.

For $K$ classes, the model outputs $K$ raw scores (logits), which can be any real number ($-\infty$ to $+\infty$).
Softmax transforms these logits $z_i$ into probabilities $p_i$:
$$p_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
Properties:

1.  All $p_i$ are positive (because $e^x > 0$).
2.  $\sum p_i = 1$.
3.  It emphasizes the largest value. If one logit is slightly larger than the others, its probability will be significantly closer to 1 (hence "Soft" max).
    This allows us to interpret the network output as "Confidence" (e.g., "90% sure this is a digit 7").

### 4. What is the difference between `nn.CrossEntropyLoss` and `nn.NLLLoss`?

- `nn.NLLLoss` (Negative Log Likelihood) expects the input to be **Log-Probabilities**. You must apply `LogSoftmax` explicitly in your model.
- `nn.CrossEntropyLoss` expects the input to be **Raw Logits** (scores). It internally applies `LogSoftmax` and then `NLLLoss`.
  Standard practice is to use `CrossEntropyLoss` and return raw logits from the model. This is numerically more stable than applying `Softmax` then `log`, which can lead to `log(0) = -inf` underflow.

### 5. Why is Accuracy a better metric than Loss for classification?

**Loss** (Cross-Entropy) is a continuous, differentiable value used for **Optimization**. It tells us "how wrong" the probability distribution is (e.g., predicting 0.6 for the correct class vs 0.9).
**Accuracy** is a discrete, non-differentiable value used for **Evaluation**. It tells us "how often" the model is correct (e.g., predicting the correct class).
Loss can decrease even if Accuracy stays the same (the model gets more confident in its correct predictions). Loss is for the machine (gradients); Accuracy is for the human (business metric).

### 6. Explain the `argmax` operation used during prediction.

The model outputs a probability vector indices [0..9], e.g., `[0.01, 0.02, 0.9, ...]`.
To convert this to a class label ("It is a 2"), we need the **Index** of the maximum value.
`torch.max(preds, dim=1)` returns two tensors: values (0.9) and indices (2).
We discard the values and keep the indices as our predictions. This discrete step is non-differentiable, which is why we cannot use Accuracy directly as a Loss function for backpropagation.

### 7. What is the purpose of extending `nn.Module` when defining a model?

Extending `nn.Module` allows PyTorch to manage the model's lifecycle automatically.

1.  **Parameter Registration**: Any `nn.Parameter` or sub-module (`nn.Linear`) assigned as a member variable (`self.layer1`) is automatically registered. `model.parameters()` will find it.
2.  **Device Management**: `.to('cuda')` recursively moves all registered sub-modules to GPU.
3.  **State Dict**: `.state_dict()` automatically recurses to save all weights.
4.  **Hooks**: Enables forward/backward hooks for debugging.
    It provides the standard interface (`__call__` wraps `forward`) that integrates with the rest of the ecosystem.

### 8. Why do we define layers in `__init__` and connectivity in `forward`?

`__init__` is for **Initialization** (State). We confirm "what components exist" (e.g., "I have two linear layers"). The weights are allocated here once.
`forward` is for **Computation** (Graph). We define "how data flows" (e.g., "Input goes through layer1, then ReLU, then layer2").
This separation allows for dynamic graphs. We can use Python control flow (loops, ifs) in `forward` to reuse the same layer multiple times (e.g., RNNs) or skip layers (ResNets), while the state (parameters) remains fixed.

### 9. What is the dimension of the Weight Matrix for MNIST Logistic Regression?

Input: 784 pixels.
Output: 10 classes.
PyTorch `nn.Linear(784, 10)` creates a weight matrix of shape `[10, 784]`.
It also creates a bias vector of shape `[10]`.
Total parameters = $(784 \times 10) + 10 = 7850$.
Each row of the weight matrix effectively learns a "Prototype Template" for one digit. The dot product measures how much the input image overlaps with that digit's template.

### 10. Why is the Validation Set size usually smaller than the Training Set?

The Training Set drives the learning; more training data generally equals better performance.
The Validation Set measures performance. It needs to be "statistically significant" enough to represent the distribution, but any data used for validation is data _stolen_ from training.
Typical splits are 80/20 or 90/10. For massive datasets (ImageNet), validation might be just 1-2%. The goal is to maximize the learning signal (Training) while maintaining a reliable trust signal (Validation).

### 11. Explain the "Batch Dimension" in image tensors.

A single image tensor is `[1, 28, 28]` (CHW).
CNNs/Linear layers expect a **Batch** of images: `[N, 1, 28, 28]`.
Even if you are inferencing on a single image, you must `unsqueeze(0)` to add the batch dimension ($N=1$).
This convention allows the underlying linear algebra libraries to optimize operations as Matrix-Matrix multiplications rather than Matrix-Vector, which is significantly faster on hardware.

### 12. What does `images.reshape(-1, 784)` do?

`reshape` changes the tensor shape.
`-1` is a wildcard that means "Infer this dimension from the others."
If input is `[64, 1, 28, 28]` (Batch 64).
We specify `784` as the second dimension.
PyTorch calculates: Total elements = $64 \times 1 \times 28 \times 28 = 50,176$.
First dimension = $50,176 / 784 = 64$.
Result: `[64, 784]`.
This effectively flattens the image dimensions while preserving the batch size. It makes the code robust to changing batch sizes (it works for different N).

### 13. Why is Gradient Descent often slower on CPU for Classification layers?

Classification often involves large output spaces (e.g., 1000 classes for ImageNet or 50k for Language Models).
The final layer performs a matrix multiplication of `[Batch, Hidden] @ [Hidden, Classes]`.
For large classes, this is computationally heavy.
CPUs have limited cores (AVX only does so much). GPUs have thousands of cores designed exactly for massive parallel dot-products. Even for MNIST (10 classes), the overhead of data loading and Python loops on CPU can make it slower than GPU, though the gap is smaller for tiny models.

### 14. What is the "super().**init**()" call in a custom model?

It calls the constructor of the parent class `nn.Module`.
This is **Mandatory**.
`nn.Module.__init__()` initializes the internal storage mechanisms (dictionaries for parameters, buffers, sub-modules, hooks).
If you forget this line, assigning `self.linear = nn.Linear(...)` will not verify parameter registration, and `model.parameters()` will return an empty list, meaning your model won't train.

### 15. How does the `dataset[0]` indexing work for MNIST?

The MNIST dataset class implements `__getitem__`.
When you access index 0, it:

1.  Locates the first image binary data on disk/memory.
2.  Applies the `transform` (ToTensor, Normalization).
3.  Returns a tuple `(image_tensor, label_integer)`.
    It does this "Lazily" (on demand). It effectively converts the data _just-in-time_ when the DataLoader requests it, rather than transforming all 60,000 images in RAM at startup.

### 16. What is the mathematical relationship between "Cross Entropy" and "Kullback-Leibler (KL) Divergence"?

Cross Entropy $H(P, Q) = H(P) + D_{KL}(P || Q)$.
$H(P)$ is the Entropy of the true distribution. Since ground truth is usually a one-hot vector (entropy 0), $H(P) = 0$.
Thus, minimizing Cross Entropy is mathematically equivalent to minimizing the **KL Divergence** between the predicted distribution $Q$ and the true distribution $P$. We are trying to make the model's probability distribution "look like" the real world's distribution.

### 17. How do we interpret the "Bias" vector in the context of digit classification?

The bias vector has shape `[10]`, one value for each digit.
It represents the **Prior Probability** (log-odds) of each class independent of the input image.
If the training set had 90% "Zeros" and 10% others, the bias for "0" would learn to be very high. The model would guess "0" by default even if the input image was black noise.
In a balanced dataset like MNIST, biases usually settle near zero, acting as a threshold adjuster for the activation.

### 18. What is "One-Hot Encoding" of the target labels?

The raw labels are integers (e.g., 5).
The model outputs a vector of 10 probabilities.
To compare them, we conceptually convert 5 into `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`.
Cross-Entropy loss performs this comparison.
It takes the log of the predicted probability at index 5 ($-\log(p_5)$).
All other indices are multiplied by 0 and ignored.
This "One-Hot" selection mechanism focuses the loss entirely on "How much probability did you assign to the _correct_ answer?"

### 19. Why do we not apply an activation function like ReLU in this simple model?

This model is a Linear Classifier. It draws linear decision boundaries (hyperplanes) in the 784-dimensional pixel space to separate digits.
If we added ReLU after the Linear layer but _before_ Softmax, it would zero out negative logits. This would prevent the model from assigning very low probabilities to wrong classes (since logits would be clamped at 0).
However, "Logistic Regression" by definition implies Linear -> Sigmoid/Softmax. Adding a hidden layer with ReLU would turn it into a Neural Network (MLP), which is a different architecture (discussed in the next module).

### 20. How does `torch.utils.data.DataLoader` handle the last batch if the dataset size isn't divisible by batch size?

By default `drop_last=False`.
If dataset=1003 and batch_size=10.
You get 100 batches of 10, and 1 batch of 3.
The last batch is smaller.
This complicates metrics calculation (cannot just average batch accuracies). You must calculate `total_correct` and `total_samples` and divide at the end.
Alternatively, `drop_last=True` discards the remaining 3 samples to ensure all batches have fixed dimensions (useful for some hardware optimizations).

### 21. What happens if we initialize weights to be too large?

If weights are huge, $z = Wx + b$ will be huge (e.g., +1000, -5000).
Softmax of large numbers: $e^{1000} \to \text{Inf}$.
Probabilities become exactly 1.0 or 0.0 (Saturated).
Gradients through Softmax vanish when probabilities are saturated.
The model starts with high confidence (and high loss) and cannot learn.
Proper small initialization ensures logits are near 0, probabilities are near $1/K$ (uniform), and gradients flow nicely.

### 22. Can we use `model.parameters()` to change the Learning Rate for a specific layer?

`model.parameters()` returns a generator of all parameters.
To target specific layers, we can access them individually: `optimizer = SGD([{'params': model.layer1.parameters()}, {'params': model.layer2.parameters(), 'lr': 1e-3}])`.
For this simple Logistic Regression, there is only one layer (`model.linear`), so this advanced technique isn't needed, but it is fundamental for fine-tuning.

### 23. What is the "Confusion Matrix" for this 10-class problem?

A $10 \times 10$ matrix.
Rows: Actual Digit.
Columns: Predicted Digit.
Diagonal Element $(i, i)$: Correct predictions for Digit $i$.
Off-diagonal $(i, j)$: How often Digit $i$ was confused for Digit $j$.
It reveals specific weaknesses. E.g., The model frequently confuses 4s and 9s (similar shapes), but rarely confuses 1s and 8s. Accuracy hides this detail; Confusion Matrix exposes the "Pattern of Error."

### 24. Why is `num_workers=0` relevant for Windows users?

In `DataLoader`, `num_workers` controls parallel sub-processes.
On Linux/Mac, `fork()` is used (fast, copy-on-write).
On Windows, `spawn()` is used (slower, pickle overhead).
Due to implementation details of Python `multiprocessing` on Windows, using `num_workers > 0` inside Jupyter Notebooks often causes broken pipe errors or freezes if not wrapped in `if __name__ == '__main__':`. Setting it to 0 forces the main process to load data, which is slower but safe/compatible.

### 25. Explain the concept of "Generalization Gap" in this model.

Validation Accuracy is usually slightly lower than Training Accuracy.
The difference (Train - Val) is the Generalization Gap.
If gap is large $\to$ Overfitting.
For Logistic Regression on MNIST, since the model has low capacity (linear boundaries), it is actually hard to overfit. You are more likely to see **Underfitting** (High Bias), where both Train and Val accuracy saturate around 92-93% because a linear model simply cannot capture the complex non-linear shapes of handwriting perfectly.

### 26. What is the exact output shape of the model given a batch of 64 images?

Input: `[64, 784]`.
Weights: `[10, 784]`.
Operation: $X W^T$.
Dimensions: `[64, 784] @ [784, 10] = [64, 10]`.
Output is `[64, 10]`.
64 samples, each with 10 logit scores.

### 27. How does the learning rate affect the stochastic nature of the loss curve?

With Mini-batch SGD, the loss curve is "Jittery" or "Noisy" because each batch is an approximation of the true dataset.
High Learning Rate: High jitter. The optimizer jumps far based on each noisy batch. Can be unstable.
Low Learning Rate: Smoother curve. The steps are small, averaging out the noise over many iterations. But convergence is slow.

### 28. What is the standard deviation of pixel values in MNIST, and why does it matter?

Pixels are [0, 1]. Many are 0 (background).
Standardization typically involves `(x - mean) / std`.
This centers data around 0.
For Logistic Regression, unnormalized data [0, 1] works "okay."
But generally, centering pixels at 0 (range -1 to 1) helps the bias term initialization (starts at 0) match the expected activation mean. It prevents zig-zagging gradients common when all inputs are positive.

### 29. Can this model detect a "7" if it appears in the top-left corner?

The training data has digits centered.
Logistic Regression learns fixed weights for fixed pixels. "Pixels in the center" get high weights for detecting digits.
"Pixels in the corner" are always background, so their weights stay 0.
If you test on a "7" shifted to the corner, the model will fail completely.
Dense layers lack **Translation Invariance**. The model does not learn "The shape of 7"; it learns "Which pixels light up for a 7." This flaw motivates CNNs.

### 30. How do we manually test the model on a single image from the internet?

1.  Load image (PIL/Opencv).
2.  Resize to 28x28.
3.  Grayscale it.
4.  Invert colors (MNIST is white-on-black; photos are usually black-on-white).
5.  `ToTensor()` (scales 0-1).
6.  Normalize (if model was trained with normalization).
7.  `unsqueeze(0)` (batch dim).
8.  `model(input)`. `argmax`.
    Most failures in "Real World Demo" come from skipping step 4 (Color Inversion) or formatting differences.

### 31. What is the memory footprint of this model during Training vs Inference?

Inference: $X$ and $Y$. Minimal RAM.
Training: Must store $X$, $Y$, and **Intermediate Activations** for backprop.
Here, graph is shallow, so cost is low.
But generally, Training uses ~3x-4x more memory than Inference (Inputs + Weights + Gradients + Optimizer States).

### 32. What is the "History" list returned by the `fit` function?

It is a list of dictionaries: `[{'val_loss': ..., 'val_acc': ...}, ...]`.
One entry per epoch.
We plot this to visualize learning curves.
If `val_loss` goes down then up, we identify the exact epoch where overfitting started, guiding how long we should train next time.

### 33. Why do we define a `training_step` and `validation_step` method in the model class?

This is a design pattern (popularized by PyTorch Lightning).
It encapsulates the logic for "What happens in a single batch?" inside the model structure itself.
Instead of writing a raw loop with `loss = sum(y-p); loss.backward()`, we abstract it to `loss = model.training_step(batch)`.
This makes the main training loop generic and reusable for any model type, separating "Optimization Orchestration" from "Model Math."

### 34. What effect does `batch_size` have on the generalization of the model?

Small Batch (e.g., 8-32): Noisier gradients. Explores flatter minima. Better Generalization.
Large Batch (e.g., 2048): More accurate gradients. Converges to sharp minima. Often worse Generalization (Sharp Minima generalize poorly to shifted test data).
Large batch is faster (GPU saturation), but might require clever learning rate scaling (Linear Scaling Rule) to maintain accuracy.

### 35. Explain the `stack` operation used in the `evaluate` function.

`outputs = [x['val_loss'] for x in batch_outputs]` -> List of tensors.
`torch.stack(outputs).mean()`
`stack` converts a List of N (0-d) tensors into a single (1-d) tensor of size N.
Then `.mean()` reduces it to a scalar.
This aggregates the metrics from all mini-batches to calculate the epoch-level average.

### 36. Why is `torch.no_grad()` used in the validation set evaluation?

We do not train on validation data.
We do not need gradients.
Without `no_grad`, PyTorch would build a massive computational graph storing every validation operation.
This would waste massive GPU memory and compute time.
`no_grad` creates a "Volatile" context where operations are executed faster and memory is instantly freed.

### 37. What is `@torch.no_grad` decorator?

Alternative syntax to `with torch.no_grad():`.
You can put it above the `evaluate` function definition.
`@torch.no_grad` guarantees that function _always_ runs without gradients.
It is cleaner than indenting code blocks, reducing visual noise.

### 38. How does Class Imbalance affect accuracy?

If MNIST had 99% Zeros and 1% others.
A model predicting "Always Zero" gets 99% Accuracy.
But it is useless.
In such cases, Accuracy is deceptive. We need **Precision/Recall** or **F1 Score**.
Standard MNIST is balanced (roughly ~6000 of each digit), so Accuracy is a safe, valid metric.

### 39. What is the default initialization for `nn.Linear` in PyTorch?

It uses **Kaiming Uniform** (LeCun Uniform) scaled by $\frac{1}{\sqrt{fan\_in}}$.
It generates weights from a uniform distribution $[-\text{bound}, \text{bound}]$.
This ensures the variance of outputs is roughly the same as inputs (initially), preventing signal vanishing.
It's much better than standard `randn` (std=1), which would cause saturation in Softmax immediately.

### 40. Is it possible to get 100% accuracy on MNIST with Logistic Regression?

No.
SOTA with CNNs is 99.8%+.
Logistic Regression (Linear Model) typically caps out at ~92-93%.
Reason: Some digits are chemically inseparable linearly (e.g., a badly written 5 vs 6). The decision boundary must be curved/non-linear. A linear model cannot bend; it can only draw straight lines. The 7-8% error represents the "Non-Linearity Gap."

### 41. Explain the `model.train()` call inside the fit function.

It sets `model.training = True`.
While Logistic Regression has no Dropout/BatchNorm, good practice demands it.
This ensures that if we later add Dropout to the class, the code still works correcty (Dropout is active during fit, and inactive during evaluate).

### 42. What is `torch.cuda.is_available()`?

Checks if a GPU is accessible.
We use it to write device-agnostic code.
`device = 'cuda' if torch.cuda.is_available() else 'cpu'`
This allows the notebook to run on a cheap laptop (CPU) or a Cloud Server (GPU) without changing a single line of code.

### 43. Explain the "Pin Memory" concept in DataLoaders for Image Data.

Images are large tensors.
Moving them from CPU RAM to GPU VRAM is a bottleneck (PCIe bus).
`pin_memory=True` allocates the CPU tensor in special "Page-Locked" RAM.
CUDA drivers can DMA (Direct Memory Access) copy from Pinned RAM to VRAM much faster than from standard Pageable RAM.
It is a "Free Speedup" switch for GPU training.

### 44. What defines a "Cycle" in One-Cycle Learning Rate Policy?

Though not used in this basic notebook, modern training uses Learning Rate Schedulers.
A Cycle involves:

1.  Ramping LR up from small to max (Warmup).
2.  Descent LR from max to min (Annealing).
    This helps the model hop out of sharp basins initially and settle into broad, stable basins later.

### 45. What is the shape of the gradient w.r.t the Weight Matrix?

Same shape as the Weight Matrix: `[10, 784]`.
Each element `grad[i, j]` tells us: "How much does the loss change if we increase the connection strength between Pixel $j$ and Digit $i$?"
The optimizer subtracts this matrix (scaled by LR) from the weights.

### 46. Why do we divide pixel values by 255?

Normalization.
Neural Nets like inputs in range [0, 1] or [-1, 1].
Integers [0, 255] are too large. Large inputs $\to$ Large Activations $\to$ Saturated Gradients $\to$ Dead Training.
Scaling by 1/255 puts the data in the "Sweet Spot" for floating point math optimization.

### 47. What is the impact of "Random Seed" on reproduciblity?

This notebook uses random initialization and random shuffling.
Every run produces a slightly different 92% accuracy model.
To fix this: `torch.manual_seed(42)`.
This makes the "Random" numbers deterministic. Essential for debugging ("Did my code change improve the model, or did I just get lucky with initialization?").

### 48. How does `len(dataset)` work?

`MNIST` class implements `__len__`.
It returns the total count (60,000).
DataLoader uses this to calculate how many batches are in an epoch (`len / batch_size`).
This defines the loop range for the progress bar.

### 49. What is the "Step" in the context of the optimizer?

`optimizer.step()`
It applies the update rule: $\theta = \theta - \alpha \nabla$.
This modifies the tensors in-place.
The optimizer holds references to the model parameters (passed during init), so calling `.step()` updates the model weights directly.

### 50. Why do we reset gradients to zero?

`optimizer.zero_grad()`.
PyTorch accumulates gradients in the `.grad` attribute.
If we didn't zero, the gradient would be the sum of all history.
We want the gradient of the _current_ batch only.
Note: Some advanced techniques (Gradient Accumulation) intentionally delay `zero_grad` to simulate larger batches.

### 51. What is the `numel()` method?

"Number of Elements".
`tensor.numel()` returns total count.
For `[64, 10]`, numel is 640.
Used in MSE calculations manually, but CrossEntropy handles averaging internally.

### 52. Explain the "Logits" terminology.

Inherited from Statistics.
In logistic regression, Logit function is the inverse of the sigmoid function.
$L = \ln(p / (1-p))$.
In Deep Learning, "Logits" loosely refers to the **Unnormalized** output of the last linear layer, before the Softmax/Sigmoid activation generates actual probabilities.

### 53. What is "Top-k Accuracy"?

Standard Accuracy is "Top-1": Is the #1 guess correct?
Top-5 Accuracy: Is the correct answer in the top 5 guesses?
For MNIST (10 classes), Top-1 is standard.
For ImageNet (1000 classes), Top-5 is often reported because distinguishing between "Siberian Husky" and "Eskimo Dog" is ambiguous even for humans.

### 54. Can we define a model without a class?

Yes, using `nn.Sequential`.
`model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))`
This is concise and functional.
We use the Class approach in tutorials to teach the general pattern needed for complex networks (ResNet, Transformers) which cannot be expressed as a simple sequence of layers.

### 55. What is the "Input Layer" size?

784 neurons.
These are not "neurons" in the computational sense (they don't compute). They are placeholders for the data.
The visible layer of the network.

### 56. What is the "Output Layer" size?

10 neurons.
One per digit.
Typically matches the number of classes.
For binary classification, we could use 1 neuron (Sigmoid), but even then, 2 neurons (Softmax) is often preferred in PyTorch for consistency with multi-class code.

### 57. How do we access the scalar value of the loss?

`loss.item()`.
`loss` is a Tensor with 1 element (Rank-0).
`.item()` extracts it as a Python `float`.
Crucial for logging loops. `training_history.append(loss)` would store the entire Tensor Graph (Memory Leak). `training_history.append(loss.item())` stores just the number (Safe).

### 58. Explain "Broadcasting" in the validation accuracy calculation.

`preds` (128) == `labels` (128).
Element-wise comparison returns a Boolean tensor `[True, False, ...]`.
`torch.sum` counts True as 1.
Division by `len` gives ratio.
Broadcasting isn't explicitly used here, but vectorization is the key to doing this in one line rather than a for-loop.

### 59. What happens to the gradients after `optimizer.step()`?

They remain in `.grad`.
PyTorch does not auto-clear them.
This allows you to inspect them after the step.
They are only cleared when _you_ call `zero_grad()`.

### 60. Final Synthesis: Why is Logistic Regression the "Baseline"?

It is the simplest possible learnable model.
It is fast, convex (easy to train), and interpretable.
In any ML project, you start with Logistic Regression.
If it gets 92%, and your complex Deep Net gets 93%, you know the Deep Net isn't adding much value.
It establishes the "Linear Baseline"—how much of the problem can be solved by simple geometry? The rest is the "Non-Linear" complexity that justifies Deep Learning.
