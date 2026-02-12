# Study Material: ResNets, Regularization & Modern Training Techniques

## Architecture Overview

This module represents a leap from "Basic CNNs" to **Modern Deep Learning**.
We tackle CIFAR10 again, but aim for **90%+ Accuracy** (vs 75%) using state-of-the-art techniques:

1.  **ResNet (Residual Network)**: Uses skip connections ($y = x + F(x)$) to allow training very deep networks without vanishing gradients.
2.  **Regularization**:
    - **Data Augmentation**: Random crops/flips to artificially expand the dataset and force invariance.
    - **Batch Normalization**: Stabilizes training by normalizing layer inputs.
    - **Weight Decay**: Penalizes large weights (L2 regularization).
    - **Gradient Clipping**: Prevents exploding gradients.
3.  **Scheduling**:
    - **One Cycle Policy**: A super-convergence scheduler that ramps learning rate up and down, significantly speeding up training.

## Exhaustive Q&A

### 1. What problem does ResNet solve?

**Vanishing Gradient Problem** in Deep Networks.
In plain CNNs, as depth increases (e.g., 20+ layers), gradients backpropagated to early layers become infinitesimally small or numerically unstable.
Adding more layers actually _hurt_ performance due to optimization difficulties (degradation problem), not just overfitting. ResNet solved this, allowing 100+ layer networks.

### 2. How does a Residual Block work mathematically?

Traditional Block: $y = F(x)$. We try to learn the mapping $F$.
Residual Block: $y = F(x) + x$. We try to learn the _residual_ (difference) $F(x)$.
If the optimal function is Identity ($y=x$), a ResNet drives weights of $F(x)$ to 0, which is easy.
In a plain net, navigating non-linear layers to output $x$ exactly is difficult.

### 3. Why do Skip Connections help gradient flow?

During Backpropagation, the gradient passes through the $+ x$ path unchanged.
$\frac{\partial y}{\partial x} = \frac{\partial (F(x)+x)}{\partial x} = \frac{\partial F}{\partial x} + 1$.
The "$+1$" term ensures that even if $\frac{\partial F}{\partial x}$ is small (vanishing), the gradient signal is preserved. It acts as a "Gradient Superhighway."

### 4. What is Data Augmentation?

Artificially increasing the diversity of training data by applying random transformations.
Examples: Random Crop, Horizontal Flip, Rotation, Color Jitter.
It creates "New" images that preserve the class label but change pixel values.
It forces the model to learn invariant features (e.g., a "Car" is still a car if shifted left or flipped).

### 5. Why do we normalize inputs using dataset Mean and Std?

Channels have different distributions.
If Red channel is $[0, 255]$ and Green is $[0, 10]$, weights would be unbalanced.
Standardization: $x' = (x - \mu) / \sigma$.
This centers data around 0 with unit variance.
It makes the error surface spherical, allowing SGD to converge faster and more evenly across dimensions.

### 6. Explain `RandomCrop(32, padding=4)`.

1.  Pad the 32x32 image with 4 pixels on all sides (creating 40x40).
2.  Randomly crop a 32x32 patch from this 40x40 image.
    Effect: The object shifts position within the frame. The model learns to recognize the object regardless of precise placement (Translation Invariance).

### 7. What is `padding_mode='reflect'`?

When padding, what value do we put in the border?
`zeros`: Black border. Can create artificial edges.
`reflect`: Mirrors pixels from inside the image. Preserves texture continuity (e.g., grass stays grass) and reduces artifacts.

### 8. What is Batch Normalization (BatchNorm)?

A layer that normalizes its inputs using the mean and variance of the _current batch_.
$y = \frac{x - \mu_B}{\sigma_B} \gamma + \beta$.
$\gamma$ (scale) and $\beta$ (shift) are learnable parameters.
It reduces "Internal Covariate Shift" (debatable theory), but practically it smooths the loss landscape, allows higher learning rates, and reduces sensitivity to initialization.

### 9. Why does BatchNorm act as a Regularizer?

The mean $\mu_B$ and variance $\sigma_B$ depend on the specific random batch.
This adds stochastic noise to the activations (similar to Dropout, but weaker).
The model cannot rely too heavily on any single neuron's exact value, improving generalization.

### 10. Can we remove Dropout if we use BatchNorm?

Often, yes.
ResNets typically use BatchNorm and _no_ Dropout.
BatchNorm provides enough regularization. Using both can sometimes lead to conflict (variance shift).

### 11. What is "Weight Decay"?

A penalty term added to the Loss function: $L_{total} = L_{data} + \lambda \sum w^2$.
It forces weights to be small.
Small weights mean the model function is smoother (less complex).
PyTorch implements this directly in the optimizer: `weight_decay=1e-4`.

### 12. What is "Gradient Clipping"?

If gradients explode (become huge), a single step can shoot the weights to Infinity (NaN).
`nn.utils.clip_grad_value_` caps gradients at a threshold (e.g., 0.1).
It ensures numerical stability, especially in Recurrent Networks or deep CNNs with high learning rates.

### 13. What is the "One Cycle Policy"?

A Learning Rate Scheduler invented by Leslie Smith.
Phase 1: LR increases from low to high (Warmup).
Phase 2: LR decreases from high to near zero (Annealing).
Rationale: High LR allows traversing saddle points and finding flat minima (better generalization). Low LR at the end fine-tunes into the minimum.

### 14. What are the benefits of `fit_one_cycle`?

Super-Convergence.
We can train in 1/10th the epochs needed for constant LR.
We can use much higher maximum learning rates than usually possible, because the schedule prevents divergence.

### 15. What is the "Learning Rate Finder"?

A technique to find the optimal `max_lr`.

1.  Start with tiny LR.
2.  Train for one epoch, exponentially increasing LR.
3.  Plot Loss vs LR.
4.  Find the point where Loss stops decreasing and starts exploding.
5.  Pick a value slightly before the minimum (typically 1/10th of the divergence point).

### 16. Why do we shuffle the training set but not the validation set?

Training: To break correlations between batches. SGD assumes i.i.d. samples.
Validation: Order doesn't matter for calculating total accuracy. Not shuffling saves a tiny bit of compute overhead.

### 17. What is `pin_memory=True` in DataLoader?

Host (CPU) RAM relies on Virtual Memory (paging).
GPU functions (CUDA) require data in "Pinned" (Page-Locked) RAM to transfer via DMA.
`pin_memory=True` allocates the staging buffer in Pinned RAM, speeding up the CPU $\to$ GPU transfer.

### 18. Why use `Conv2d` followed by `BatchNorm2d` followed by `ReLU`?

This is the standard "block".
Conv acts as the linear transformation.
BN centers the data for the activation.
ReLU applies non-linearity.
(Note: Pre-activation ResNets use BN-ReLU-Conv order, but standard ResNet is Conv-BN-ReLU).

### 19. Why do we remove Bias from `Conv2d` when using `BatchNorm`?

BatchNorm subtracts the mean $\mu$.
Any bias $b$ added by Conv acts as a constant shift to the mean.
$BN(Wx + b) = BN(Wx)$. The bias is cancelled out by the mean subtraction.
Having a bias in Conv is redundant and wastes parameters. BN has its own bias $\beta$.

### 20. What is "Global Average Pooling" (GAP)?

Instead of Flattening `[Batch, 512, 4, 4]` to `[Batch, 8192]`:
We take the Average of each 4x4 feature map.
Result: `[Batch, 512, 1, 1]` $\to$ `[Batch, 512]`.

1.  **Parameter Efficiency**: Removes the massive Linear layer weights.
2.  **Spatial Robustness**: Doesn't care where the feature is, just sums its presence.
3.  **Variable Input Size**: Works on any image size.

### 21. Why does the ResNet block use two $3 \times 3$ convs?

Why not one $5 \times 5$?
Two $3 \times 3$ typically have the same receptive field as one $5 \times 5$ ($5$ pixels).
But fewer parameters ($9+9=18$ vs $25$).
And **two non-linearities** (ReLUs) instead of one. More expressive power per parameter.

### 22. What happens to the dimensions in a Skip Connection?

We add $x + F(x)$.
They must have the same shape.
If $F(x)$ changes channels or resolution (stride), we must apply a transformation to $x$ (usually a $1 \times 1$ conv or pooling) to match dimensions before adding.

### 23. What is the `num_workers` trade-off?

`num_workers=0`: Main process loads data. Slow (GPU waits for CPU).
`num_workers=N`: N sub-processes.
Pros: Parallel loading.
Cons: High RAM usage (each worker copies dataset meta-data). Initial startup overhead.
Guideline: `num_workers` = number of CPU cores.

### 24. Why is `momentum` used in SGD?

Grades are noisy.
Momentum accumulates a running average of gradients.
It smooths out the zig-zagging in steep valleys.
It helps plow through flat regions (plateaus).
In `fit_one_cycle`, momentum is often cycled inversely to LR (High momentum with Low LR, Low momentum with High LR).

### 25. Explain the GPU memory usage of ResNet vs Simple CNN.

ResNet is deeper: stores more intermediate activations for backprop.
However, GAP layer saves parameter memory.
Bottleneck is usually Activation Memory (features maps).
ResNet9 (tutorial version) is light. ResNet50 is heavy.

### 26. Why validation accuracy might be higher than training accuracy initially?

1.  **Dropout/Augmentation**: Active during Train, disabled during Val. Validation data is "cleaner" and easier.
2.  **Loss Calculation**: Training loss is a moving average during the epoch. Validation loss is calculated at the end of the epoch (after model has improved).

### 27. What is `model.eval()` doing?

1.  Disables Dropout (neurons always active).
2.  Freezes BatchNorm statistics (uses running mean/var instead of batch mean/var).
    Essential for consistent inference.

### 28. Why `model.train()`?

Re-enables Dropout and BatchNorm updates.
Forgetting to switch back to `.train()` is a common bug.

### 29. What is "Transfer Learning" (mentioned as future step)?

Instead of training ResNet from scratch (random weights), download a ResNet pre-trained on ImageNet (1.2M images).
Fine-tune on CIFAR10.
Drastically reduces training time and usually improves accuracy (knowledge transfer).

### 30. Why is ResNet9 chosen for this tutorial?

A simplified variant (Davidnet).
8 Convolutional layers + 1 Linear.
It captures the essence of ResNet (Residuals, BN) but trains in seconds/minutes on Colab.
Full ResNet18 is often overkill for CIFAR10.

### 31. How does `weight_decay` prevent overfitting?

Large weights imply the model is relying on high-frequency signal (noise/sharp transitions).
Small weights imply a smoother function.
Occam's Razor: Simpler (smoother) models generalize better.

### 32. What is the computational cost of BatchNorm?

Very low.
Element-wise shifts and scales.
During inference, $\mu$ and $\sigma$ are folded into the Conv weights, effectively becoming free (zero cost).

### 33. Why use `Conv2d` with `stride=2` instead of MaxPool?

It allows the learnable weights to decide _how_ to downsample, rather than a fixed "Max" operation.
It's more flexible.
Modern architectures ("All-Convolutional Nets") often ditch Pooling layers entirely in favor of Strided Convs.

### 34. What is the tensor shape flow in the first layers?

Input: `[B, 3, 32, 32]`.
Conv: `[B, 64, 32, 32]` (assuming padding).
Pool/Stride: `[B, 64, 16, 16]`.
The spatial resolution halves while channels increase.

### 35. What is the "Epoch" vs "Batch" vs "Iteration" distinction?

Epoch: One pass over full dataset (50k images).
Batch: A chunk processed at once (e.g., 256 images).
Iteration: One update step.
Steps per Epoch = $\frac{50000}{256} \approx 196$.

### 36. Why is high Learning Rate risky without Scheduling?

Early in training, we are far from the optimum. Gradients are huge.
High LR can cause divergence (shooting off to infinity).
Warmup (One Cycle) allows starting huge steps safely once gradient direction stabilizes.

### 37. What is the visual effects of `RandomHorizontalFlip`?

Mirror image.
A left-facing car becomes right-facing.
Valid for natural scenes.
Invalid for text (MNIST digits): a '5' flipped is not a real number. We must choose augmentations carefully based on domain.

### 38. How does `test_loader` differ from `train_loader`?

`train_loader`: `shuffle=True`, Augmentations enabled.
`test_loader`: `shuffle=False`, No Augmentations (just Normalize).
We need stable, deterministic evaluation.

### 39. What is the `running_mean` in BatchNorm?

BN tracks the average mean of training batches.
This "Running Mean" is used during inference to normalize single examples (where batch statistics wouldn't exist or would be noisy).

### 40. Why do we wrap the model in `to_device`?

To seamlessly move parameters to GPU.
PyTorch models initialize on CPU by default.
Calculations fail if Inputs are on GPU and Weights are on CPU.

### 41. What is the disadvantage of Data Augmentation?

Slower Epochs?
CPU work increases (computing transforms).
If CPU is slow, it can starve the GPU.
Convergence takes _more_ epochs because the model sees "harder/changing" data. (But final accuracy is higher).

### 42. Explain the "Residual" logic: $F(x)$ vs $H(x) = F(x) + x$.

If identity is desired, $F(x) \to 0$.
Zero weights are the default initialization/decay target.
So the network naturally defaults to identity behavior (passing info through) and only adds complexity where needed.

### 43. Why do we visualize the data _after_ transformation?

The raw images on disk look normal.
The tensors entering the model are Normalized (negative values, weird colors).
We need to "Un-normalize" (multiply by std, add mean) to display them correctly for debugging.

### 44. What determines the number of channels (128, 256)?

Hyperparameter tuning.
Rule of thumb: Double channels when spatial resolution halves.
Balances total activations roughly constant.

### 45. What is `grad_clip` threshold choice (e.g. 0.1)?

Empirical.
Too low: Slows learning (directions cut short).
Too high: Doesn't prevent explosions.
0.1 to 1.0 is standard.

### 46. Why do we compute `len(train_loader)`?

To tell the Scheduler how many steps there are in an epoch.
Cyclic schedulers need to know the _total steps_ to plan the curve up and down.

### 47. What is "Adam" vs "SGD"?

SGD: Simple gradient descent.
Adam: Adaptive Moment Estimation. Maintains per-parameter learning rates.
Often converges faster.
With well-tuned schedules (One Cycle), SGD+Momentum is competitive or better (generalizes better) than Adam for Image Classification.

### 48. Why does accuracy plateau?

1.  Learning rate too high (bouncing around minimum).
2.  Model capacity reached (cannot learn more features).
3.  Data noise (irreducible error).
    LR annealing (dropping LR) usually breaks the plateau.

### 49. Can we use `ResNet` for regression?

Yes.
Replace the final `Linear(in, 10)` with `Linear(in, 1)`.
Change loss to MSE.
Architecture is valid for any task mapping Image $\to$ Output.

### 50. Why `padding_mode='reflect'` is better than `zeros`?

`zeros` creates a hard edge $0 \to image\_value$.
Convolution detects this as a "feature" (edge).
Model learns that objects near the edge have a black line. Testing images might not have this.
Reflection is smoother.

### 51. What is the role of the loss function?

Using `cross_entropy`.
It combines Softmax (probs) + Negative Log Likelihood.
Encourages correct class probability $\to 1$.

### 52. Why is `torch.no_grad()` used in validation?

We don't need gradients for validation.
Saves memory (doesn't store computation graph).
Saves compute.

### 53. How does PyTorch handle RGB vs Grayscale internally?

Just `Channels` dimension.
$C=3$ vs $C=1$.
Conv2d kernels adjust depth automatically.

### 54. What is Inductive Bias?

Built-in assumptions.
CNN: Locality, Translation Invariance.
ResNet: Depth is good, Gradient flow matters.
Augmentation: Viewpoint invariance.

### 55. Why do we check `if torch.cuda.is_available()`?

Portability.
Code runs on a laptop (CPU) or Server (GPU) without crashes.

### 56. What does `images.to(device, non_blocking=True)` do?

Asynchronous transfer.
Allows CPU to fetch next batch while GPU processes current transfer.
Hides latency.

### 57. Parameter count of ResNet9?

~6.5 Million.
Small enough to train fast.
Large enough to model CIFAR10.

### 58. Why do we define a `BaseImageClassificationBase` class?

Boilerplate reduction.
`training_step`, `validation_step`, `epoch_end` are common patterns.
Inheritance keeps the main model code clean.

### 59. What is "Mode Collapse" (not GAN)?

In training?
Maybe getting stuck in bad local minima.
One Cycle helps explore to avoid this.

### 60. Final synthesis: Why did we achieve 90% in 5 mins?

Combination of:

1.  **ResNet**: Efficient architecture.
2.  **GPU**: Massive parallelism.
3.  **One Cycle**: Fastest convergence path.
4.  **Batch Norm**: Stable, aggressive updates.
    This represents the modern Deep Learning efficiency baseline.
