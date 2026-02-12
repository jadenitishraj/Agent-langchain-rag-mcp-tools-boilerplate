# Study Material: Convolutional Neural Networks (CNNs) with CIFAR10

## Architecture Overview

This module introduces **Convolutional Neural Networks (CNNs)**, the gold standard for computer vision.
We move from grayscale MNIST (28x28) to color CIFAR10 (32x32 RGB).
Unlike MLPs which flatten images immediately, destroying spatial structure, CNNs process images as 3D Tensors (Channels, Height, Width) using **Convolutions**.
Key Components:

1.  **Conv2d Layer**: Applies learnable filters (kernels) to extract local features like edges and textures. It preserves spatial relationships.
2.  **ReLU**: Introduces non-linearity.
3.  **MaxPool2d**: Downsamples the image (reduces H and W) to reduce computation and induce **Translation Invariance**.
4.  **Classifier Head**: A final Flatten + Linear layer to map the high-level features to class probabilities.
    The notebook also covers handling real-world datasets using `torchvision.datasets.ImageFolder`, which loads images from a directory structure (Class Name $\to$ Folder).

## Exhaustive Q&A

### 1. Why are MLPs (Dense Networks) unsuitable for complex image datasets like CIFAR10?

1.  **Parameter Explosion**: For a $256 \times 256$ RGB image, input size is ~200,000. A hidden layer of 1000 units would require $200,000 \times 1000 = 200 \text{ Million}$ weights. This causes overfitting and massive memory usage.
2.  **Loss of Spatial Structure**: Flattening destroys variables like "pixel A is next to pixel B."
3.  **No Translation Invariance**: An MLP learns distinct weights for a cat in the top-left vs a cat in the bottom-right. It has to re-learn the visual concept of "cat" for every position.

### 2. How does a Convolutional Layer solve the parameter explosion problem?

**Weight Sharing**.
Instead of connecting every input pixel to every output neuron, we scan a small "Kernel" (e.g., $3 \times 3$) across the image.
The _same_ kernel weights are applied at every position.
Regardless of image size, the number of parameters depends only on the kernel size (9 weights) and number of channels. This efficiency allows deeper networks.

### 3. What are "Channels" in a CNN?

Input Image: 3 Channels (Red, Green, Blue).
Hidden Layers: $N$ Channels (Feature Maps).
Each channel represents a specific visual feature (e.g., Channel 1 activates on horizontal edges, Channel 2 on distinct texture).
As we go deeper, channels represent higher-level concepts (Eyes, Ears, Wheels).

### 4. Explain the dimensionality change: `Conv2d(3, 16, kernel_size=3, padding=1)`.

Input: `[Batch, 3, 32, 32]`.
Filter: $3 \times 3$.
Padding: 1 (Maintains Heigth/Width).
Output Channels: 16.
Result: `[Batch, 16, 32, 32]`.
We transformed 3 color planes into 16 feature maps. The spatial resolution (32x32) is preserved due to padding.

### 5. What is the role of "Padding"?

Without padding, the output feature map shrinks with every convolution.
A $3 \times 3$ kernel on a $32 \times 32$ image produces $30 \times 30$.
After 10 layers, the image would disappear ($0 \times 0$).
**Padding** adds border pixels (usually zeros) to the input so the kernel can center on edge pixels. `padding=1` for a $3 \times 3$ kernel keeps output size equal to input size ("Same Padding").

### 6. What is the role of "Stride"?

Stride controls the step size of the kernel scan.
Stride 1: Move 1 pixel. Output size $\approx$ Input.
Stride 2: Move 2 pixels. Skips every other pixel. Output size $\approx$ Input / 2.
Stride is used for **Downsampling**, an alternative to Max Pooling.

### 7. Explain "Max Pooling".

It looks at a window (e.g., $2 \times 2$) and picks the **Maximum** value.

1.  **Data Reduction**: Reduces spatial dimensions by factor of 2 (75% data removal). Reduces computation.
2.  **Invariance**: If the feature shifts slightly (1 pixel), the max value in the $2 \times 2$ window remains the same. The network becomes robust to small translations/jitter.

### 8. What is the "Receptive Field"?

The region of the original input image that affects a specific neuron in a deep layer.
Layer 1 neuron sees $3 \times 3$ pixels.
Layer 2 neuron sees $3 \times 3$ features from Layer 1, which represents $5 \times 5$ original pixels.
As we go deeper, the receptive field grows. The final neuron sees the entire image globally, allowing it to make a classification decision based on the whole context.

### 9. Why do we need `ImageFolder`? Why not just `TensorDataset`?

Real-world data is stored as files on disk (`train/cat/001.jpg`).
It is too large to load all into RAM as a single Tensor.
`ImageFolder` handles:

1.  Walking the directory structure.
2.  Mapping folder names ("cat") to integer labels (0).
3.  Lazy loading of images (reading from disk only when requested).

### 10. What is the memory layout difference between `NCHW` and `NHWC`?

PyTorch uses `NCHW` (Batch, Channels, Height, Width).
TensorFlow/Keras (historically) and standard images (matplotlib) use `NHWC`.
This is why we need `permute(1, 2, 0)` before plotting a tensor in matplotlib. We must move the Channel dimension from index 0 to index 2.

### 11. Calculate parameters for `Conv2d(3, 16, 3, 1)`.

Weights: $Out_{channels} \times In_{channels} \times K_h \times K_w$.
$16 \times 3 \times 3 \times 3 = 432$.
Bias: $16$.
Total: 448 parameters.
Compare this to a Dense layer mapping $3072 \to 16$, which would be $49,168$ parameters. CNN is 100x more efficient here.

### 12. Why do CNNs typically double the number of channels after pooling?

Pooling reduces spatial dimensions ($H, W$) by half. Area reduces by 4.
Information is lost.
To compensate, we usually double the number of Channels.
This keeps the computational density and information capacity roughly constant throughout the network. ("Squeeze spatially, expand deeper").

### 13. What is the output size of `MaxPool2d(2, 2)` on a `16x32x32` input?

Pooling operates on each channel independently.
$16$ channels remain $16$.
$32 \times 32$ reduces to $16 \times 16$.
Output: `[16, 16, 16]`.

### 14. What causes "Out of Memory" (OOM) errors in CNN training?

1.  **Batch Size too large**: Storing activations for 32x32 images is cheap, but for 512x512 images, it explodes.
2.  **Too many filters**: $1024$ filters require immense VRAM for weights and feature maps.
3.  **Linear layers**: The first Linear layer after flattening often consumes 80% of the total parameter memory if the spatial size wasn't reduced enough by pooling.

### 15. Why do we use `.tar.gz` for datasets?

Thousands of small files are slow to transfer/unzip due to file system overhead.
A Tarball lumps them into one continuous file block.
Always download archives, then extract on the local machine (preferably SSD).

### 16. What does `tar.extractall` do?

Unpacks the archive.
For CIFAR10, it creates a folder structure.
PyTorch scripts usually include a check: `if not os.path.exists(path): extract()`. This prevents extracting 60,000 files every time you restart the kernel.

### 17. How does CIFAR10 resolution affecting modeling difficulty compared to MNIST?

32x32 is low res, but RGB adds complexity.
Real-world objects (Frogs, Ships) have internal variance (colors, pose, background clutter) unlike white digits on black background.
A simple MLP fails on CIFAR10 (~40-50% acc) because it cannot handle this variance. CNNs are required to reach 80%+.

### 18. Why do we define `self.network = nn.Sequential(...)` inside the model?

Instead of defining `self.conv1`, `self.conv2`... and calling them in `forward`, we pack the whole featurizer into one container.
`forward` becomes simply: `return self.network(xb)`.
This is cleaner when the network is a simple stack of layers without branches (like ResNet connections).

### 19. What is the "Flatten" layer in `nn.Sequential`?

Between the Conv blocks and the Linear classifier, we need to reshape.
`nn.Flatten()` converts `[Batch, Channels, H, W]` to `[Batch, Channels*H*W]`.
Without this, the dense layer would complain about dimension mismatch.

### 20. How likely is the model to guess randomly on CIFAR10?

10 classes.
Random guess = 10% accuracy.
Any model getting ~10% is not learning.
A simple linear model gets ~30-40%.
A basic CNN gets ~70-75%.
State of the art gets 99%.

### 21. Why is `F.relu` used in `forward` but `nn.ReLU` in `Sequential`?

`nn.Sequential` requires objects (Modules). It calls their `forward()` method.
You cannot put a function `F.relu` into a Sequential list. You must wrap it in the `nn.ReLU()` class.

### 22. What is the effect of Kernel Size choice?

$3 \times 3$: Standard. Efficient. Captures fine details.
$5 \times 5$: Captures larger features but computationally expensive ($25$ weights vs $9$).
$1 \times 1$: Used for changing channel depth without spatial aggregation (dimensionality reduction).
Modern architectures stack many small $3 \times 3$ kernels rather than using large kernels.

### 23. Why do we normalize images?

Pixel values [0, 1] or [0, 255] are all positive.
Weights in the first layer will be biased to be negative to produce mean-zero activations.
Gradients will zig-zag.
Normalizing to mean 0, std 1 helps convergence.
`transforms.Normalize(means, stds)`.

### 24. What is the consequence of not using a GPU for CNNs?

Convolutions are computationally intensive ($H \times W \times C_{in} \times C_{out}$).
On CPU, CIFAR10 training might take hours per epoch.
On GPU, it takes seconds.
CNNs are practically unusable without hardware acceleration.

### 25. Explain the `permute` method usage for plotting.

`image_tensor` is `[3, 32, 32]`.
`plt.imshow` expects `[32, 32, 3]`.
`img.permute(1, 2, 0)` reorders axis 1$\to$0, axis 2$\to$1, axis 0$\to$2.
This is a metadata view change (fast).

### 26. What is "Underfitting" in the context of this tutorial?

If training accuracy is low (e.g., 50%) and validation accuracy is low.
The model is too simple (not deep enough, not enough channels) to capture the complexity of the data.
Solution: Add more layers, make layers wider (more kernels), or train longer.

### 27. What is "Overfitting" in the context of this tutorial?

Training accuracy is high (90%), Validation accuracy stops improving or drops (70%).
The model memorized the training pixels (including noise/background).
It fails to generalize to new images.
Solution: Regularization (Dropout, Weight Decay), Data Augmentation.

### 28. How many layers does the tutorial model have?

3 Convolutional blocks.
3 Linear layers.
Total depth ~6 layers.
This is considered a "Shallow" CNN by modern standards (ResNets are 50+ layers), but sufficient for CIFAR10.

### 29. Why `padding=1` is typically paired with `kernel_size=3`?

Formula: $Out = (In - K + 2P)/S + 1$.
We want $Out = In$ (stride 1).
$In = In - 3 + 2P + 1 \implies 2P = 2 \implies P = 1$.
This "Same Padding" simplifies architecture design; we only downsample explicitly via Pooling, never accidentally via Convolution shrinking.

### 30. What is the `num_workers` parameter in DataLoader?

It spawns sub-processes to load data from disk in parallel.
For `ImageFolder`, disk I/O is the bottleneck.
`num_workers=2` or `4` ensures the CPU prepares the next batch while the GPU processes the current one.
Too many workers can overwhelm CPU context switching; 2-4 is usually the sweet spot.

### 31. Explain the `__call__` vs `forward` invocation.

We write `model(images)`, not `model.forward(images)`.
`model(images)` calls `__call__`, which handles hooks (logging, autodiff setup) and _then_ calls `forward`.
Never call `forward` directly.

### 32. What visual limitations does this 32x32 dataset impose?

Objects are extremely blurry.
Fine details (text on a truck, breed of dog) are lost.
The model relies on dominant colors and blobs.
Humans sometimes struggle to classify CIFAR10. If a human can't do it, we shouldn't expect the model to have 100% accuracy.

### 33. Why is `Sigmoid` rarely used in CNNs?

Sigmoid maps to $(0, 1)$. Mean is 0.5.
Activations are always positive $\to$ Zig-zag gradients.
Saturation at tails $\to$ Vanishing gradients.
ReLU is superior for deep visual hierarchies.

### 34. How does `dataset.classes` get populated?

`ImageFolder` scans the subdirectories in `data/cifar10/train`.
It sorts them alphabetically.
`['airplane', 'automobile', 'bird'...]`.
It assigns index 0 to the first folder, 1 to second, etc.

### 35. What is the relationship between `batch_size` and training stability?

Smaller batch (32): Noisier gradients, can jump out of local minima, but slower epoch (PCIe overhead).
Larger batch (256): Smoother gradients, faster epoch, but might converge to sharp minima (worse generalization).
128 is a common compromise for CIFAR10.

### 36. Why do we define `conv_layer` as a reusable function?

DRY (Don't Repeat Yourself).
Every block in this CNN follows the pattern: Conv -> BatchNorm? -> ReLU -> Pool.
Creating a helper function `conv_block` makes the model definition readable and structurally consistent.

### 37. What role do the "corners" play in convolution?

Corner pixels are seen fewer times by the kernel than center pixels (assuming no padding).
With padding, they are treated more equally, but boundary artifacts can still exist.
In classification, main objects usually center, so boundary effects are negligible.

### 38. Can a CNN work on images of different sizes?

Convolutional layers are size-agnostic. They work on any $H \times W$.
However, the **Flatten** transition to the Linear layer imposes a fixed size requirement (matrix multiplication needs fixed columns).
To accept arbitrary sizes, we would need **Global Average Pooling** (GAP) instead of Flattening, which reduces distinct spatial dimensions to $1 \times 1$.

### 39. What is "Feature Hierachy"?

Layer 1: Edges, Colors.
Layer 2: Corners, Texture.
Layer 3: Parts (Wheels, Beaks).
Classification Layer: Objects (Truck, Bird).
This hierarchy emerges naturally during training.

### 40. Why does `valid_loss` matter if we care about `accuracy`?

Accuracy is discrete. It can plateau.
Loss is continuous. It shows if the model is getting "more confident" even if the top-1 prediction hasn't flipped yet.
Diverging loss (going up) while accuracy stays flat is an early warning of overfitting.

### 41. How does the kernel weights initialization affect training?

If initialized to 0, no symmetry breaking (dead training).
CNNs need careful variance scaling (Kaiming Init) because the number of inputs (fan-in) for a pixel is $K \times K \times C_{in}$.
PyTorch handles this by default.

### 42. What is the computational complexity of one Conv layer?

$O(H \cdot W \cdot C_{out} \cdot C_{in} \cdot K^2)$.
It scales linearly with pixels ($H \cdot W$) and quadratically with depth/channels ($C$).
This is why we reduce $H, W$ (Pooling) before we increase $C$ (Channels) deep in the network.

### 43. Why do we assume "Translation Invariance" is good?

A cat is a cat, whether on the left or right.
We want the class label to be invariant to position.
CNNs impose this prior belief structurally.
However, if position matters (e.g., "Is the cancer tumor in the left or right lung?"), CNNs with aggressive pooling might lose that spatial location info.

### 44. What is the difference between `cross_entropy` and `binary_cross_entropy`?

`cross_entropy`: Multi-class (10 classes). Softmax output. One correct class.
`binary_cross_entropy`: Two classes (Yes/No). Sigmoid output. Or Multi-label (can be tag A and tag B).
CIFAR10 is mutually exclusive multi-class, so we use `cross_entropy`.

### 45. Why is 3 Channels hardcoded in the first layer `Conv2d(3, ...)`?

Input images are RGB.
If we had Grayscale (MNIST), it would be 1.
If we had Hyperspectral/Satellite, it could be 4+.
The first layer _must_ match the data channels.

### 46. What happens if we deploy this model on a CPU server?

It works, just slower.
Inference (calculating one image) is fast enough on CPU for real-time web apps (~50ms).
Training is what requires GPUs.

### 47. Explain "Epoch".

One pass through the entire dataset (50,000 images).
We need multiple epochs because one gradient update step only nudges the weights slightly. It takes many nudges to traverse the loss landscape.

### 48. What is the "Bias-Variance Tradeoff" visualized here?

Early epochs: High Bias (Underfitting). Model is simplistic.
Late epochs: High Variance (Overfitting). Model is too sensitive to training noise.
We aim for the "Goldilocks" zone, usually found via Early Stopping.

### 49. How do we access the final probability of "Frog"?

`probs = F.softmax(model(image), dim=1)`.
`frog_prob = probs[0][6]` (assuming frog is index 6).
The raw output are logits, which aren't interpretable percentages until softmaxed.

### 50. Why use `tolist()` when plotting losses?

Matplotlib expects Python lists or NumPy arrays.
It cannot plot PyTorch Tensors directly (especially if they are on GPU).
`loss.item()` or `tensor.cpu().numpy()` converts them.

### 51. What does `torch.save` actually save?

It saves a serialized dictionary of arrays (Weights/Biases) using `pickle`.
It does NOT save the model class definition code.
You need the code `class Cifar10CnnModel...` present to load the weights back.

### 52. Why is the Validation set important for Hyperparameter Tuning?

If we tune Learning Rate, Batch Size, Architecture based on Test Set accuracy, we "leak" Test info into the model design.
The Test set is no longer an unbiased estimate of future performance.
Validation set is the "sandbox" for tuning. Test set is the "final exam."

### 53. Can we visualize the learned Filters?

Yes.
The weights of the first layer `model.network[0].weight` are `[16, 3, 3, 3]`.
We can normalize these small 3x3 grids and plot them as RGB images.
They often look like Gabor filters (edge detectors) or colored blobs.

### 54. What is "Gradient Accumulation" (concept)?

If GPU memory is too small for batch size 128.
Use batch size 32.
Run forward/backward 4 times _without_ calling `optimizer.step()`.
Gradients sum up.
Step once.
Effectively simulates batch size 128.

### 55. Why is `F.relu` stateless?

It performs an element-wise operation $y = max(0, x)$.
It doesn't need to learn anything (no weights).
So it doesn't need to be a class instance with parameters.

### 56. What is the "Top-5 Accuracy" for CIFAR10?

Usually not reported because 10 classes is small.
Top-1 is standard.
For 1000-class ImageNet, Top-5 is standard because distinctions are subjective.

### 57. How to improve the 75% accuracy?

1.  Deeper ResNet (Residual connections).
2.  Data Augmentation (Random crops, flips).
3.  Regularization (Weight Decay, Dropout).
4.  Learning Rate Scheduler (OneCycleLR).
    These are covered in the next tutorial (Notebook 05b).

### 58. What is the file extension `.pth`?

Visual convention for PyTorch weights.
PyTorch doesn't care (could be .txt), but `.pth` or `.pt` is standard.

### 59. Why do we need `optimizer.zero_grad()`?

PyTorch buffers gradients.
This is a feature (for RNNs and Gradient Accumulation).
For standard feedforward, we must manually clear the buffer before every batch calculation, or gradients will mix with history.

### 60. Final Synthesis: Why is CNN the "Inductive Bias" for images?

We assume the universe has translation symmetry (objects are the same everywhere).
We assume locality (pixels close together matter more than pixels far apart).
CNNs hardcode these assumptions into the architecture (Convolution + Pooling).
This makes them incredibly sample-efficient for images compared to generic MLPs which have to "learn" these rules from scratch.
