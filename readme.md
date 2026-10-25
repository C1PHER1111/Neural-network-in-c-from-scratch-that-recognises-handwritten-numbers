```markdown
# Building a C++ Neural Network from Scratch

I wanted to move beyond just importing high-level ML libraries like TensorFlow and actually figure out how neural networks work under the hood. Between October 2nd and October 25th, I spent around 53 hours on this project taking it all the way from studying the raw math to building a working C++17 app from scratch. The core model is a Multilayer Perceptron with a 784-node input layer for pixel images, a 128-node hidden layer, and a 10-node softmax output layer to classify handwritten digits. To keep performance fast and avoid the slowdowns of interpreted languages like Python, I wrote a custom memory engine that stores matrix data in contiguous blocks. This keeps dynamic memory allocations to a minimum and cuts epoch times down to milliseconds.

To wrap my head around the theoretical side, I went through the Mathematics for Machine Learning masterclass modules from Imperial College London and read through Chapter 1 of Michael Nielsen’s Neural Networks and Deep Learning. I used vector spaces, matrix multiplication, and linear transformations to code the forward pass where data flows through the model. Implementing backpropagation meant using partial derivatives, partial gradients, and the chain rule to work out how to update every single weight and bias to reduce overall error. Instead of using classic Sigmoid functions which lead to the vanishing gradient problem where learning stalls, I went with LeakyReLU paired with He weight initialisation to keep gradient flow strong across deep layers.

Coding the math revealed a lot of performance bottlenecks pretty quickly. My early versions were really sluggish, so I added mini-batch Stochastic Gradient Descent alongside epoch-level data shuffling to stop the network getting stuck in local minima. To get it running at proper speed, I engineered a zero-allocation backpropagation pass that reuses pre-allocated workspace buffers and cached weight transposes, rather than allocating memory on the heap during hot loops. These fixes dropped epoch times from 10 seconds down to a fraction of a second, letting the model hit over 95 percent accuracy in seconds.

Right now, single-digit recognition is rock solid and super accurate on clean, isolated numbers, and I am still working on getting the system to reliably recognize multiple numbers together in a single image.

---

## Features
- Zero-Allocation Engine: Custom 1D-flat matrix structures to maximize CPU cache locality.
- Modern C++ Architecture: Written entirely in C++17 without external heavy ML frameworks.
- Optimized Backprop Pass: Pre-allocated workspace buffers eliminate dynamic memory allocations in hot loops.
- Custom Preprocessing Pipeline: Integrated stb_image.h for thresholding, morphological dilation, and Connected Component Analysis.

---

## Tech Stack & Learning Resources
- Language: C++17
- Compiler: g++ / clang++
- Header Libraries: stb_image.h / stb_image_write.h
- Mathematics: Imperial College London (Mathematics for Machine Learning)
- Deep Learning Theory: Michael Nielsen (Neural Networks and Deep Learning, Chapter 1)

---

## How to Build and Run

### Prerequisites
Make sure you have a C++17 compatible compiler (g++ or clang++).

### 1. Compile the Project
```bash
g++ -std=c++17 main.cpp -o handwriting_app

```

### 2. Run Test

```bash
./handwriting_app -- your_image.jpg

```

---

## Output Example

```text
====================================================
  C++ MNIST Recognizer (fixed + multi-digit)        
====================================================

Loading MNIST training binaries...
Loaded 30000 samples successfully.

Training (target 95%)...
[Epoch   1] Accuracy: 96.04%  (lr 0.10)
>>> Target reached in 20.57 s (best 96.04%)

```

```

```