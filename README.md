# MNIST Digit Classifier — PyTorch CNN

A convolutional neural network trained from scratch on the MNIST dataset.  
Achieves **~99% test accuracy** in 10 epochs on a CPU.

![Training Curves](training_curves.png)

---

## Architecture

```
Input [B, 1, 28, 28]
  └─ ZeroPad2d(2)         → [B,  1, 32, 32]
  └─ Conv2d(1→16, k=5)    → [B, 16, 28, 28]
  └─ BatchNorm2d + ReLU   → [B, 16, 28, 28]
  └─ MaxPool2d(2)         → [B, 16, 14, 14]
  └─ Flatten              → [B, 3136]
  └─ Linear(3136→10)      → [B, 10]  logits
```

Total trainable parameters: **~31 800**

---

## Results

| Split      | Accuracy |
|------------|----------|
| Train      | ~99.7%   |
| Validation | ~99.5%   |
| Test       | ~99.3%   |

Trained on 5 000 samples (subset) to keep runtime short.  
Full dataset training pushes accuracy above 99.5%.

---

## Quickstart

```bash
git clone https://github.com/your-username/mnist-cnn
cd mnist-cnn
pip install -r requirements.txt
jupyter notebook mnist_cnn.ipynb
```

The MNIST dataset downloads automatically on first run (~11 MB).

---

## Project structure

```
mnist-cnn/
├── mnist_cnn.ipynb       # main notebook
├── requirements.txt
├── training_curves.png   # generated after training
└── mnist_cnn.pth         # saved model weights (generated after training)
```

---

## Key implementation notes

- **No Softmax in the model** — `CrossEntropyLoss` applies log-softmax internally. Adding it explicitly causes numerical instability.
- **Loss averaged over all batches**, not taken from the last batch only.
- **`argmax` for predictions**, not `round` — this is a multiclass problem.
- **`scheduler.step()`** called once per epoch after the optimiser step.
- **Reproducible** — all random seeds fixed via `torch.manual_seed(42)`.

---

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- torchvision, torchmetrics, torchsummary, matplotlib
