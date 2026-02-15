# ML Sandbox

Revisiting core mathematics and implementations across deep learning and classical ML. Datasets are typically sourced from Kaggle or standard benchmarks (MNIST, IMDB, breast cancer Wisconsin).

---

### `numpy_nets/` — Neural Networks from First Principles

Pure NumPy implementation of a 3-layer network (784 → 10 → 10) for MNIST digit classification. No frameworks — just manual forward propagation, backpropagation, and gradient descent to build intuition for what PyTorch abstracts away.

Covers: forward/backward pass derivations, ReLU and softmax activations, one-hot encoding, weight initialisation, and vanilla gradient descent.

![forward/backward prop and parameter update equations](docs/relevant%20eqs.png)

---

### `Neural_Net_Theory/` — PyTorch Modular Training

Modular PyTorch project with reusable components for training and evaluation. Includes a TinyVGG CNN for image classification (pizza/steak/sushi) and an optimizer comparison study (Adam vs SGD vs SGD+momentum) on MNIST.

Covers: convolutional blocks (Conv2D, MaxPool, ReLU), training/evaluation loops, optimizer dynamics, model checkpointing, and TensorBoard visualisation.

---

### `tensorflow_intro.ipynb` — TensorFlow / Keras Walkthrough

End-to-end notebook covering the Keras workflow across image and text domains. Builds dense networks and CNNs for MNIST, plus embedding-based models for IMDB sentiment analysis.

Covers: Sequential and Functional APIs, activation/loss/optimizer selection guidelines, dropout and L1/L2 regularisation, batch normalisation, early stopping, learning rate scheduling, and embedding layers.

---

### `classification_revisit/` — Classical ML Evaluation

Scikit-learn binary classification on the breast cancer dataset with a focus on proper model evaluation.

Covers: logistic regression, stratified K-fold cross-validation, ROC curves, AUC scoring, and preprocessing pipelines.

---

### `ml_templates/` — Production ML Patterns

Reusable classification templates demonstrating production-grade practices: config management with dataclasses, `GridSearchCV` hyperparameter tuning, and structured train/evaluate/report workflows.

Covers: pipeline construction, stratified K-fold, confusion matrices, F1 optimisation, and reproducibility via random seeds.

---

### `nlp/` — Text and Language Models

Text classification with TF-IDF + logistic regression for sentiment analysis, and a HuggingFace script for loading BERT (`bert-base-uncased`) as a starting point for fine-tuning.

Covers: TF-IDF vectorisation, text preprocessing, pre-trained transformer models, and masked language modelling.
