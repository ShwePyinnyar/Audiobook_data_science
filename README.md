# Audiobooks Customer Purchase Prediction

## Overview

This project uses a Neural Network built with TensorFlow/Keras to predict whether an audiobook customer will purchase again in the future.

The dataset contains customer behavior and purchase-related features. The objective is to classify customers into two categories:

- **0** → Customer will not purchase again
- **1** → Customer will purchase again


---

## Project Workflow

### 1. Data Loading

The audiobook dataset was loaded using NumPy.

```python
df = np.loadtxt('audiobooks_data.csv', delimiter=',')
```

---

### 2. Data Preparation

The dataset was separated into:

- Input Features 
- Target Variable 

```python
unscaled_inputs_all = df[:, 1:-1]
targets_all = df[:, -1]
```

---

### 3. Class Balancing

The original dataset was imbalanced, with significantly more non-converting customers than converting customers.

To prevent model bias, the majority class was undersampled to create a balanced dataset.

Benefits:

- Reduces bias toward the majority class
- Improves classification performance
- Produces equal class priors

---

### 4. Feature Scaling

Input features were standardized before training.

```python
from sklearn import preprocessing

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
```

---

### 5. Data Shuffling

The dataset was randomly shuffled before splitting.

```python
np.random.shuffle(shuffled_indices)
```

This helps prevent ordering bias during training.

---

### 6. Train / Validation / Test Split

The balanced dataset was divided into:

| Dataset | Purpose |
|----------|----------|
| Training | Model learning |
| Validation | Hyperparameter tuning |
| Testing | Final evaluation |

---

### 7. Neural Network Model

A Deep Neural Network was implemented using TensorFlow/Keras.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])
```

---

### Model Configuration

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

#### Optimizer

- Adam

#### Loss Function

- Sparse Categorical Crossentropy

#### Metric

- Accuracy

---

### Early Stopping

Early stopping was implemented to prevent overfitting.

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=2
)
```

Benefits:

- Prevents unnecessary training
- Reduces overfitting
- Saves computational resources

---

## Results

| Metric | Value |
|----------|----------|
| Test Accuracy | **80%+** |

The model achieved over **80% classification accuracy** on unseen test data.

---

## Technologies Used

- Python
- NumPy
- TensorFlow
- Keras
- Scikit-learn

---


