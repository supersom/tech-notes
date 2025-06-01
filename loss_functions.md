## Loss functions

### Overview of Loss functions

Loss functions can be broadly categorized based on the type of learning task they address. Here's a breakdown of the main families:

1. Classification Loss Functions:

    **Task**: Used for classification tasks, where the goal is to predict a categorical label or class for a given input.

    **Purpose**: Guide the model to make the correct class predictions.

    **Key Characteristic**: These loss functions penalize the model when it makes incorrect class predictions.

    **Examples**:

    A. Core Classification Losses:  

    - **Cross-Entropy Loss**: The most common loss function for multi-class classification problems.
        $$\mathcal{L} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

        Where:​

        - $\hat{y}_i$ is the predicted probability from softmax for class $i$
        - $y_i$ is the ground truth one-hot encoded label (i.e., 1 if correct class, else 0)

    - **Binary Cross-Entropy Loss**: Used for binary classification problems.
        $$\mathcal{L}_{\text{binary}}(y, \hat{y}) = -\left[y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right]$$

        Where:

        - $y \in \{0, 1\}$ is the ground truth label
        - $\hat{y} \in (0, 1)$ is the predicted probability of class 1 (after sigmoid)

    - **Categorical Cross-Entropy Loss**: Another name for cross-entropy when the outputs are one-hot encoded.
        $$\mathcal{L}_{\text{categorical}}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

        Where:

        - $y = \left[y_1, y_2, \dots, y_K \right]$ is the one-hot encoded true label
        - $\hat{y} = [\hat{y}_1, \hat{y}_2, \dots, \hat{y}_K]$ are the predicted class probabilities (after softmax)
        - $K$ is the number of classes

    B. Modified/Advanced Classification Losses:

    - **Hinge Loss**: Used in support vector machines (SVMs) for binary classification with a clear margin.
        $$\mathcal{L}_{\text{hinge}} = \max(0, 1 - y \cdot f(x))$$
        Where:

        - $y \in \{-1,1\}$ is the true label
        - $f(x)$ is the model output (usually a score, not a probability)

        For the multiclass hinge loss (also known as the structured SVM loss):

        $$\mathcal{L}_{\text{multi-hinge}} = \sum_{i \neq y} \max\left(0, s_i - s_y + \Delta\right)$$

        Where:

        - $s_i$ is the predicted score for class $i$,
        - $s_y$ is the score for the true class $y$,
        - $\Delta$ is the margin (usually set to 1).

    - **Focal Loss**: A modified version of cross-entropy that addresses class imbalance by focussing on difficult-to-classify examples during training.
    - **Dice Loss**: Used in medical image segmentation and other tasks where overlap is important.
    - **Tversky Loss**: Another loss function used in segmentation.
    - **Exponential Loss**: Used in boosting algorithms like AdaBoost.

    C. Losses for Specific Classification Problems:

    - **Softmax Loss**: Cross-entropy loss paired with a Softmax activation in multi-class classification.

        $$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
    $$
        Where:
        ​
        - $z_i$ is the input logit for class $i$
        - $K$ is the total number of classes
        - Logit: raw, unnormalized output score of a classification model—typically the last linear layer before applying the softmax function. So in an NN, Input → Hidden Layers → Linear Layer → (Logits) → Softmax → (Probabilities)

    - **Connectionist Temporal Classification (CTC) Loss**: Used for sequence-to-sequence classification problems like speech recognition or handwriting recognition.

2. Regression Loss Functions:

    **Task**: Used for regression tasks, where the goal is to predict a continuous numerical value.

    **Purpose**: Guide the model to make accurate value predictions.

    **Key Characteristic**: Penalize the model based on the difference between its predictions and the actual values.

    **Examples**:

    A. Core Regression Losses:
    - **Mean Squared Error (MSE) / L2 Loss**: Calculates the average of the squared differences between predictions and true values. Penalizes larger errors more heavily.
    - **Mean Absolute Error (MAE) / L1 Loss**: Calculates the average of the absolute differences between predictions and true values. More robust to outliers than MSE.
    - **Root Mean Squared Error (RMSE)**: Returns the error to the same scale as the target.

    B. Robust Regression Losses:
    - **Huber Loss**: A robust loss function that combines the benefits of MSE and MAE, making it less sensitive to outliers.
    - **Log-Cosh Loss**: Similar to Huber loss but smoother and differentiable everywhere.
    - **Quantile Loss**: Allows for different quantiles of predictions.

    C. Specialized Regression Losses:
    - **Poisson Loss**: Used for count-based regression problems.
    - **Tweedie Loss**: A more general form of Poisson loss for various distributions.

3. Metric Learning Loss Functions:

    **Task**: Used for learning embeddings that reflect the similarity between data points.

    **Purpose**: Learn an embedding space that reflects similarity relationships between data.

    **Key Characteristic**: Penalize the model if similar data points have dissimilar embeddings and vice versa.

    **Examples**:

    - **Contrastive Loss**: Minimizes the distance between similar pairs and maximizes the distance between dissimilar pairs.
    - **Triplet Loss**: Aims to minimize the distance between the anchor and the positive example while maximizing the distance between the anchor and the negative example.
    - **N-Pair Loss**: Extends the triplet loss using N positive and N negative examples.
    - **Lifted Structured Loss**: Maximizes the margin between positive and negative examples.

4. Loss Functions for Generative Models:

    **Task**: Used for learning generative models, where the goal is to reconstruct or generate data similar to the training data.

    **Purpose**: Train generative models that accurately reproduce the input data.

    **Key Characteristic**: Measures how well the model can reconstruct the input data.

    **Examples**:
    Reconstruction Loss: Used in autoencoders, which are models that learn to reconstruct their inputs.

    A. Variational Autoencoders (VAEs):

    - **Reconstruction Loss**: Measures the difference between the generated and the original data.
    - **KL Divergence Loss**: Measures the difference between the learned distribution and a prior.
    - **ELBO Loss**: The combination of reconstruction loss and KL divergence used in VAEs.

    B. Generative Adversarial Networks (GANs):

    - **Adversarial Loss**: Trains a generator to fool a discriminator and a discriminator to distinguish between generated and real data.

5. Other Loss Functions:

    **Task**: Other types of loss functions that don't neatly fit into the above categories, such as those for sequence-to-sequence models or for specific architectures.

    **Purpose**: These losses are used in specialized tasks.

    **Examples**:

    A. Losses for Reinforcement Learning:

    - **Policy Gradient Loss**: Used in policy-based reinforcement learning algorithms.
    - **Temporal Difference Loss**: Used in temporal difference learning algorithms.

    B. Other Losses:

    - **Perceptual Loss**: Used in image generation and style transfer.
    - **Wasserstein Loss (Earth Mover's Distance)**: Measures the distance between probability distributions, used in GANs and other areas.

Relationship Between Categories:

Classification, regression, and metric learning are the core loss function families, each with their distinct characteristics.
Other loss functions may be variations of those core loss functions or may be used in combination with them. Reconstruction losses are specifically used for generative models.

In summary, loss functions can be categorized into families based on the type of learning task they address. These include classification, regression, metric learning, reconstruction, and other specialized losses.


#### Key Considerations

**Choosing the Right Loss Function**: The appropriate loss function depends on the specific machine learning task, data distribution, and goals.

**Understanding Properties**: Familiarize yourself with the characteristics of each loss function.

This overview provides a more complete picture of the diverse landscape of loss functions used in machine learning. But new loss functions are continually being developed, so it's good to stay up-to-date with research in the field.

---

### Metric learning loss functions - Deep dive

Metric learning loss functions, with their unique focus on learning embeddings based on similarity, form an important and distinct family of loss functions.

Various **metric learning loss functions** differ mainly in:

- how they define and process **similarity/dissimilarity**,
- the **number of samples** they consider at a time,
- the **type of distance or similarity metric** used, and,
- whether they are used for **supervised**, **unsupervised**, or **self-supervised** learning.

Here's a comparative summary to help clarify their differences:

---

### **1. Pair-based Contrastive Loss (Siamese-style)**

**Samples used**: Anchor + 1 other sample (positive or negative)

**Distance metric**: Usually **Euclidean distance**

**Goal**:

- Minimize distance between positive pairs (same class)  
- Maximize distance between negative pairs (different classes)

**Loss**:

$$ L = y \cdot D^2 + (1 - y) \cdot \max(0, m - D)^2 $$

**Use case**: Signature verification, face verification (e.g., FaceNet), image similarity

**Pros**:

- Simple and intuitive

**Cons**:

- Doesn't leverage full batch context  
- Can be inefficient for large-scale training

---

### **2. Triplet Loss**

**Samples used**: Anchor + 1 positive + 1 negative

**Distance metric**: Usually **Euclidean** or **cosine distance**

**Goal**:

- Pull anchor-positive close  
- Push anchor-negative apart by a margin $\alpha$

**Loss**:

$$ L = \max(0, |f(a) - f(p)|^2 - |f(a) - f(n)|^2 + \alpha) $$

**Use case**: Face recognition, person re-identification

**Pros**:

- Captures relative distance between classes

**Cons**:

- Needs careful triplet mining (hard/semi hard negatives)  
- Computationally expensive due to combinatorial triplets

---

### **3. NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy Loss)**

**Samples used**: Anchor + 1 positive + many in-batch negatives

**Similarity metric**: **Cosine similarity**, with temperature scaling

**Goal**:

- Maximize similarity between anchor and its positive (e.g., different augmentations)  
- Minimize similarity with all other negatives in the batch

**Loss**:

$$ L = -\log \frac{\exp(\text{sim}(a, p)/\tau)}{\sum_{i=1}^{2N} \exp(\text{sim}(a, i)/\tau)} $$

**Use case**: Self-supervised learning (e.g., SimCLR, MoCo)

**Pros**:

- Effective batch-level contrast  
- No need for explicit labels (self-supervised)

**Cons**:

- Needs large batch size or memory bank to work well

---

### **4. Supervised Contrastive Loss (SupCon)**

**Samples used**: Anchor + all other samples with the same label in the batch

**Similarity metric**: **Cosine similarity**

**Goal**:

- Encourage anchor to be close to **all positives** (same class), not just one  
- Separate from all negatives (different class)

**Loss**:

$$ L_i = \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p)/\tau)}{\sum_{a \in A(i)} \exp(\text{sim}(z_i, z_a)/\tau)} $$

Where:

- $P(i)$ are same-class examples  
- $A(i)$ are all other examples in batch

**Use case**: Image classification (when labels are available)

**Pros**:

- Uses label information efficiently

**Cons**:

- Still requires large batches for stable training

---

### **5. Cosine Contrastive Loss**

**Samples used**: Pairs (positive/negative)

**Similarity metric**: **Cosine similarity**

**Goal**:

- Maximize cosine similarity between similar pairs  
- Minimize cosine similarity between dissimilar pairs

**Loss**:

$$ L = y \cdot (1 - \cos(q, d)) + (1 - y) \cdot \max(0, \cos(q, d) - \delta)^2 $$

\[I’m not so sure about having the ^2\]

**Use case**: Text embedding models (e.g., sentence transformers)

**Pros**:

- Works well in high-dimensional vector spaces (like NLP) 

**Cons**:

- Less suitable when magnitude of embedding matters

---

### ✅ Summary Table

| Loss Type | Sample Type | Metric | Margin? | Task Type | Batch-Dependent? | Supervised? |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Pairwise Contrastive | Anchor \+ 1 other | Euclidean | ✅ | Similarity search | ❌ | ✅ |
| Triplet Loss | Anchor, positive, negative | Euclidean | ✅ | Metric learning | ❌ | ✅ |
| NT-Xent | Anchor \+ in-batch pairs | Cosine \+ Softmax | ❌ | Self-supervised | ✅ | ❌ |
| SupCon (Supervised) | Anchor \+ same-labels | Cosine \+ Softmax | ❌ | Classification | ✅ | ✅ |
| Cosine Contrastive | Anchor \+ 1 other | Cosine | ✅ | Text/embedding | ❌ | ✅ |

---
