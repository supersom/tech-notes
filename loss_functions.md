## Loss functions

#### Contrastive loss functions

Various **contrastive loss functions** differ mainly in:

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
