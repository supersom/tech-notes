## TF-IDF

**IDF (Inverse Document Frequency)** is a statistical measure used in information retrieval to quantify how **important** or **rare** a word is across a collection of documents (corpus).

---

### 🔹 Why IDF?

* Some words (like "the", "is", "and") appear in nearly every document — they don’t help distinguish one document from another.  
* **IDF downweights common words** and boosts rare, more informative ones.

---

### 🔹 IDF Formula:

$$
\text{IDF}(t) = \log\left(\frac{N}{1 + DF(t)}\right)
$$

Where:

* $\text{t}$: the term (word)  
* $N$: total number of documents  
* $\text{DF}(t)$: number of documents that contain term t  
* $\text{1+}$ in the denominator is for smoothing to avoid division by zero.

---

### 🔸 Example:

Assume you have 1000 documents:

* Word "the" appears in **950 documents** →  

$$
\text{IDF}(the) = \log{\left(\frac{1000}{951}\right)} \approx 0.02
$$
➝ Very low weight  

* Word "photosynthesis" appears in **10 documents** →  

$$
\text{IDF}(photosynthesis) = \log{\left(\frac{1000}{11}\right)} \approx 2.0
$$
➝ High weight

---

### 🔹 IDF is used in:

* **TF-IDF** \= Term Frequency × Inverse Document Frequency ➝ Emphasizes terms that are frequent **in one document** but **rare overall**.  

---

Here’s a **side-by-side table** illustrating **TF (Term Frequency)**, **IDF (Inverse Document Frequency)**, and **TF-IDF** effects for different words in a collection of **1000 documents**.

---

### **Example Table of TF, IDF, and TF-IDF**

| Word | TF in Document (Count) | DF (Docs Containing Word) | IDF $$\log{\left(\frac{N}{1+DF}\right)}$$ | TF-IDF $$\text{TF x IDF}$$ |
| :---- | :---- | :---- | :---- | :---- |
| **the** | 20 | 950 | $\log{(1000 / 951)} \approx 0.02$ | **0.4** |
| **is** | 10 | 900 | $\log{(1000 / 901)} \approx 0.05$ | **0.5** |
| **apple** | 5 | 100 | $\log{(1000 / 101)} \approx 1.0$ | **5.0** |
| **photosynthesis** | 3 | 10 | $\log{(1000 / 11)} \approx 2.0$ | **6.0** |
| **quantum** | 2 | 5 | $\log{(1000 / 6)} \approx 2.5$ | **5.0** |

---

### **Key Observations:**

1. **Common words ("the", "is") have low IDF** → Since they appear in almost all documents, their IDF is **close to 0**, meaning they contribute **little to TF-IDF**.  
2. **Less frequent words ("photosynthesis", "quantum") have high IDF** → Since they appear in **fewer documents**, they are more **important** in distinguishing documents.  
3. **TF-IDF balances term frequency and importance** → Even if a term appears multiple times in a document (high TF), it **only gets a high TF-IDF if it is rare in the corpus**.

## BoW \- Bag of Words

The **Bag of Words (BoW)** model is one of the simplest and most widely used representations of text in natural language processing (NLP) and information retrieval.

---

### **🔹 What is the Bag of Words Model?**

At its core:

BoW represents a **document** as a **set of individual words**, ignoring grammar, word order, and sentence structure — keeping only **which words appear** and **how often**.

---

### **🧱 How it works:**

1. **Build a vocabulary**:  
    From a collection of documents, list **all unique words**.

2. **Vectorize each document**:  
    For every document, count how many times each word in the vocabulary appears.  
    These counts form the **feature vector** for that document.

---

### **🧾 Example:**

Suppose we have 3 documents:

* Doc1: `"the cat sat"`

* Doc2: `"the dog barked"`

* Doc3: `"the cat barked"`

**Vocabulary**: `["the", "cat", "sat", "dog", "barked"]`

**BoW Vectors**:

| Document | the | cat | sat | dog | barked |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Doc1 | 1 | 1 | 1 | 0 | 0 |
| Doc2 | 1 | 0 | 0 | 1 | 1 |
| Doc3 | 1 | 1 | 0 | 0 | 1 |

* These vectors can now be used for **similarity comparisons**, **classification**, or **retrieval**.

---

### **✅ Pros:**

* Simple and fast

* Captures basic term frequency

* Works well with small datasets

### **❌ Cons:**

* Ignores word order and context ("Paris is beautiful" ≈ "Beautiful is Paris")

* Can produce large, sparse vectors

* Doesn't handle synonyms or semantics

---

## CBoW \- Continuous Bag of Words

The **Continuous Bag of Words (CBOW)** model is a variant of Word2Vec used to **learn word embeddings** — dense vector representations of words based on their surrounding context.

---

### 🔍 CBOW in a Nutshell

**CBOW predicts a target word from its context words.** That is, given the words around a missing word, the model tries to guess the missing word.

---

### 🧠 How It Works

Suppose you have this sentence:

"The quick brown fox jumps over the lazy dog"

To train on the word `"fox"` with a window size of 2, CBOW takes the **context words**:

\["quick", "brown", "jumps", "over"\]

And trains the model to **predict** the target word:

→ "fox"

So it learns:

f("quick", "brown", "jumps", "over") ≈ "fox"

It does this over many such examples in a large corpus, and in the process learns embeddings for each word.

---

### 📦 Architecture Summary:

1. **Input**: Several context words (e.g., 4 words)  
2. **Embedding lookup**: Get vectors for each context word  
3. **Average** (or sum) those vectors  
4. **Feed into a softmax layer** to predict the center (target) word

---

### 🤝 CBOW vs Skip-gram (Other Word2Vec Model)

| Model | Predicts | Training Target | Good for |
| :---- | :---- | :---- | :---- |
| **CBOW** | Center word from context | Given neighbors → predict word | **Faster** on frequent words |
| **Skip-gram** | Context from center word | Given word → predict neighbors | Better for **rare words** |

---

### 🧠 Why It’s Called "Continuous" Bag of Words?

* Like Bag of Words (BoW), it treats input as a **set of words**, ignoring word order (it's still a “bag”).  
* But it's **"continuous"** because it uses **continuous-valued vectors (embeddings)** rather than binary or count features.

---
