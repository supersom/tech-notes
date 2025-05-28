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
