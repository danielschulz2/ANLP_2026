### 1. Main Run Command

```bash
python -m src.train --epochs 50 --margin 0.2 --seed 13

```

### 2. Retrieval Metrics

| Run             | Split | Recall@1 | Recall@3 | MRR                |
|-----------------|-------|----------|----------|--------------------|
| Before training | test  | 0.75     | 0.875    | 0.828125           |
| After training  | test  | 0.875    | 1.0      | 0.9166666666666666 |

### 3. Objective Function Explanation

The contrastive hinge loss function checks the cosine similarity scores between a query, a correct document, and an incorrect document. The formula is:

$$Loss = \max(0, \text{margin} + S(q, d_{negative}) - S(q, d_{positive}))$$

where $S$ is the cosine similarity score. The loss is zero if the correct document scores higher than the incorrect document by at least the margin amount. If it does not, the model gets a penalty. Training with this loss function forces the model to push correct documents closer to the query and push incorrect documents further away.

### 4. Successful Retrieval Query

* **Query:** "What are the requirements for operating the laser cutter in the metal workshop?"
* **Result:** The model correctly ranked document D15 at position 1 with a score of 0.530.

### 5. Failed Retrieval Query

* **Query:** "Which library provides group study rooms that can be booked online?"
* **Result:** The model incorrectly ranked D02 at position 1 with a score of 0.579, even though D02 states that the rooms "cannot be booked online". D01 was ranked second.
* **Failure Analysis:** This failure happens because of the model's basic design. It uses small word embeddings and takes their average, without looking at word order or grammar. It only checks for matching words like "library", "study rooms", "booked", and "online". Because all the words are averaged into a single vector, the model cannot understand negative phrases like "cannot be". It gives a high score simply because the vocabulary overlaps.

### 6. Experimental Observation

The Bag-of-Words (`bow`) model scored perfectly (Recall@1 = 1.0, Recall@3 = 1.0, MRR = 1.0) on the test split before any training was done. The default embedding model needed 50 epochs of training just to reach a Recall@1 of 0.875. This shows that for this small set of documents, exact word matching works perfectly. Using dense embeddings adds extra complexity that the small model cannot handle well.