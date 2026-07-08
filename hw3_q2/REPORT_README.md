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

The contrastive hinge loss function calculates a penalty based on the relative cosine similarities of a query to a positive document and a negative document. The mathematical formulation is:

$$Loss = \max(0, \text{margin} + S(q, d_{negative}) - S(q, d_{positive}))$$

where $S$ represents the cosine similarity function. The loss evaluates to zero if the positive document's similarity score exceeds the negative document's score by at least the specified margin. If this condition is not met, a positive loss is incurred. Optimization of this objective pulls positive documents closer to the query vector while pushing negative documents further away.

### 4. Successful Retrieval Query

* **Query:** "Which Luna has rabbit food and a yellow collar?"
* **Result:** Successfully retrieved the correct document (D09) at Rank 1 with a cosine similarity score of 0.741.

### 5. Failed Retrieval Query

* **Query:** Test query T06 (After training)
* **Result:** The gold document (D13) was ranked at position 3 with a score of 0.398, subordinate to D06 (0.452) and D21 (0.447).
* **Analysis:** This failure is attributable to the model size and architectural limitations. The default model utilizes small, mean-pooled word embeddings devoid of attention mechanisms or contextual awareness. Consequently, it prioritizes superficial token overlap over complex semantic alignment, failing to distinguish between documents sharing similar vocabulary but disparate meanings.

### 6. Experimental Observation

The `bow` (Bag-of-Words) model achieved maximum performance metrics (Recall@1 = 1.0, Recall@3 = 1.0, MRR = 1.0) on both dev and test splits prior to any parameter updates. This demonstrates that the retrieval logic in this specific, constrained dataset is solvable entirely through exact keyword frequency matching. Conversely, the default continuous-vector embedding model required 50 epochs of training to reach suboptimal test metrics (Recall@1 = 0.875), establishing that dense representations introduce unnecessary complexity for this specific collection compared to a naive lexical baseline.