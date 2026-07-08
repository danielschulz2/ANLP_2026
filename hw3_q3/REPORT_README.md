## 1. dpo_objective.py

The file can be found in the $src$-folder

## 2. Test Output (`pytest -q`)
```text
........                                                                     [100%]
8 passed in 0.18s

```

## 3. Experiment Output

Command executed: `python run_experiment.py --data data/toy_preferences.jsonl --beta 0.5`

```text
Loaded 12 preference pairs
beta: 0.5
mean DPO loss: 0.6371
preference accuracy from DPO logits: 0.583

Per-example diagnostics:
ex01: logit= 0.350, loss= 0.533, prompt=Explain why RLHF can be unstable.
ex02: logit= 0.250, loss= 0.576, prompt=What is Direct Preference Optimization?
ex03: logit=-0.050, loss= 0.718, prompt=Summarize the main advantage of pairwise preference training.
ex04: logit= 0.400, loss= 0.513, prompt=Why keep a reference policy in DPO?
ex05: logit= 0.300, loss= 0.554, prompt=What does beta control in DPO?
ex06: logit=-0.075, loss= 0.731, prompt=Define a chosen response in preference data.
ex07: logit= 0.300, loss= 0.554, prompt=Explain why scalar reward RL can be difficult for language models.
ex08: logit=-0.150, loss= 0.771, prompt=What is a rejected response?
ex09: logit= 0.350, loss= 0.533, prompt=Why might DPO be easier to implement than PPO-based RLHF?
ex10: logit=-0.125, loss= 0.758, prompt=What is the intuition behind the DPO objective?
ex11: logit=-0.250, loss= 0.826, prompt=What does a negative DPO margin suggest?
ex12: logit= 0.250, loss= 0.576, prompt=How does DPO use preference pairs?

```

## 4. Answers to Questions

**What does a positive DPO logit mean?**
A positive DPO logit indicates that the current policy favors the chosen response over the rejected response more strongly than the reference policy does.

**Why does DPO compare the current policy against a reference policy?**
The reference policy establishes a baseline distribution. Comparing the current policy against it penalizes excessive deviation, preventing the model from generating degenerate text while optimizing for the provided preference data.

**What does the beta parameter control?**
The beta parameter controls the strength of the penalty applied for diverging from the reference policy. It scales the difference between the policy log-ratios and reference log-ratios.

**Pick one example with a low loss and one example with a high loss. Explain the difference.**

* **Low Loss (ex01):** Loss = 0.533, Logit = 0.350. The low loss occurs because the current policy already assigns a higher relative probability to the chosen response compared to the reference policy. The objective is satisfied, resulting in a minimal parameter update.
* **High Loss (ex11):** Loss = 0.826, Logit = -0.250. The high loss occurs because the current policy favors the rejected response. The objective is unsatisfied, indicating the model would receive a larger parameter update during training to correct this preference.
""")
