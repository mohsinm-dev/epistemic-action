# epistemic-action

Small experiments on when an agent should gather information before acting.

The repository starts with a two-door clue task, then moves to evidence acquisition and model-misspecification stress tests. Each experiment keeps the assumptions explicit before introducing full active inference.

## 1. Clue experiment

A hidden reward is behind one of two doors. The agent can act immediately or inspect a noisy clue first.

For belief $q(s)$, expected information gain is

$$
IG = H[q(s)] - E_o[H[q(s|o)]].
$$

The simple epistemic policy trades task reward against clue cost and information gain. This is **active-inference-inspired**, not a full implementation of variational or expected free energy.

Run it with:

```bash
python -m epistemic_action.experiment
python -m epistemic_action.plot
```

## 2. Evidence acquisition

The second experiment asks:

> Is the most informative observation also the most useful observation?

A synthetic transaction has a hidden state: `legitimate` or `suspicious`. Before approving or rejecting it, a policy may acquire at most one evidence source. Sources differ in sensitivity, specificity, and acquisition cost.

The benchmark compares:

- `greedy`: act immediately
- `random`: acquire one random source
- `information_gain`: maximize expected entropy reduction
- `value_of_information`: acquire evidence only when expected decision improvement exceeds cost

For source $v$, the one-step net value is

$$
VoI(v) = E_o[max_d U(d|o,v)] - max_d U(d) - C(v).
$$

Run:

```bash
python -m epistemic_action.evidence_experiment
```

The source parameters are synthetic and are not estimates for real fraud or finance systems.

## 3. Correlation and calibration stress test

The one-step benchmark assumes evidence is correctly specified. The stress test deliberately violates that assumption.

Two evidence channels have the same marginal accuracy but share a signal draw with probability $rho$. When $rho=0$, the channels are conditionally independent. When $rho=1$, the second channel is completely redundant.

The benchmark compares four inference models:

- `single_source`: ignore the second observation
- `naive_independent`: use both observations but assume $rho=0$
- `correlation_aware`: use the configured dependence strength
- `oracle`: use the true dependence and true source accuracy

It separately varies the **true** source accuracy and the accuracy assumed by the inference model. This separates dependence misspecification from calibration error.

Run:

```bash
python -m epistemic_action.stress_experiment
```

It writes `results/stress.csv` with decision accuracy, asymmetric decision loss, Brier score, log loss, and mean posterior movement.

A useful diagnostic condition is:

```bash
python -m epistemic_action.stress_experiment \
  --priors 0.02 \
  --true-accuracy 0.85 \
  --assumed-accuracies 0.85 \
  --correlations 0.8 \
  --false-approve-cost 5 \
  --episodes 20000
```

Under strong positive dependence, a model that incorrectly treats repeated evidence as independent can become overconfident and cross the decision boundary when a correlation-aware model would not. This is the failure mode the stress test is designed to expose.

## Setup

Python 3.12+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plot]'
pytest
```

## Layout

```text
src/epistemic_action/
├── agents.py               # original clue policies and information gain
├── environment.py          # two-door clue environment
├── experiment.py           # clue reliability/cost sweep
├── plot.py                 # clue experiment plots
├── evidence.py             # binary evidence model, Bayes updates, VoI
├── policies.py             # one-step evidence-selection baselines
├── evidence_experiment.py  # reproducible evidence benchmark
├── stress.py               # correlated evidence model
└── stress_experiment.py    # correlation/calibration stress sweep

tests/
```

## Next question

The next step is sequential acquisition: after observing one source, should the agent stop, query another source, or escalate? That introduces policy depth and is the point where a POMDP, Bayesian sequential planner, and eventually expected-free-energy planning become meaningful comparisons.

## License

MIT
