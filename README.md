# epistemic-action

Small experiments on when an agent should gather information before acting.

The repository starts with a two-door clue task, then moves to a one-step evidence-acquisition benchmark where information has reliability, decision value, and cost. The aim is to keep each assumption visible before introducing full active inference.

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

The second experiment asks a sharper question:

> Is the most informative observation also the most useful observation?

A synthetic transaction has a hidden state: `legitimate` or `suspicious`. Before approving or rejecting it, a policy may acquire at most one evidence source. Sources differ in sensitivity, specificity, and acquisition cost.

The benchmark compares four policies:

- `greedy`: act immediately
- `random`: acquire one random source
- `information_gain`: choose the source with maximum expected entropy reduction
- `value_of_information`: choose evidence only when its expected improvement in decision utility exceeds its cost

For evidence source $v$, the one-step net value is

$$
VoI(v) = E_o[max_d U(d|o,v)] - max_d U(d) - C(v).
$$

A positive value means the evidence is worth acquiring under the stated decision costs.

The source parameters in this benchmark are **synthetic**. They are not estimates for real fraud or finance systems.

Run the benchmark with:

```bash
python -m epistemic_action.evidence_experiment
```

It writes `results/evidence.csv` with accuracy, decision loss, evidence cost, total loss, evidence rate, and manual-review rate.

A useful default condition is:

```bash
python -m epistemic_action.evidence_experiment \
  --priors 0.1 \
  --false-approve-costs 5 \
  --episodes 20000
```

Under this setup, pure information gain prefers the most informative source even when it is expensive. The value-of-information policy can prefer a cheaper source because it optimizes decision value rather than uncertainty reduction alone.

That distinction is the point of the experiment.

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
└── evidence_experiment.py  # reproducible evidence benchmark

tests/
```

## Next question

The current evidence model assumes each source is correctly specified and conditionally independent given the hidden state. The next useful stress test is to violate those assumptions with correlated or miscalibrated evidence and measure when the acquisition policies become overconfident.

## License

MIT
