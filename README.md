# epistemic-action

Small experiments on when an agent should gather information before acting.

The repository starts with a two-door clue task, then moves to evidence acquisition, model-misspecification stress tests, and finite-horizon sequential planning. Each experiment keeps the assumptions explicit before introducing full active inference.

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

## 4. Sequential evidence acquisition

The next experiment asks:

> Can an observation be worth acquiring only because it changes which future observation becomes useful?

This is the first genuinely non-myopic benchmark in the repository. The agent may now repeatedly choose between:

- acting immediately
- acquiring another unused evidence source
- escalating to a human at a fixed cost

The benchmark compares:

- `greedy`: never acquire evidence
- `information_gain`: repeatedly choose the most entropy-reducing source, ignoring cost
- `myopic_voi`: query only when one-step decision value is positive
- `lookahead`: exact finite-horizon Bayesian planning over future signals and remaining sources

The default environment contains a cheap `screen` and a stronger, more expensive `review`. At the configured prior, neither is worth buying as a one-step purchase. A two-step planner can still prefer the screen because a clear result lets it stop while a flagged result can make the stronger review worthwhile.

This is option value from information acquisition: the value of an observation comes partly from how it changes later decisions.

Run:

```bash
python -m epistemic_action.sequential_experiment
```

A useful diagnostic is:

```bash
python -m epistemic_action.sequential_experiment \
  --prior 0.05 \
  --false-approve-cost 5 \
  --escalation-cost 0.4 \
  --horizon 2 \
  --episodes 20000
```

With horizon `1`, exact lookahead reduces to the myopic policy. With horizon `2`, it can discover the value of screening before deciding whether stronger evidence is worth acquiring.

This remains an exact Bayesian planner under a tiny synthetic model. It is **not** yet an Active Inference implementation.

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
├── agents.py                 # original clue policies and information gain
├── environment.py            # two-door clue environment
├── experiment.py             # clue reliability/cost sweep
├── plot.py                   # clue experiment plots
├── evidence.py               # binary evidence model, Bayes updates, VoI
├── policies.py               # one-step evidence-selection baselines
├── evidence_experiment.py    # reproducible evidence benchmark
├── stress.py                 # correlated evidence model
├── stress_experiment.py      # correlation/calibration stress sweep
├── sequential.py             # finite-horizon Bayesian planner
└── sequential_experiment.py  # sequential acquisition benchmark

tests/
```

## Next question

The repository now contains the control problem we needed before introducing Active Inference: a belief state, sequential observations, action costs, stopping, escalation, and finite-horizon policies.

The next scientifically useful step is to represent the same sequential task explicitly as a small POMDP and compare three planners on the **same generative model**:

1. exact Bayesian dynamic programming
2. a tractable approximate planner
3. Expected Free Energy / discrete Active Inference

The goal is not to make Active Inference win. It is to identify when its epistemic term changes behavior, whether that behavior improves decision utility, and what computational price it pays.

## License

MIT
