# epistemic-action

Small experiments on when an agent should gather information before acting.

The repository starts with a two-door clue task, then moves through evidence valuation, model misspecification, finite-horizon Bayesian planning, and finally a controlled Active Inference comparison. Each stage keeps the assumptions visible instead of introducing the full framework at once.

## 1. Clue experiment

A hidden reward is behind one of two doors. The agent can act immediately or inspect a noisy clue first.

For belief $q(s)$, expected information gain is

$$
IG = H[q(s)] - E_o[H[q(s|o)]].
$$

The simple epistemic policy trades task reward against clue cost and information gain. This is **active-inference-inspired**, not a full implementation of variational or expected free energy.

Run:

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

The benchmark compares:

- `single_source`: ignore the second observation
- `naive_independent`: use both observations but assume $rho=0$
- `correlation_aware`: use the configured dependence strength
- `oracle`: use the true dependence and true source accuracy

Run:

```bash
python -m epistemic_action.stress_experiment
```

It separately varies true source accuracy and assumed source accuracy so dependence misspecification and calibration error can be studied independently.

## 4. Sequential evidence acquisition

The next experiment asks:

> Can an observation be worth acquiring only because it changes which future observation becomes useful?

The agent may now repeatedly choose between acting, querying another unused source, or escalating at a fixed cost.

The benchmark compares:

- `greedy`: no evidence acquisition
- `information_gain`: maximize expected entropy reduction, ignoring cost
- `myopic_voi`: one-step decision value
- `lookahead`: exact finite-horizon Bayesian planning over future signals

The default environment contains a cheap `screen` and a stronger `review`. At the default prior neither source is worthwhile as a one-step purchase, but a two-step planner can still value the screen because a clear result lets it stop while a flagged result can make the review worthwhile.

Run:

```bash
python -m epistemic_action.sequential_experiment
```

This is still an exact Bayesian planner, not Active Inference.

## 5. Active Inference comparison

The fifth experiment keeps the **same sequential task** and adds a transparent discrete Active Inference formulation.

`active_inference.py` contains:

- a factorized A/B/C/D-style generative model
- expected hidden-state information gain in nats
- an open-loop `standard_efe` planner
- an observation-contingent `sophisticated_efe` planner

The hidden transaction state is static. A controlled context factor records whether the current action is `screen`, `review`, `approve`, `reject`, or `escalate`. Evidence outcomes, terminal outcomes, and query costs are represented as separate observation modalities. Source availability is fully observed task memory and is enforced by the planner rather than represented as a hidden state.

Task losses are mapped into log-preference units with a positive `preference_precision` parameter. This scaling is intentionally swept rather than treated as a free tuning knob.

The comparison includes:

- `bayes_lookahead`: exact finite-horizon expected-utility planning
- `standard_efe`: open-loop negative-EFE policy scoring
- `sophisticated_efe`: recursive observation-contingent negative-EFE planning

Run:

```bash
python -m epistemic_action.active_inference_experiment
```

A useful precision sweep is:

```bash
python -m epistemic_action.active_inference_experiment \
  --prior 0.05 \
  --false-approve-cost 5 \
  --escalation-cost 0.4 \
  --horizon 2 \
  --precisions 0.5,1,2,5,10,20 \
  --episodes 20000
```

The distinction between the two EFE planners is deliberate. The standard form propagates predicted beliefs forward without conditioning later policy choices on anticipated observations. The sophisticated form recursively branches over possible observations, updates beliefs, and then chooses the next action. In this small benchmark that distinction is exactly what determines whether the planner can represent the option value of screening.

This module is a benchmark-specific, inspectable implementation. It is **not a reimplementation of `pymdp`**. Reproducing the same task in the current JAX-first `pymdp` stack is a later validation step.

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
├── agents.py                         # original clue policies and information gain
├── environment.py                    # two-door clue environment
├── experiment.py                     # clue reliability/cost sweep
├── plot.py                           # clue experiment plots
├── evidence.py                       # binary evidence model, Bayes updates, VoI
├── policies.py                       # one-step evidence-selection baselines
├── evidence_experiment.py            # reproducible evidence benchmark
├── stress.py                         # correlated evidence model
├── stress_experiment.py              # correlation/calibration stress sweep
├── sequential.py                     # exact finite-horizon Bayesian planner
├── sequential_experiment.py          # sequential acquisition benchmark
├── active_inference.py               # A/B/C/D model and EFE planners
└── active_inference_experiment.py    # Bayesian vs EFE comparison

tests/
```

## Research questions now

The repository is finally at the point where Active Inference can be evaluated rather than merely discussed.

The next questions are:

1. At what preference precision does epistemic value improve or hurt task utility?
2. When does standard open-loop EFE fail because useful future actions depend on future observations?
3. When does sophisticated EFE recover the same behavior as exact Bayesian planning, and when does it deliberately differ?
4. How do those conclusions change under source correlation and calibration error?
5. Does the same behavior reproduce in `pymdp` rather than only in this transparent reference implementation?

The goal is not to make Active Inference win. The goal is to identify which assumptions create useful epistemic behavior and what utility or computational cost comes with it.

## License

MIT
