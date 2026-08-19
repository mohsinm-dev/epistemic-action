# epistemic-action

Small experiments on when an agent should gather information before acting.

The repository starts with a two-door clue task, then moves through evidence valuation, model misspecification, finite-horizon Bayesian planning, and a controlled Active Inference comparison. Each stage keeps the assumptions visible instead of introducing the full framework at once.

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

The agent may repeatedly choose between acting, querying another unused source, or escalating at a fixed cost.

The benchmark compares:

- `greedy`: no evidence acquisition
- `information_gain`: maximize expected entropy reduction, ignoring cost
- `myopic_voi`: one-step decision value
- `lookahead`: exact finite-horizon Bayesian planning over future signals

The corrected default option-value condition is:

```text
P(suspicious)       = 0.05
false approve loss  = 5.0
false reject loss   = 1.0
escalation cost     = 0.40

screen accuracy     = 0.70
screen cost         = 0.05

review accuracy     = 0.95
review cost         = 0.20
```

Under this condition neither source has positive one-step decision value. Acting immediately has expected loss `0.250`. A two-step Bayesian planner still selects `screen`, because a clear result lets it stop while a flagged result can make `review` worth acquiring. Its exact expected loss is `0.212`.

This is the intended non-myopic option-value case.

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

Task losses are mapped into log-preference units with a positive `preference_precision` parameter. The scaling is swept rather than tuned to one favorable result.

The comparison includes:

- `bayes_lookahead`: exact finite-horizon expected-utility planning
- `standard_efe`: open-loop negative-EFE policy scoring
- `sophisticated_efe`: recursive observation-contingent negative-EFE planning

Run:

```bash
python -m epistemic_action.active_inference_experiment
```

The distinction between the two EFE planners is deliberate. The standard form propagates predicted beliefs without conditioning later policy choices on anticipated observations. The sophisticated form branches over possible observations, updates beliefs, and then chooses the next action.

At the corrected default condition and preference precision `5`, the exact Bayesian and sophisticated-EFE planners both select `screen`, while standard EFE stops. At low preference precision such as `0.5`, the epistemic term can dominate enough that sophisticated EFE selects the expensive `review` first. That is treated as a failure mode to measure, not a desired result.

This module is a benchmark-specific, inspectable implementation. It is **not a reimplementation of `pymdp`**.

## 6. Planner disagreement campaign

A single diagnostic condition is not enough to support a research claim. `disagreement_experiment.py` therefore sweeps the task and computes **exact expected trajectory loss** rather than relying on Monte Carlo estimates.

It varies:

- prior suspicious probability
- screen accuracy and cost
- review accuracy and cost
- Active Inference preference precision

For each condition it records:

- first action
- exact expected total loss
- exact expected query count
- exact expected evidence cost
- regret relative to finite-horizon Bayesian planning

The compared planners are:

- `myopic_voi`
- `bayes_lookahead`
- `standard_efe`
- `sophisticated_efe`

Run the default campaign with:

```bash
python -m epistemic_action.disagreement_experiment
```

The default grid contains `1,350` unique conditions and writes `5,400` planner rows to `results/disagreement.csv`.

The campaign is designed to answer two separate questions:

1. where does non-myopic planning matter?
2. where does the epistemic term create useful exploration versus unnecessary exploration?

The Bayesian planner is the utility-optimal reference **only for the correctly specified synthetic model used by this campaign**. Later misspecification experiments should not treat it as an oracle for the real world.

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
├── active_inference_experiment.py    # Bayesian vs EFE comparison
└── disagreement_experiment.py        # exact disagreement/regret campaign

tests/
```

## Research questions now

The repository is now set up to evaluate, rather than merely discuss, Active Inference.

The next questions are:

1. At what preference precision does epistemic value improve or hurt task utility?
2. When does standard open-loop EFE fail because useful future actions depend on future observations?
3. When does sophisticated EFE recover the same behavior as exact Bayesian planning, and when does it deliberately differ?
4. How do those conclusions change under source correlation, calibration error, and prior shift?
5. Does the same behavior reproduce in `pymdp` rather than only in this transparent reference implementation?

The goal is not to make Active Inference win. The goal is to identify which assumptions create useful epistemic behavior and what utility or computational cost comes with it.

## License

MIT
