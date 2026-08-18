# epistemic-action

A minimal experiment on when an agent should gather information before acting.

The environment hides reward behind one of two doors. A noisy clue can be inspected before choosing, but inspecting it may have a cost. The repository compares a greedy agent with an epistemic agent that values both expected task reward and expected information gain.

This is intentionally small. It is meant to make epistemic action inspectable before moving to full active-inference implementations.

## Question

When is information worth seeking?

For a belief $q(s)$, the epistemic agent compares acting immediately with a policy that first observes the clue. Its information term is the expected reduction in belief entropy:

$$
\operatorname{IG} = H[q(s)] - \mathbb{E}_{o}\left[H[q(s \mid o)]\right].
$$

The clue policy is scored as

$$
\mathbb{E}[R \mid \text{clue}] - c + \lambda\,\operatorname{IG},
$$

where $c$ is clue cost and $\lambda$ controls the weight on information gain.

This is **active-inference-inspired**, not a full implementation of variational free energy or expected free energy. Keeping that distinction explicit is part of the exercise.

## Run

Python 3.12+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plot]'

pytest
python -m epistemic_action.experiment
python -m epistemic_action.plot
```

The sweep writes `results/sweep.csv` and one reward plot per clue cost under `results/figures/`.

You can change the experiment directly from the command line:

```bash
python -m epistemic_action.experiment \
  --episodes 20000 \
  --reliabilities 0.5,0.6,0.7,0.8,0.9,1.0 \
  --costs 0.0,0.05,0.1,0.2,0.3,0.5 \
  --information-weight 1.0 \
  --seed 7
```

## What to look for

At chance-level reliability, the clue carries no information. As reliability rises, observing it becomes useful. Raising clue cost should eventually suppress information seeking.

The useful boundary is not whether the epistemic agent "wins", but where its behavior changes as reliability, cost, and information weight change. That boundary is the starting point for more serious experiments on model misspecification and calibrated uncertainty.

## Layout

```text
src/epistemic_action/
├── agents.py       # Bayesian update, information gain, agent policies
├── environment.py  # binary latent state and noisy clue
├── experiment.py   # reproducible sweeps and CSV output
└── plot.py         # simple result plots

tests/
```

## License

MIT
