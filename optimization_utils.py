"""
Utility functions and classes to assist Bayesian Optimization Experiments.
"""
import gpytorch
import botorch
import xopt


def run_chain(eval_function = eval_function, n_init: int=5, n_steps: int = 60):
  low = 0.01
  high = 0.99
  vocs = VOCS(
    variables = {"x1": [low, high],
                 "x2": [low, high],
                "x3": [low, high],
                "x4": [low, high],
                "x5": [low, high],
                "x6": [low, high],
                "x7": [low, high],
                "x8": [low, high]
                },
    objectives = {"f": "MAXIMIZE"},
  )
  evaluator = Evaluator(function=eval_function)
  generator = ExpectedImprovementGenerator(vocs=vocs)
  X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
  X.random_evaluate(n_init)
  for i in range(n_steps):
    print(i)
    X.step()
  y1 = X.generator.data["f"]
  y1_maxs = np.maximum.accumulate(y1)
  del vocs, evaluator, generator, X, y1

  return y1_maxs
