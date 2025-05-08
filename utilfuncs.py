import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc
from split_and_delay import SND
from xopt import Xopt, Evaluator
from xopt.generators.bayesian import MOBOGenerator
from xopt.generators.bayesian import ExpectedImprovementGenerator, UpperConfidenceBoundGenerator
from xopt.generators.bayesian import MOBOGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt import VOCS
from xopt import Xopt


def get_snd_outputs(inputs):
  """
  Perturbs the Split & Delay to simulate outputs.
  Takes a 8 dim array ( t1_th1, t1_th2, chi1, chi2, t4_th1, t4_th2, chi1, chi2)
  The entries lie in the uniform unit interval.
  They are scaled in the function to lie in [-100e-6, +100e-6].
  Returns a torch tensor of dim 2 of (BPE(microns), Intensity/60000)
  Args:
    inputs: array of dim 8
  Returns:  
    torch tensor of 2 dim
  """
  inputs = inputs*200e-6 - 100e-6
  result = []
  for x in inputs:
    snd = SND(9500)
    x = np.array(x)
    snd.mvr_t1_th1(x[0])
    snd.mvr_t1_th2(x[1])
    snd.mvr_t1_chi1(x[2])
    snd.mvr_t1_chi2(x[3])
    snd.mvr_t4_th1(x[4])
    snd.mvr_t4_th2(x[5])
    snd.mvr_t4_chi1(x[6])
    snd.mvr_t4_chi2(x[7])
    snd.propagate_delay()
    dh1 = snd.get_t1_dh_sum()
    dd = snd.get_dd_sum()
    dh4 = snd.get_t4_dh_sum()
    do = snd.get_do_sum()
    my_IP_sum = snd.get_IP_sum()
    my_intensity = dh1 + dd + dh4 + do + my_IP_sum
    do_centroid = snd.get_IP_r()
    result.append([do_centroid*1e6, my_intensity/60000])
  return torch.tensor(result, dtype=torch.float)


def eval_function_intensity(input_dict: dict) -> dict:
  """
  Evaluates the SND function for input, returns the Intensity only.
  """
  x1, x2, x3, x4, x5, x6, x7, x8 = input_dict["x1"], input_dict["x2"], input_dict["x3"], input_dict["x4"], input_dict["x5"], input_dict["x6"], input_dict["x7"], input_dict["x8"]
  Xinp = np.expand_dims(np.array([x1, x2, x3, x4, x5, x6, x7, x8]), axis=0)
  output = get_snd_outputs(Xinp)
  f = output[0][1].item()
  return {"f": f}


def eval_function_bpe(input_dict: dict) -> dict:
  """
  Evaluates the SND function for input, returns the Beam Position Error.
  """
  x1, x2, x3, x4, x5, x6, x7, x8 = input_dict["x1"], input_dict["x2"], input_dict["x3"], input_dict["x4"], input_dict["x5"], input_dict["x6"], input_dict["x7"], input_dict["x8"]
  Xinp = np.expand_dims(np.array([x1, x2, x3, x4, x5, x6, x7, x8]), axis=0)
  output = get_snd_outputs(Xinp)
  f = output[0][0].item()
  return {"f": f}

def eval_function_mobo(input_dict: dict) -> dict:
  """
  Evaluates the SND function for input, returns the Beam Position Error (0-350 in microns) 
  and the Intensity (0-100, in percentage).
  """
  x1, x2, x3, x4, x5, x6, x7, x8 = input_dict["x1"], input_dict["x2"], input_dict["x3"], input_dict["x4"], input_dict["x5"], input_dict["x6"], input_dict["x7"], input_dict["x8"]
  Xinp = np.expand_dims(np.array([x1, x2, x3, x4, x5, x6, x7, x8]), axis=0)
  output = get_snd_outputs(Xinp).squeeze()
  f1, f2 =  output[0].item(), output[1].item() #BPE (0-350), Intensity (0-100)
  return {"f1": f1, "f2": f2}


def eval_function_scalarized(input_dict: dict) -> dict:
  """
  Evaluates the SND function for input, returns the scalarized objective: log(BPE / Intensity**2)
  """
  x1, x2, x3, x4, x5, x6, x7, x8 = input_dict["x1"], input_dict["x2"], input_dict["x3"], input_dict["x4"], input_dict["x5"], input_dict["x6"], input_dict["x7"], input_dict["x8"]
  Xinp = np.expand_dims(np.array([x1, x2, x3, x4, x5, x6, x7, x8]), axis=0)
  output = get_snd_outputs(Xinp).squeeze()
  f1, f2 =  output[0].item()/350, output[1].item() #BPE (0-350), Intensity (0-100)
  f = np.log(f1 / f2**2)
  return {"f": f}


def eval_function_constrained_bpe(input_dict: dict) -> dict:
  """
  Evaluates the SND function for input, returns the Beam Position Error as 
  a constraint, and the Intensity as the objective.
  """
  x1, x2, x3, x4, x5, x6, x7, x8 = input_dict["x1"], input_dict["x2"], input_dict["x3"], input_dict["x4"], input_dict["x5"], input_dict["x6"], input_dict["x7"], input_dict["x8"]
  Xinp = np.expand_dims(np.array([x1, x2, x3, x4, x5, x6, x7, x8]), axis=0)
  output = get_snd_outputs(Xinp)
  f = output[0][1].item()
  c = output[0][0].item()
  return {"f": f, "c": c}


def eval_function_constrained_intensity(input_dict: dict) -> dict:
  """
  Evaluates the SND function for input, returns the Intensity as 
  a constraint, and the Beam Position Error as the objective.
  """
  x1, x2, x3, x4, x5, x6, x7, x8 = input_dict["x1"], input_dict["x2"], input_dict["x3"], input_dict["x4"], input_dict["x5"], input_dict["x6"], input_dict["x7"], input_dict["x8"]
  Xinp = np.expand_dims(np.array([x1, x2, x3, x4, x5, x6, x7, x8]), axis=0)
  output = get_snd_outputs(Xinp)
  c = output[0][1].item()
  f = output[0][0].item()
  return {"f": f, "c": c}


def get_detailed_outputs(X_init, eval_function):
  """
  Takes in a dataframe of points to sample at, returns an Xopt object with the samplign points,
  the BPE and the intensity.
  """
  low, high = 0.0, 1.0
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
      objectives = {"f": "MINIMIZE"},
    )
  evaluator = Evaluator(function=eval_function)
  generator = ExpectedImprovementGenerator(
      vocs=vocs, turbo_controller="optimize"
  )
  X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
  X.evaluate_data(X_init)
  return X


def run_mobo(eval_function = eval_function_mobo, n_init: int=64, n_steps: int = 150):
  """
  Runs MOBO chain on the eval function with objectives, 
  with n_init initial samples followed
  by n_steps samples.
  Returns the Xopt object.
  """
  low = 0.0
  high = 1.0
  vocs = VOCS(
    variables = {"x1": [low, high],
                 "x2": [low, high],
                 "x3": [low, high],
                 "x4": [low, high],
                 "x5": [low, high],
                 "x6": [low, high],
                 "x7": [low, high],
                 "x8": [low, high]},
    objectives = {"f1": "MINIMIZE", "f2": "MAXIMIZE"},
  )
  gigo = np.random.rand(8)
  evaluator = Evaluator(function=eval_function)
  ref_point = eval_function({"x1": gigo[0], "x2": gigo[1], "x3": gigo[2],
                             "x4": gigo[3], "x5": gigo[4], "x6": gigo[5],
                             "x7": gigo[6], "x8": gigo[7]})
  generator = MOBOGenerator(vocs=vocs, reference_point= ref_point)
  generator.n_monte_carlo_samples = 512
  generator.numerical_optimizer.n_restarts = 80
  X = Xopt(generator=generator, evaluator=evaluator, vocs=vocs)
  X.random_evaluate(n_init)
  for i in range(n_steps):
    print(i)
    X.step()

  return X


def run_bo(eval_function, 
           objective:str="MAXIMIZE", 
           init_samples=64, 
           len_chain=150):
  """
  Runs BO chain on the eval function with objective, 
  with init_samples initial samples followed
  by len_chain samples.
  Returns the Xopt object.
  """
  low, high = 0.0, 1.0
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
      objectives = {"f": objective},
    )
  evaluator = Evaluator(function=eval_function)
  generator = ExpectedImprovementGenerator(vocs=vocs)
  X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
  X.random_evaluate(n_samples=init_samples)
  for i in range(len_chain):
    X.step()
  return X


def run_turbo(eval_function=eval_function_intensity, 
              objective:str="MAXIMIZE",
              init_samples=32, 
              len_chain=50):
  """
  Runs TurBO chain on the eval function with objective, 
  with init_samples initial samples followed by len_chain samples.
  Returns the Xopt object.
  """
  low, high = 0.0, 1.0
  n_init = init_samples
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
      objectives = {"f": objective},
    )
  evaluator = Evaluator(function=eval_function)
  generator = ExpectedImprovementGenerator(
      vocs=vocs, turbo_controller="optimize"
  )
  X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
  sampler = qmc.LatinHypercube(d=8)
  xs = sampler.random(n=n_init)
  init_samples = pd.DataFrame({f'x{i+1}': xs[:,i] for i in range(xs.shape[1])})
  X.evaluate_data(init_samples)
  
  for i in range(len_chain):
    if i % 10 == 0:
      print(f"Step: {i+1}")
    X.step()
  return X


def run_turbo_constrained(eval_function, 
                          X_init,
                          objective:str="MAXIMIZE",
                          len_chain=64):
  """
  Runs constrained TurBO on the eval function with objective, 
  with X_init as initial samples followed by len_chain samples.
  Returns the Xopt object.
  """
  low, high = 0.0, 1.0
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
      objectives = {"f": objective},
      constraints={"c": ["LESS_THAN", 10.0]},
    )
  evaluator = Evaluator(function=eval_function)
  generator = ExpectedImprovementGenerator(
      vocs=vocs, turbo_controller="safety"
  )
  X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
  X.evaluate_data(X_init)
  X.generator.train_model()
  X.generator.turbo_controller.update_state(X.generator.data)
  X.generator.turbo_controller.get_trust_region(X.generator.model)

  for i in range(len_chain):
    if i % 10 == 0:
      print(f"Step: {i+1}")
    X.step()
  return X


def run_bo_constrained(eval_function, 
                       X_init,
                       objective:str="MAXIMIZE",  
                      len_chain=100):
  """
  Runs constrained BO on the eval function with objective, 
  with X_init as initial samples followed by len_chain samples.
  Returns the Xopt object.
  """
  low, high = 0.0, 1.0
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
      objectives = {"f": objective},
      constraints={"c": ["LESS_THAN", 10.0]},
    )
  evaluator = Evaluator(function=eval_function)
  generator = ExpectedImprovementGenerator(vocs=vocs)
  X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
  X.evaluate_data(X_init)
  for i in range(len_chain):
    X.step()
  return X


def plot_results(X, objective="MAXIMIZE")->None:
  """
  Takes an Xopt object and the optimization goal. Plots
  the optimum as a line, and the samples as points.
  """
  y1 = X.generator.data["f"]
  if objective=="MAXIMIZE":
    y1_optimum = np.maximum.accumulate(y1)
  else: 
    y1_optimum = np.minimum.accumulate(y1)
  idx = np.arange(len(y1_optimum))
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 7))
  ax0.scatter(idx, y1)
  ax0.plot(idx, y1_optimum,'k')
  ax0.grid()
  ax0.set_xlabel("Sample Number")
  ax0.set_ylabel("Objective")
  ax1.plot(idx[-25:], y1_optimum[-25:],'k')
  ax1.plot(idx[-25:], y1[-25:],'ok')
  ax1.grid()
  ax1.set_ylabel("Objective");
  plt.show()


def plot_constrained(X, constrain:str="MINIMIZE", objective:str="MAXIMIZE")->None:
  """
  Takes an Xopt object with constraints, the optimization and constraining goal. Plots
  the optimum as a line, and the samples as points. Does the same for the constrained 
  feature.
  """
  c = X.generator.data["c"]
  c_mins = np.minimum.accumulate(c)
  f = X.generator.data["f"]
  f_opts = np.maximum.accumulate(f)
  idx = np.arange(len(c))

  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 7))
  ax0.scatter(idx, c)
  ax0.plot(idx, c_mins,'k')
  ax0.grid()
  ax0.set_xlabel("Sample Number")
  ax0.set_ylabel("Constraint")

  ax1.plot(idx, f_opts,'k')
  ax1.plot(idx, f,'ok')
  ax1.grid()
  ax1.set_ylabel("Objective");
  plt.show()


def run_turbo_bpe(eval_function, 
                  thetas: list=[0.75, 0.75, 0.75, 0.75],
                  objective:str="MINIMIZE",
                  n_init: int=16,
                  len_chain=64):
  low, high, eps = 0.0, 1.0, 1e-3
  vocs = VOCS(
      variables = {
                  "x1": [thetas[0], thetas[0]+eps],
                  "x2": [thetas[1], thetas[1]+eps],
                  "x3": [low, high],
                  "x4": [low, high],
                  "x5": [thetas[2], thetas[2]+eps],
                  "x6": [thetas[3], thetas[3]+eps],
                  "x7": [low, high],
                  "x8": [low, high]
                  },
      objectives = {"f": objective},
    )
  evaluator = Evaluator(function=eval_function)
  generator = ExpectedImprovementGenerator(
      vocs=vocs, turbo_controller="optimize"
  )
  X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
  X.random_evaluate(n_samples=n_init)
  for i in range(len_chain):
    if i % 10 == 0:
      print(f"Step: {i+1}")
    X.step()
  return X


def run_turbo_intensity(eval_function, 
                        chis: list=[0.75, 0.75, 0.75, 0.75],
                        objective:str="MAXIMIZE",
                        n_init: int=16,
                        len_chain=64):
  low, high, eps = 0.0, 1.0, 1e-3
  vocs = VOCS(
      variables = {
                  "x1": [low, high],
                  "x2": [low, high],
                  "x3": [chis[0], chis[0]+eps],
                  "x4": [chis[1], chis[1]+eps],
                  "x5": [low, high],
                  "x6": [low, high],
                  "x7": [chis[2], chis[2]+eps],
                  "x8": [chis[3], chis[3]+eps]
                  },
      objectives = {"f": objective},
    )
  evaluator = Evaluator(function=eval_function)
  generator = ExpectedImprovementGenerator(
      vocs=vocs, turbo_controller="optimize"
  )
  X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
  X.random_evaluate(n_samples=n_init)
  for i in range(len_chain):
    if i % 10 == 0:
      print(f"Step: {i+1}")
    X.step()
  return X
