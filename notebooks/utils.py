import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lcls_beamline_toolbox.xraywavetrace.beam1d as beam
import lcls_beamline_toolbox.xraywavetrace.optics1d as optics
import lcls_beamline_toolbox.xraywavetrace.beamline1d as beamline
import scipy.optimize as optimize
import copy
import scipy.spatial.transform as transform
from scipy.stats import qmc
from split_and_delay import SND
import torch
import gpytorch
import botorch
import math
import warnings
warnings.filterwarnings("ignore")
from xopt import Xopt, Evaluator
from xopt.generators.bayesian import MOBOGenerator
from xopt.generators.bayesian import ExpectedImprovementGenerator, UpperConfidenceBoundGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
from xopt import VOCS
from xopt import Xopt


def get_snd_outputs(inputs):
  """
  Takes an [n, 8] dim np array of
  [samples, ( t1_th1, t1_th2, chi1, chi2, t4_th1, t4_th2, chi1, chi2)].
  The entries lie in the uniform unit interval.
  They are scaled to lie in [-100e-6, +100e-6].
  Returns a torch tensor of dim [n, ] of the objective function at these
  sample locations.
  PLEASE note the upper and lower bounds of +100e-6 and -100e-6 used in scaling!
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
    my_intensity *= (1.0 + np.random.rand() * 0.01)

    do_centroid = snd.get_IP_r()
    do_centroid *= (1.0 + np.random.rand() * 0.01)
    result.append( (np.log(do_centroid*1e6) - 2*np.log(my_intensity)).item())

  return torch.tensor(result, dtype=torch.float)


def get_snd_outputs_detailed(inputs):
  """
  Takes the X opt features, returns the values of the Beam Position Error and 
  Intensity at each sample.
  PLEASE note the upper and lower bounds of +100e-6 and -100e-6 used in scaling!
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
    do_centroid_x = snd.get_IP_cx()
    do_centroid_y = snd.get_IP_cy()

    result.append([do_centroid*1e6, my_intensity])

  return torch.tensor(result, dtype=torch.float)


def eval_function(input_dict: dict) -> dict:
  x1, x2, x3, x4, x5, x6, x7, x8 = input_dict["x1"], input_dict["x2"], input_dict["x3"], input_dict["x4"], input_dict["x5"], input_dict["x6"], input_dict["x7"], input_dict["x8"]
  Xinp = np.expand_dims(np.array([x1, x2, x3, x4, x5, x6, x7, x8]), axis=0)
  output = get_snd_outputs(Xinp).squeeze()
  f = output.item()
  return {"f": f}


def run_bo(eval_function=eval_function, init_samples=64, len_chain=150):
  """
  Runs Bayesian Optimization chain on the eval function, with init_samples initial samples followed 
  by len_chain samples.
  Returns the Xopt object.
  """
  low = 0.0
  high = 1.0
  n_init = 64
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
  generator = ExpectedImprovementGenerator(vocs=vocs)
  X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)
  X.random_evaluate(init_samples)
  for i in range(len_chain):
    X.step()
  return X


def run_turbo(eval_function=eval_function, init_samples=64, len_chain=150):
  """
  Runs TurBO chain on the eval function, with init_samples initial samples followed 
  by len_chain samples.
  Returns the Xopt object.
  """
  low = 0.0
  high = 1.0
  n_init = 64
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
  X.generator.turbo_controller.scale_factor = 1.5
  X.generator.turbo_controller.success_tolerance = 10
  X.generator.turbo_controller.failure_tolerance = 10
  X.generator.turbo_controller.length = 0.6

  sampler = qmc.LatinHypercube(d=8)
  xs = sampler.random(n=n_init)
  init_samples = pd.DataFrame({f'x{i+1}': xs[:,i] for i in range(xs.shape[1])})
  X.evaluate_data(init_samples)

  X.generator.train_model()
  X.generator.turbo_controller.update_state(X.generator.data)
  X.generator.turbo_controller.get_trust_region(X.generator.model)

  for i in range(len_chain):
    # if i % 10 == 0:
    #   print(f"Step: {i+1}")
    model = X.generator.train_model()
    trust_region = X.generator.turbo_controller.get_trust_region(generator.model)\
          .squeeze()
    scale_factor = X.generator.turbo_controller.length
    region_width = trust_region[1] - trust_region[0]
    best_value = X.generator.turbo_controller.best_value

    n_successes = X.generator.turbo_controller.success_counter
    n_failures = X.generator.turbo_controller.failure_counter

    acq = X.generator.get_acquisition(model)

    X.step()

  X.generator.turbo_controller

  return X


def get_optimum_details(X, plot=True):
  """
  Takes Xopt object, finds the sample with the minimum and prints the settings 
  of this sample, the beam position error and the inensity at this setting.
  """
  min_idx = np.argmin(X.generator.data["f"])
  min_val = X.generator.data["f"][min_idx]
  X_min = [X.generator.data["x1"][min_idx],
          X.generator.data["x2"][min_idx],
            X.generator.data["x3"][min_idx],
          X.generator.data["x4"][min_idx],
          X.generator.data["x5"][min_idx],
          X.generator.data["x6"][min_idx],
            X.generator.data["x7"][min_idx],
            X.generator.data["x8"][min_idx]
          ]
  inputs = np.array(X_min)
  inputs = inputs[np.newaxis,:]
  outs = get_snd_outputs_detailed(inputs)
  print("Optimum Inputs: ", X_min)
  print("BPE: ", outs[0][0].item())
  print("Intensity:", outs[0][1].item())

  if plot:
    y1 = X.generator.data["f"]
    y1_mins = np.minimum.accumulate(y1)

    idx = np.arange(len(y1_mins))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 7))
    ax0.plot(idx, y1_mins,'k')
    ax0.grid()
    ax0.set_xlabel("Sample Number")
    ax0.set_ylabel("Objective")
    ax1.plot(idx[-50:], y1_mins[-50:],'k')
    ax1.grid()
    ax1.set_ylabel("Objective");
    plt.savefig('objective.pdf', bbox_inches='tight')

