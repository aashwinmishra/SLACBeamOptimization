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
import argparse
from utils import eval_function, run_turbo, run_bo, get_optimum_details


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="TurBO")
parser.add_argument("--init_samples", type=int, default=64)
parser.add_argument("--len_chain", type=int, default=250)
args = parser.parse_args()

if args.model == "TurBO":
  X = run_turbo(eval_function=eval_function, init_samples=args.init_samples, len_chain=args.len_chain)
  get_optimum_details(X)
  X.generator.data.to_csv('Xopt_data.csv', index=False)
elif args.model == "BO":
  X = run_bo(eval_function=eval_function, init_samples=args.init_samples, len_chain=args.len_chain)
  get_optimum_details(X)
  X.generator.data.to_csv('Xopt_data.csv', index=False)

