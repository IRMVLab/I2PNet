
import os
from pathlib import Path

import argparse

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', required=True, help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--target', required=True, help='Dump dir to save model checkpoint [default: log]')
FLAGS = parser.parse_args()

LOGDIR = FLAGS.log_dir
TAG = FLAGS.target

class Evaluator(object):
    def __init__(self):

        save_path = Path(LOGDIR) / "info_test"
        self.result_path = str(save_path/"analysis")
        os.makedirs(self.result_path,exist_ok=True)

        self.name = TAG.split('.')[0]
        result = np.load(str(save_path/TAG))
        self.RRE = result["RRE"]
        self.RTE = result["RTE"]

    def analysis(self):
        # recall
        plt.figure(0)
        RTE_range = np.arange(0, 15, 0.5)
        RTE_ratio = []
        for i in RTE_range:
            RTE_ratio.append(np.sum(self.RTE < i) / np.shape(self.RTE)[0])
        RTE_ratio = np.array(RTE_ratio)
        plt.plot(RTE_range, RTE_ratio)
        plt.xlim([0,15.1])
        plt.ylim([0,1])
        plt.xticks(np.arange(0, 15, 2.5))
        plt.savefig(os.path.join(self.result_path,'RTE_ratio.png'))

        plt.figure(1)
        RRE_range = np.arange(0, 30, 1)
        RRE_ratio = []
        for i in RRE_range:
            RRE_ratio.append(np.sum(self.RRE < i) / np.shape(self.RRE)[0])
        RRE_ratio = np.array(RRE_ratio)
        plt.plot(RRE_range, RRE_ratio)
        plt.xlim([0,30])
        plt.ylim([0,1])
        plt.xticks(np.arange(0, 30, 5))
        plt.savefig(os.path.join(self.result_path,'RRE_ratio.png'))

        mask = np.logical_and(self.RRE<10,self.RTE<5)
        # print(f"{np.mean(self.RRE[mask]):3f}+-{np.std(self.RRE[mask]):3f}")
        # print(f"{np.mean(self.RTE[mask]):3f}+-{np.std(self.RTE[mask]):3f}")
        # print(f"{mask.sum()/len(mask):3f}")

        # print(f"{np.mean(self.RRE):3f}+-{np.std(self.RRE):3f}")
        # print(f"{np.mean(self.RTE):3f}+-{np.std(self.RTE):3f}")
        # print(f"{mask.sum()/len(mask):3f}")\
        print(f"{np.mean(self.RRE[mask]):.2f}+-{np.std(self.RRE[mask]):.2f}")
        print(f"{np.mean(self.RTE[mask]):.2f}+-{np.std(self.RTE[mask]):.2f}")
        print(f"{mask.sum()/len(mask)*100:.2f}")

        print(f"{np.mean(self.RRE):.2f}+-{np.std(self.RRE):.2f}")
        print(f"{np.mean(self.RTE):.2f}+-{np.std(self.RTE):.2f}")
        print(f"{mask.sum()/len(mask)*100:.2f}")

        plt.figure(2)
        plt.hist(self.RTE, bins=np.arange(0, 15, 0.5), weights=np.ones(self.RTE.shape[0]) / self.RTE.shape[0])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.title('hist RTE')
        plt.savefig(os.path.join(self.result_path,'RTE.png'))

        plt.figure(3)
        plt.hist(self.RRE, bins=np.arange(0, 30, 1), weights=np.ones(self.RRE.shape[0]) / self.RRE.shape[0])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.title('hist RRE')
        plt.savefig(os.path.join(self.result_path,'RRE.png'))



if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.analysis()
