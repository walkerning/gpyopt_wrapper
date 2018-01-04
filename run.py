# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import cPickle

from manager import Manager

parser = argparse.ArgumentParser()
parser.add_argument("module_file", help="The interface module file of your algorithm.")
parser.add_argument("--gpu-ids", default="0,1,2,3")
parser.add_argument("--max-exp", type=int, default=None)
parser.add_argument("--save-xy", default=None)

args = parser.parse_args()

gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
num_processes = len(gpu_ids)

manager = Manager(args.module_file, num_processes=num_processes, gpu_ids=gpu_ids, max_exp=args.max_exp)

xs, ys = manager.get_best()
print("Current found best 3 configuration:\n\t" + "\n\t".join(["{} : {}".format(x,y[0]) for x, y in zip(xs, ys)]))

if args.save_xy:
    print("Save xs and ys to {}".format(args.save_xy))
    with open(args.save_xy, "w") as f:
        cPickle.dump((manager.X, manager.Y), f, protocol=cPickle.HIGHEST_PROTOCOL)
