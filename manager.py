# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import signal
import itertools
import multiprocessing
import Queue

import numpy as np
import GPyOpt
from GPyOpt.core.task.space import Design_space
from GPyOpt.experiment_design import initial_design
from GPyOpt.core.evaluators.batch_local_penalization import estimate_L

def pool_func(p_id, gpu_id, args, run, queue):
    ans = run(p_id, gpu_id, args)
    queue.put((p_id, gpu_id, args, ans))
    return ans

class Manager(object):
    def __init__(self, module_file, num_processes=2, gpu_ids=[0, 1], X_init=None, Y_init=None,
                 initial_design_type=None, initial_design_numdata=None, eps=None, max_exp=None):
        # TODO: move these configuration all into configuration module
        self.module_file = module_file
        self.module = {"__file__": module_file}
        execfile(module_file, self.module)
        assert "run" in self.module
        assert "domains_def" in self.module

        self.num_opt_arg = len(self.module["domains_def"])

        self.max_exp = max_exp or self.module.get("max_exp", 20)
        self.eps = eps or self.module.get("eps", 1e-2)
        self.normalize_Y = self.module.get("normalize_Y", True)

        self.num_processes = num_processes
        self.gpu_ids = gpu_ids
        self.gpus = itertools.cycle(gpu_ids)
        self.p_gpu_map = dict(list(zip(range(num_processes), self.gpus)))
        self.pid_p_map = {}
        self.pid_x_map = {}

        # initial design
        self.pool = None
        self.initial_design_type  = initial_design_type or self.module.get("initial_design_type", "random")
        self.initial_design_numdata = initial_design_numdata or self.module.get("initial_design_numdata", 5)

        self.X = X_init
        self.Y = Y_init
        self.domain = self.module["domains_def"]
        self.constraints = None
        self.space = Design_space(self.domain, self.constraints)
        self.ans_queue = multiprocessing.Queue(maxsize=num_processes)

        self.worker = self._get_worker()

        self.num_start_exp = 0
        self.handle_signal()

    def start(self):
        if self.Y is None:
            if "Y_init" in self.module and "X_init" in self.module:
                self.X = self.module["X_init"]
                self.Y = self.module["Y_init"]
            else:
                if "X_init" in self.module:
                    self.X = self.module["X_init"]
                elif self.X is None:
                    self.X = initial_design(self.initial_design_type, self.space, self.initial_design_numdata)
                self.Y = self.run_initial(self.X) # FIXME: wait for all thest to finish

        assert len(self.X) == len(self.Y)
        print("Initial: {} points. Max exp num: {}.".format(len(self.X), self.max_exp))

        opt = GPyOpt.methods.BayesianOptimization(f=None, domain=self.domain,
                                                  constraints=self.constraints, cost_withGradients=None, model_type='GP',
                                                  X=self.X, Y=self.Y, acquisition_type='EI',
                                                  acquisition_optimizer_type='lbfgs',
                                                  evaluator_type='local_penalization',
                                                  batch_size=self.num_processes)
        self.init_opt(opt)
        first_batch_X = opt.suggest_next_locations()
        self.run_batch(first_batch_X)
        self.wait_and_run()

    def get_best(self, n=3):
        inds = np.argsort(np.squeeze(self.Y))[:n]
        return self.X[inds], self.Y[inds]

    def init_opt(self, opt):
        opt.context = None
        opt.num_acquisitions = 0

    def handle_signal(self):
        def signal_handler(sig, frame):
            print("Receive sigint, stop sub processes...")
            for pid, p in self.pid_p_map.iteritems():
                if p is not None:
                    print("Stoping process #{}...".format(pid))
                    p.terminate()
                    self.pid_p_map[pid] = None
            print("Stop sub processses finished.")
            xs, ys = self.get_best()
            print("Current found best 3 configuration:\n\t" + "\n\t".join(["{} : {}".format(x,y[0]) for x, y in zip(xs, ys)]))
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

    def _get_worker(self):
        def func(t_id, p_id, gpu_id, args):
            ans = self.module["run"](p_id, gpu_id, args)
            self.ans_queue.put((t_id, p_id, gpu_id, args, ans))
            return ans
        return func

    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        return np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))

    def run_initial(self, X):
        print("Start running {} initial tasks".format(len(X)))
        first_batch = X[:self.num_processes]
        Y = np.zeros((0, 1))
        ind = self.num_processes
        self.run_batch(first_batch, prefix="[INITIAL] ")
        while 1:
            t_id, p_id, _, args, ans = self.ans_queue.get()
            gpu_id = self.p_gpu_map[p_id]
            Y = np.vstack((Y, ans))        
            self.pid_p_map[p_id] = None
            self.pid_x_map[p_id] = None
            print("[INITIAL] Process #{} (gpu {}): Finish task #{} x={}; y={}".format(p_id, gpu_id, t_id, args, ans))
            if ind >= len(X):
                if len([x for x in self.pid_x_map.itervalues() if x is not None]) == 0:
                    break
                continue
            new_args = X[ind]
            ind += 1
            t_id = self.num_start_exp + 1
            print("[INITIAL] Process #{} (gpu {}): Start task #{} x={}".format(p_id, gpu_id, t_id, new_args))
            p = multiprocessing.Process(target=self.worker, args=(t_id, p_id, gpu_id, new_args))
            self.pid_p_map[p_id] = p
            self.pid_x_map[p_id] = new_args
            self.num_start_exp += 1
            p.start()

        print("Finish running {} initial tasks.".format(len(X)))
        return Y

    def run_batch(self, X, prefix=""):
        assert len(X) <= self.num_processes
        for p_id, x in zip(range(self.num_processes), X):
            gpu_id = self.p_gpu_map[p_id]
            # FIXME: signal handling not correct
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            t_id = self.num_start_exp + 1
            p = multiprocessing.Process(target=self.worker, args=(t_id, p_id, gpu_id, x))
            signal.signal(signal.SIGINT, original_sigint_handler)
            self.pid_p_map[p_id] = p
            self.pid_x_map[p_id] = x
            print("{}Process #{} (gpu {}): Start task #{} x={}".format(prefix, p_id, gpu_id, self.num_start_exp + 1, x))
            p.start()
            self.num_start_exp += 1

    def wait_and_run(self):
        while 1: # FIXME: do not entrance signal handler multiple time...
            # TODO: get multiple answer at the same time
            t_id, p_id, _, args, ans = self.ans_queue.get()
            gpu_id = self.p_gpu_map[p_id]
            self.X = np.vstack((self.X, args))
            self.Y = np.vstack((self.Y, ans))
            self.pid_p_map[p_id] = None
            self.pid_x_map[p_id] = None
            print("Process #{} (gpu {}): Finish task #{} x={}; y={}".format(p_id, gpu_id, t_id, args, ans))
            if (self.num_start_exp >= self.max_exp) or (self._distance_last_evaluations() < self.eps):
                # do not acuision new
                if len([x for x in self.pid_x_map.itervalues() if x is not None]) == 0:
                    break
                continue
            opt = GPyOpt.methods.BayesianOptimization(f=None, domain=self.domain,
                                                      constraints=self.constraints, cost_withGradients=None, model_type='GP',
                                                      X=self.X, Y=self.Y, acquisition_type='EI',
                                                      normalize_Y=self.normalize_Y, exact_feval=False, acquisition_optimizer_type='lbfgs',
                                                      model_update_interval=1, evaluator_type='local_penalization',
                                                      batch_size=2, num_cores=1)
            new_args = self.compute_next(opt)[0]
            print("Process #{} (gpu {}): Start task #{} x={}".format(p_id, gpu_id, self.num_start_exp + 1, new_args))
            t_id = self.num_start_exp + 1
            p = multiprocessing.Process(target=self.worker, args=(t_id, p_id, gpu_id, new_args))
            self.pid_p_map[p_id] = p
            self.pid_x_map[p_id] = new_args
            self.num_start_exp += 1
            p.start()

        for pid, p in self.pid_p_map.iteritems():
            if p is not None:
                p.join()
            self.pid_p_map[pid] = None

        x, y = self.get_nowait()
        self.X = np.vstack((self.X, x))
        self.Y = np.vstack((self.Y, y))

    def get_nowait(self):
        args_lst = []
        ans_lst = []
        while 1:
            try:
                p_id, _, args, ans = self.ans_queue.get_nowait()
                args_lst.append(args)
                ans_lst.append(ans)
            except Queue.Empty as e:
                break
        if len(args_lst) == 0:
            return np.zeros((0, self.num_opt_arg)), np.zeros((0, 1))
        return np.stack(args_lst, axis=0), np.stack(ans_lst, axis=0)

    def compute_next(self, opt, num=1):
        """
        Computes the next elements.
        """

        assert num >= 1

        opt.model_parameters_iterations = None
        opt.num_acquisitions = 0
        opt.context = None
        opt._update_model(opt.normalization_type) # call update model first

        evaluator = opt.evaluator
        from GPyOpt.acquisitions import AcquisitionLP
        assert isinstance(evaluator.acquisition, AcquisitionLP)
        evaluator.acquisition.update_batches(None,None,None)

        if self.num_processes > 1:
            L = estimate_L(evaluator.acquisition.model.model, evaluator.acquisition.space.get_bounds())
            Min = evaluator.acquisition.model.model.Y.min()
            running_xs = [x for x in self.pid_x_map.itervalues() if x is not None]
            X_batch = np.stack(running_xs) if len(running_xs) > 0 else np.zeros((0, self.num_opt_arg)) # current running xs
            evaluator.acquisition.update_batches(X_batch,L,Min)
        else:
            X_batch = np.zeros((0, self.num_opt_arg))
        cur_len = len(X_batch)
        k = 0
        while 1:
            new_sample = evaluator.acquisition.optimize()[0]
            X_batch = np.vstack((X_batch, new_sample))
            k += 1
            if k >= num:
                break
            evaluator.acquisition.update_batches(X_batch, L, Min)
        
        evaluator.acquisition.update_batches(None,None,None)
        return X_batch[cur_len:]

    def plot_convergence(self, filename=None):
        opt = GPyOpt.methods.BayesianOptimization(f=None, domain=self.domain,
                                                  constraints=self.constraints, cost_withGradients=None, model_type='GP',
                                                  X=self.X, Y=self.Y, acquisition_type='EI',
                                                  normalize_Y=self.normalize_Y, exact_feval=False, acquisition_optimizer_type='lbfgs',
                                                  model_update_interval=1, evaluator_type='local_penalization',
                                                  batch_size=2, num_cores=1)
        opt._compute_results()
        return opt.plot_convergence(filename=filename)

    def plot_acquisition(self,filename=None):
        X = self.X[np.where(self.Y < np.inf)[0]]
        Y = self.Y[np.where(self.Y < np.inf)[0]]
        opt = GPyOpt.methods.BayesianOptimization(f=None, domain=self.domain,
                                                  constraints=self.constraints, cost_withGradients=None, model_type='GP',
                                                  X=X, Y=Y, acquisition_type='EI',
                                                  normalize_Y=self.normalize_Y, exact_feval=False, acquisition_optimizer_type='lbfgs',
                                                  model_update_interval=1, evaluator_type='local_penalization',
                                                  batch_size=2, num_cores=1)
        opt._compute_results()
        return opt.plot_acquisition(filename=filename)
