"""Profile synchronous worker barriers and transfer fixed-size images through shared RAM.

Uses SB3's pinned worker command protocol without changing action ordering,
automatic resets, terminal observations, or recurrent PPO rollout semantics.
"""

import ctypes
import multiprocessing as mp
import time
from multiprocessing.connection import wait

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _stack_obs, _worker

from smg_rl.diagnostics import StageProfiler


class WorkerConnection:
    def __init__(self, remote, pixels, statistics, index, shape):
        self.remote, self.statistics, self.index = remote, statistics, index
        self.images = np.frombuffer(pixels, dtype=np.uint8).reshape((-1, *shape)) if pixels is not None else None
        self.command = None
        self.started = 0.0

    def recv(self):
        result = self.remote.recv()
        self.command = result[0]
        self.started = time.perf_counter()
        return result

    def send(self, result):
        if self.command in ("step", "reset"):
            if self.images is not None:
                observation = dict(result[0])
                image = observation["image"]
                if image.dtype != np.uint8 or image.shape != self.images[self.index].shape:
                    raise ValueError("Worker image does not match the shared uint8 observation layout.")
                self.images[self.index] = image
                observation["image"] = None
                result = (observation, *result[1:])
            ready = time.perf_counter()
            self.statistics[self.index * 2] = self.started
            self.statistics[self.index * 2 + 1] = ready
        self.remote.send(result)

    def close(self):
        self.remote.close()


def measured_worker(remote, parent, factory, pixels, statistics, index, shape):
    # _worker retains SB3's reset/termination/env_method behavior and exceptions.
    _worker(WorkerConnection(remote, pixels, statistics, index, shape), parent, factory)


class ParentConnection:
    def __init__(self, remote, pixels, index, shape):
        self.remote, self.index = remote, index
        self.images = np.frombuffer(pixels, dtype=np.uint8).reshape((-1, *shape)) if pixels is not None else None

    def recv(self):
        result = self.remote.recv()
        if (self.images is not None and isinstance(result, tuple) and len(result) in (2, 5)
                and isinstance(result[0], dict) and result[0].get("image", False) is None):
            observation = dict(result[0])
            # Return an owned copy: subsequent worker writes must never mutate
            # SB3's previous observations or its terminal rollout history.
            observation["image"] = self.images[self.index].copy()
            result = (observation, *result[1:])
        return result

    def __getattr__(self, name):
        return getattr(self.remote, name)


class MeasuredSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, image_shape, shared=True, receive_order="ready", start_method="spawn"):
        if receive_order not in ("ready", "ordered"):
            raise ValueError("Unknown worker receive order.")
        self.waiting, self.closed = False, False
        self.profiler = StageProfiler()
        self.receive_order = receive_order
        self.last_statistics = {}
        self.vector_steps = 0
        self.compute_totals = np.zeros(len(env_fns), dtype=np.float64)
        self.idle_totals = np.zeros(len(env_fns), dtype=np.float64)
        self.bottleneck_counts = np.zeros(len(env_fns), dtype=np.int64)
        self.dispatched = 0.0
        context = mp.get_context(start_method)
        self.statistics = context.RawArray(ctypes.c_double, len(env_fns) * 2)
        self.pixels = context.RawArray(ctypes.c_ubyte, int(len(env_fns) * np.prod(image_shape))) if shared else None
        raw_remotes, self.work_remotes = zip(*[context.Pipe() for _ in env_fns])
        self.remotes = tuple(ParentConnection(remote, self.pixels, index, image_shape)
                             for index, remote in enumerate(raw_remotes))
        self.processes = []
        for index, (remote, work, factory) in enumerate(zip(raw_remotes, self.work_remotes, env_fns)):
            process = context.Process(target=measured_worker,
                                      args=(work, remote, CloudpickleWrapper(factory), self.pixels,
                                            self.statistics, index, image_shape), daemon=True)
            process.start()
            self.processes.append(process)
            work.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self.dispatched = time.perf_counter()
        with self.profiler.measure("dispatch"):
            super().step_async(actions)

    def step_wait(self):
        results = [None] * self.num_envs
        received = [0.0] * self.num_envs
        order = []
        pending = {remote: index for index, remote in enumerate(self.remotes)}
        with self.profiler.measure("barrier"):
            while pending:
                if self.receive_order == "ready":
                    with self.profiler.measure("wait_ready"):
                        ready = wait(list(pending))
                else:
                    ready = [next(iter(pending))]
                for remote in ready:
                    index = pending.pop(remote)
                    with self.profiler.measure("receive"):
                        results[index] = remote.recv()
                    received[index] = time.perf_counter()
                    order.append(index)
        self.waiting = False
        ready_times = [self.statistics[i * 2 + 1] for i in range(self.num_envs)]
        slowest = int(np.argmax(ready_times))
        self.last_statistics = {
            "compute_ms": [(ready_times[i] - self.statistics[i * 2]) * 1000 for i in range(self.num_envs)],
            "ready_after_dispatch_ms": [(t - self.dispatched) * 1000 for t in ready_times],
            "barrier_idle_ms": [(max(ready_times) - t) * 1000 for t in ready_times],
            "reply_transfer_and_receive_ms": [(received[i] - ready_times[i]) * 1000 for i in range(self.num_envs)],
            "slowest_worker": slowest, "receive_order": order,
            "shared_observations": self.pixels is not None,
            "note": "Synchronous on-policy barrier preserved. Idle/transfer intervals overlap; ordered receives include head-of-line waiting.",
        }
        self.vector_steps += 1
        self.compute_totals += self.last_statistics["compute_ms"]
        self.idle_totals += self.last_statistics["barrier_idle_ms"]
        self.bottleneck_counts[slowest] += 1
        observations, rewards, dones, infos, self.reset_infos = zip(*results)
        return _stack_obs(observations, self.observation_space), np.stack(rewards), np.stack(dones), infos

    def timing_snapshot(self):
        return {"stages": self.profiler.snapshot(), "last_barrier": self.last_statistics,
                "vector_steps": self.vector_steps,
                "mean_worker_compute_ms": (self.compute_totals / max(1, self.vector_steps)).tolist(),
                "mean_worker_barrier_idle_ms": (self.idle_totals / max(1, self.vector_steps)).tolist(),
                "bottleneck_count_per_worker": self.bottleneck_counts.tolist()}
