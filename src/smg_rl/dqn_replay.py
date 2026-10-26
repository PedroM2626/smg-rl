"""Bounded frame-sharing prioritized replay with worker-local n-step boundaries."""

import json
from collections import deque
from copy import deepcopy

import numpy as np


class PrioritizedFrames:
    def __init__(self, capacity, image_size, stack=4, workers=1, n_step=3, gamma=.997,
                 alpha=.5, seed=42):
        if (min(capacity, image_size, stack, workers, n_step) < 1 or not 0 < gamma <= 1
                or not 0 <= alpha <= 1):
            raise ValueError('Invalid frame replay dimensions/discount/priority exponent.')
        self.capacity, self.image_size, self.stack = capacity, image_size, stack
        self.workers, self.n_step, self.gamma, self.alpha = workers, n_step, gamma, alpha
        # Each decision creates one successor and, at most, one reset frame.
        # Extra slots protect active stacks and not-yet-emitted n-step prefixes.
        self.frame_capacity = 2 * capacity + 2 * workers * (n_step + stack) + 16
        self.frames = np.empty((self.frame_capacity, image_size, image_size), np.uint8)
        self.frame_ids = np.full(self.frame_capacity, -1, np.int64)
        self.previous = np.full(self.frame_capacity, -1, np.int64)
        self.arrays = {key: np.zeros(capacity, dtype) for key, dtype in {
            'state': np.int64, 'next': np.int64, 'action': np.int64, 'scenario': np.int64,
            'reward': np.float32, 'discount': np.float32, 'horizon': np.int16,
            'terminal': np.bool_, 'assisted': np.bool_}.items()}
        self.base = 1 << (capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.base, np.float64)
        self.minimum = np.full(2 * self.base, np.inf, np.float64)
        self.next_frame, self.position, self.size, self.emitted, self.max_priority = 0, 0, 0, 0, 1.
        self.rng = np.random.default_rng(seed)
        self.pending = [deque() for _ in range(workers)]

    @property
    def bytes_allocated(self):
        return self.frames.nbytes + self.frame_ids.nbytes + self.previous.nbytes + sum(
            a.nbytes for a in self.arrays.values()) + self.tree.nbytes + self.minimum.nbytes

    def add_frame(self, image, previous=None):
        image = np.asarray(image)
        if image.shape != (self.image_size, self.image_size) or image.dtype != np.uint8:
            raise ValueError('Replay stores exact cropped uint8 frames.')
        if previous is not None:
            self._slot(previous)
        identity = self.next_frame
        slot = identity % self.frame_capacity
        self.frames[slot], self.frame_ids[slot] = image, identity
        self.previous[slot] = -1 if previous is None else previous
        self.next_frame += 1
        return identity

    def _slot(self, identity):
        slot = int(identity) % self.frame_capacity
        if identity < 0 or self.frame_ids[slot] != identity:
            raise ValueError('Stale frame generation; reset rather than crossing overwritten history.')
        return slot

    def observation(self, identity):
        frames = []
        for _ in range(self.stack):
            slot = self._slot(identity)
            frames.append(self.frames[slot])
            previous = int(self.previous[slot])
            if previous >= 0:
                identity = previous
        return np.stack(frames[::-1])

    def _priority(self, index, value):
        weight = value ** self.alpha
        node = self.base + int(index)
        self.tree[node], self.minimum[node] = weight, weight
        while node > 1:
            node //= 2
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
            self.minimum[node] = min(self.minimum[2 * node], self.minimum[2 * node + 1])

    def priorities(self, indices, values):
        # Repeated samples receive their largest proposed priority, not update-order noise.
        grouped = {}
        for index, value in zip(indices, values):
            if not 0 <= index < self.size or not np.isfinite(value) or value <= 0:
                raise ValueError('Priorities must be finite positive values for valid entries.')
            grouped[int(index)] = max(grouped.get(int(index), 0.), float(value))
        for index, value in grouped.items():
            self._priority(index, value)
            self.max_priority = max(self.max_priority, value)

    def append(self, worker, state, action, reward, successor, scenario, terminal, ended, assisted=False):
        if not 0 <= worker < self.workers or terminal and not ended or not np.isfinite(reward):
            raise ValueError('Invalid real replay transition/boundary.')
        self._slot(state)
        self._slot(successor)
        queue = self.pending[worker]
        queue.append({'state': state, 'action': action, 'reward': reward, 'next': successor,
                      'scenario': scenario, 'terminal': bool(terminal), 'assisted': bool(assisted)})
        if ended:
            while queue:
                self._emit(queue)
        elif len(queue) >= self.n_step:
            self._emit(queue)

    def _emit(self, queue):
        items = list(queue)[:self.n_step]
        first, last = items[0], items[-1]
        self.observation(first['state'])
        self.observation(last['next'])
        row = {**first, 'next': last['next'],
               'reward': sum(self.gamma ** i * item['reward'] for i, item in enumerate(items)),
               'discount': 0. if last['terminal'] else self.gamma ** len(items),
               'horizon': len(items), 'terminal': last['terminal']}
        for key, array in self.arrays.items():
            array[self.position] = row[key]
        self._priority(self.position, self.max_priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.emitted += 1
        queue.popleft()

    def flush_cuts(self):
        """Collector interruption retains valid bootstrap and never invents death."""
        for queue in self.pending:
            while queue:
                self._emit(queue)

    def sample(self, batch_size, beta):
        if self.size == 0 or batch_size < 1 or not 0 <= beta <= 1:
            raise ValueError('Need nonempty replay, positive batch and valid importance exponent.')
        points = (np.arange(batch_size) + self.rng.random(batch_size)) * self.tree[1] / batch_size
        indices = []
        for value in points:
            node = 1
            while node < self.base:
                if value < self.tree[2 * node]:
                    node *= 2
                else:
                    value -= self.tree[2 * node]
                    node = 2 * node + 1
            indices.append(node - self.base)
        indices = np.asarray(indices)
        probability = self.tree[self.base + indices] / self.tree[1]
        minimum = self.minimum[1] / self.tree[1]
        weights = (probability / minimum) ** (-beta)
        return {'images': np.stack([self.observation(i) for i in self.arrays['state'][indices]]),
                'next_images': np.stack([self.observation(i) for i in self.arrays['next'][indices]]),
                **{k: v[indices] for k, v in self.arrays.items()},
                'weights': weights.astype(np.float32), 'indices': indices}

    def save(self, path):
        if any(self.pending):
            raise ValueError('Flush collector cuts before publishing a resumable replay.')
        metadata = {'schema': 1, 'capacity': self.capacity, 'image_size': self.image_size,
            'stack': self.stack, 'workers': self.workers, 'n_step': self.n_step, 'gamma': self.gamma,
            'alpha': self.alpha, 'next_frame': self.next_frame, 'position': self.position,
            'size': self.size, 'emitted': self.emitted, 'max_priority': self.max_priority,
            'rng': self.rng.bit_generator.state}
        used = min(self.next_frame, self.frame_capacity)
        with open(path, 'xb') as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)),
                frames=self.frames[:used], frame_ids=self.frame_ids[:used], previous=self.previous[:used],
                tree=self.tree, minimum=self.minimum, **self.arrays)

    def expanded_workers(self, workers):
        """Explicitly grow a flushed collector layout without changing experience.

        Frame IDs survive ring wrap; copying physical offsets into a larger ring
        would corrupt their modulo mapping. Remap every retained generation and
        its previous-frame ID. Model/replay counters, priorities and RNG survive.
        """
        if type(workers) is not int or workers <= self.workers or any(self.pending):
            raise ValueError('Worker expansion requires a larger integer count and flushed collector cuts.')
        result = type(self)(self.capacity, self.image_size, self.stack, workers,
                            self.n_step, self.gamma, self.alpha)
        slots = np.flatnonzero(self.frame_ids >= 0)
        targets = self.frame_ids[slots] % result.frame_capacity
        if len(np.unique(targets)) != len(targets):
            raise ValueError('Retained frame generations collide in the expanded ring.')
        for start in range(0, len(slots), 4096):
            source, target = slots[start:start + 4096], targets[start:start + 4096]
            result.frames[target] = self.frames[source]
            result.frame_ids[target] = self.frame_ids[source]
            result.previous[target] = self.previous[source]
        for key, array in self.arrays.items():
            result.arrays[key][:] = array
        result.tree[:], result.minimum[:] = self.tree, self.minimum
        for key in ('next_frame', 'position', 'size', 'emitted', 'max_priority'):
            setattr(result, key, getattr(self, key))
        result.rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        return result

    @classmethod
    def load(cls, path, expected):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            required = ('capacity', 'image_size', 'stack', 'workers', 'n_step', 'gamma', 'alpha')
            if meta.get('schema') != 1 or any(meta[k] != expected[k] for k in required):
                raise ValueError('Replay capacity/control/schema differs from this checkpoint.')
            result = cls(**{k: meta[k] for k in required})
            used = min(meta['next_frame'], result.frame_capacity)
            for key in ('frames', 'frame_ids', 'previous'):
                source, target = archive[key], getattr(result, key)
                if source.shape != target[:used].shape or source.dtype != target.dtype:
                    raise ValueError('Invalid frame replay array layout.')
                target[:used] = source
            for key, target in {**result.arrays, 'tree': result.tree, 'minimum': result.minimum}.items():
                source = archive[key]
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError('Invalid transition replay array layout.')
                target[:] = source
            for key in ('next_frame', 'position', 'size', 'emitted', 'max_priority'):
                setattr(result, key, meta[key])
            if not 0 <= result.size <= result.capacity or not 0 <= result.position < result.capacity:
                raise ValueError('Invalid replay ring metadata.')
            result.rng.bit_generator.state = meta['rng']
            return result
