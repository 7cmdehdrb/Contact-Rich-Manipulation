"""CPU bookkeeping for bounded, periodically refreshed group initial states."""
import random


class GroupedResetPool:
    def __init__(self, groups, capacity, refresh_interval, seed):
        if min(groups, capacity, refresh_interval) < 1:
            raise ValueError('Reset pool settings must be positive integers')
        self.entries = [[] for _ in range(groups)]
        self.calls = [0] * groups
        self.replace = [0] * groups
        self.capacity = capacity
        self.refresh_interval = refresh_interval
        self.rng = random.Random(seed)
        self.key = None
        self.generated = 0
        self.reused = 0

    def select(self, group, key):
        if key != self.key:
            self.entries = [[] for _ in self.entries]
            self.calls = [0] * len(self.entries)
            self.replace = [0] * len(self.entries)
            self.key = key
        entries = self.entries[group]
        self.calls[group] += 1
        if len(entries) < self.capacity:
            entries.append(None)
            slot, fresh = len(entries) - 1, True
        elif (self.calls[group] - self.capacity) % self.refresh_interval == 0:
            slot, fresh = self.replace[group], True
            self.replace[group] = (slot + 1) % self.capacity
        else:
            slot, fresh = self.rng.randrange(len(entries)), False
        self.generated += int(fresh)
        self.reused += int(not fresh)
        return slot, fresh
