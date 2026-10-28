"""Opt-in batch occupancy and queue measurements; never changes scheduling.

Batch occupancy is elapsed simulated batch time, including communication and
memory waits. It is not GPU SM utilization. This observer supports colocated
PP=1 replicas, whose batches do not overlap within one replica.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


class LoadObserver:
    def __init__(self, path, num_instances, window_start_ns, window_end_ns,
                 interval_ns=1_000_000_000):
        if not 0 <= window_start_ns < window_end_ns or interval_ns <= 0:
            raise ValueError("load window must have 0 <= start < end and interval > 0")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.num_instances = num_instances
        self.start = window_start_ns
        self.end = window_end_ns
        self.interval = interval_ns
        self.active = {}
        self.busy_ns = [0] * num_instances
        self.bin_busy = defaultdict(int)
        self.boundary_active = Counter()
        self.last_end = [0] * num_instances
        self.batch_sizes = Counter()
        self.batches = 0
        self.next_sample = 0
        self.samples = []

    def begin(self, instance, batch):
        key = (instance, batch.batch_id)
        if key in self.active:
            raise RuntimeError("batch observed twice")
        self.active[key] = (batch.batch_time, len(batch.requests))

    def complete(self, instance, batch, finish):
        begin, size = self.active.pop((instance, batch.batch_id))
        if finish < begin or begin < self.last_end[instance]:
            raise RuntimeError("load observer requires non-overlapping PP=1 batches")
        self.last_end[instance] = finish
        left, right = max(begin, self.start), min(finish, self.end)
        if right <= left:
            return
        self.busy_ns[instance] += right - left
        self.batch_sizes[size] += 1
        self.batches += 1
        first, last = (left - self.start) // self.interval, (right - self.start - 1) // self.interval
        for index in range(first, last + 1):
            boundary = self.start + index * self.interval
            self.bin_busy[index] += min(right, boundary + self.interval) - max(left, boundary)
            # Exact count at each bin's left boundary; not an unsampled peak.
            if begin <= boundary < finish:
                self.boundary_active[index] += 1

    def sample(self, current, schedulers):
        if current < self.next_sample:
            return
        self.next_sample = current + self.interval
        self.samples.append((current, sum(len(s.running) for s in schedulers),
                             sum(sum(r.arrival <= current for r in s.waiting)
                                 for s in schedulers)))
        # Small live checkpoint for calibration. In-flight intervals are not
        # included until they complete, so this is explicitly a lower bound.
        elapsed = min(current, self.end) - self.start
        if elapsed > 0:
            progress = {"simulated_time_ns": current,
                        "completed_interval_busy_fraction_lower_bound":
                            sum(self.busy_ns) / (self.num_instances * elapsed),
                        "running_requests": self.samples[-1][1],
                        "waiting_requests": self.samples[-1][2],
                        "batches_overlapping_window": self.batches}
            progress_path = self.path.with_suffix(".progress.json")
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress) + "\n")
            temporary.replace(progress_path)

    def finish(self, simulation_end_ns):
        if self.active:
            raise RuntimeError("simulation ended with observed batches in flight")
        actual_end = min(self.end, simulation_end_ns)
        duration = max(0, actual_end - self.start)
        bins = []
        for index in range((duration + self.interval - 1) // self.interval):
            begin = self.start + index * self.interval
            width = min(self.interval, actual_end - begin)
            busy = self.bin_busy[index]
            bins.append({"start_ns": begin, "end_ns": begin + width,
                         "mean_active_replicas": busy / width,
                         "active_replicas_at_start": self.boundary_active[index]})
        with self.path.with_suffix(".queues.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("simulated_time_ns", "running_requests", "arrived_waiting_requests"))
            writer.writerows(self.samples)
        occupancies = [busy / duration if duration else 0 for busy in self.busy_ns]
        window_samples = [row for row in self.samples if self.start <= row[0] < actual_end]
        queues = [row[2] for row in window_samples]
        total_batch_requests = sum(size * count for size, count in self.batch_sizes.items())
        return {"schema_version": 1, "definition": "Colocated PP=1 replica batch-busy fraction; includes memory/communication stalls, not SM utilization",
                "requested_window_ns": [self.start, self.end],
                "observed_window_ns": [self.start, actual_end],
                "window_complete": simulation_end_ns >= self.end,
                "replicas": self.num_instances,
                "mean_busy_fraction": sum(occupancies) / self.num_instances,
                "per_replica_busy_fraction": occupancies,
                "replicas_with_work_in_window": sum(busy > 0 for busy in self.busy_ns),
                "sampled_peak_active_replicas": max(self.boundary_active.values(), default=0),
                "batches_overlapping_window": self.batches,
                "mean_batch_size": total_batch_requests / self.batches if self.batches else 0,
                "batch_size_histogram": dict(sorted(self.batch_sizes.items())),
                "queue_sample_count": len(queues),
                "mean_sampled_waiting_requests": sum(queues) / len(queues) if queues else None,
                "max_sampled_waiting_requests": max(queues, default=0),
                "bins": bins}
