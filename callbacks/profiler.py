import os
import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.tensorboard import SummaryWriter
from .base_callbacks import BaseCallback


class ProfilerCallback(BaseCallback):
    def __init__(
        self,
        log_dir: str = "runs/profiler",
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
    ):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard writer to emit a minimal events file
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # profiler trace handler will deposit JSON into plugins/profile/
        self.handler = tensorboard_trace_handler(self.log_dir)

        # schedule of profiling windows
        self.schedule = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.profiler = None

    def on_train_begin(self, trainer):
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self.profiler = profile(
            activities=activities,
            schedule=self.schedule,
            on_trace_ready=self.handler,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        self.profiler.__enter__()

    def on_batch_end(self, trainer):
        if self.profiler is not None:
            # advance profiling
            self.profiler.step()

        # ——— Important: emit a dummy scalar so TensorBoard sees an events file ———
        epoch = trainer.epoch
        # you can log a real metric too, e.g. trainer.train_metrics["Train_loss"]
        self.writer.add_scalar("Profiler/dummy", 0, epoch)
        self.writer.flush()

    def on_train_end(self, trainer):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
        self.writer.close()
        print(f"[ProfilerCallback] Wrote data under: {self.log_dir}")


def build_callback(config):
    return ProfilerCallback(
        log_dir=config.get("log_dir", f"runs/profiler"),
        wait=config.get("wait", 1),
        warmup=config.get("warmup", 1),
        active=config.get("active", 3),
        repeat=config.get("repeat", 1),
        record_shapes=config.get("record_shapes", False),
        profile_memory=config.get("profile_memory", False),
        with_stack=config.get("with_stack", False),
    )
