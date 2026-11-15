"""Progress bar utilities for long-running operations."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Generator, Iterable, TypeVar

from tqdm import tqdm
from tqdm.auto import trange

T = TypeVar("T")


class ProgressBar:
    """Wrapper around tqdm for consistent progress bars."""

    def __init__(
        self,
        total: int | None = None,
        desc: str = "Processing",
        unit: str = "items",
        disable: bool = False,
        leave: bool = True,
        position: int | None = None,
    ):
        self._bar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            disable=disable,
            leave=leave,
            position=position,
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self._bar.update(n)

    def set_description(self, desc: str) -> None:
        """Set progress bar description."""
        self._bar.set_description(desc)

    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix values."""
        self._bar.set_postfix(**kwargs)

    def close(self) -> None:
        """Close the progress bar."""
        self._bar.close()

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


@contextmanager
def progress_bar(
    total: int | None = None,
    desc: str = "Processing",
    unit: str = "items",
    disable: bool = False,
    leave: bool = True,
) -> Generator[ProgressBar, None, None]:
    """Context manager for progress bar."""
    bar = ProgressBar(total=total, desc=desc, unit=unit, disable=disable, leave=leave)
    try:
        yield bar
    finally:
        bar.close()


def track(
    iterable: Iterable[T],
    desc: str = "Processing",
    total: int | None = None,
    unit: str = "items",
    disable: bool = False,
) -> Generator[T, None, None]:
    """Track progress of an iterable."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            pass

    with tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=disable,
        file=sys.stdout,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for item in pbar:
            yield item


def track_range(
    n: int,
    desc: str = "Processing",
    disable: bool = False,
) -> Generator[int, None, None]:
    """Track progress of a range."""
    with trange(
        n,
        desc=desc,
        disable=disable,
        file=sys.stdout,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for i in pbar:
            yield i


class MultiProgress:
    """Manage multiple progress bars."""

    def __init__(self):
        self._bars: dict[str, ProgressBar] = {}

    def add_bar(
        self,
        key: str,
        total: int | None = None,
        desc: str = "Processing",
        unit: str = "items",
        position: int | None = None,
    ) -> ProgressBar:
        """Add a new progress bar."""
        if position is None:
            position = len(self._bars)

        bar = ProgressBar(
            total=total,
            desc=desc,
            unit=unit,
            position=position,
            leave=False,
        )
        self._bars[key] = bar
        return bar

    def get_bar(self, key: str) -> ProgressBar | None:
        """Get a progress bar by key."""
        return self._bars.get(key)

    def remove_bar(self, key: str) -> None:
        """Remove and close a progress bar."""
        if key in self._bars:
            self._bars[key].close()
            del self._bars[key]

    def close_all(self) -> None:
        """Close all progress bars."""
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()

    def __enter__(self) -> "MultiProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close_all()


class Spinner:
    """Simple spinner for indeterminate progress."""

    SPINNERS = {
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "line": ["|", "/", "-", "\\"],
        "arrow": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        "bounce": ["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"],
    }

    def __init__(self, desc: str = "Processing", style: str = "dots", disable: bool = False):
        self.desc = desc
        self.style = style
        self.disable = disable
        self._index = 0
        self._running = False
        self._thread = None

    def start(self) -> None:
        """Start the spinner."""
        if self.disable:
            return
        self._running = True
        self._print_frame()

    def stop(self, final_msg: str | None = None) -> None:
        """Stop the spinner."""
        if self.disable:
            return
        self._running = False
        if final_msg:
            print(f"\r{final_msg}", file=sys.stdout, flush=True)
        else:
            print("\r" + " " * (len(self.desc) + 4) + "\r", file=sys.stdout, flush=True, end="")

    def _print_frame(self) -> None:
        """Print current frame."""
        if not self._running:
            return
        frames = self.SPINNERS.get(self.style, self.SPINNERS["dots"])
        frame = frames[self._index % len(frames)]
        print(f"\r{self.desc} {frame}", file=sys.stdout, flush=True, end="")
        self._index += 1
        if self._running:
            import threading
            self._thread = threading.Timer(0.1, self._print_frame)
            self._thread.daemon = True
            self._thread.start()

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_bytes(bytes_val: int) -> str:
    """Format bytes in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}PB"