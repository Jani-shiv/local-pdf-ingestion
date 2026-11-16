"""Progress bar utilities for long-running operations.

This module provides:
- ProgressBar: Wrapper around tqdm with consistent formatting
- progress_bar(): Context manager for progress bars
- track(): Generator wrapper for iterable progress
- track_range(): Generator wrapper for range progress
- MultiProgress: Manage multiple concurrent progress bars
- Spinner: Indeterminate progress spinner
- format_duration(): Human-readable duration formatting
- format_bytes(): Human-readable byte formatting

Example:
    with progress_bar(total=100, desc="Processing") as pbar:
        for item in items:
            process(item)
            pbar.update(1)

    for item in track(items, desc="Processing"):
        process(item)
"""
from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from tqdm import tqdm
from tqdm.auto import trange

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

T = TypeVar("T")


class ProgressBar:
    """Wrapper around tqdm for consistent progress bars.

    Provides a clean interface for progress tracking with
    consistent formatting and easy customization.

    Example:
        bar = ProgressBar(total=100, desc="Processing")
        for item in items:
            process(item)
            bar.update()
        bar.close()
    """

    def __init__(
        self,
        total: int | None = None,
        desc: str = "Processing",
        unit: str = "items",
        disable: bool = False,
        leave: bool = True,
        position: int | None = None,
    ) -> None:
        """Initialize progress bar.

        Args:
            total: Total number of items (None for unknown)
            desc: Description prefix
            unit: Unit name (items, files, chunks, etc.)
            disable: Disable progress bar output
            leave: Keep bar visible after completion
            position: Position for multi-bar display
        """
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
        """Update progress by n steps.

        Args:
            n: Number of steps to advance
        """
        self._bar.update(n)

    def set_description(self, desc: str) -> None:
        """Set progress bar description.

        Args:
            desc: New description text
        """
        self._bar.set_description(desc)

    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix values displayed after the bar.

        Args:
            **kwargs: Key-value pairs to display
        """
        self._bar.set_postfix(**kwargs)

    def close(self) -> None:
        """Close the progress bar and clean up."""
        self._bar.close()

    def __enter__(self) -> ProgressBar:
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
    """Context manager for progress bar.

    Args:
        total: Total number of items
        desc: Description prefix
        unit: Unit name
        disable: Disable progress bar
        leave: Keep bar after completion

    Yields:
        ProgressBar instance
    """
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
    """Track progress of an iterable.

    Automatically determines total from len() if available.

    Args:
        iterable: Iterable to track
        desc: Description prefix
        total: Total items (auto-detected if None)
        unit: Unit name
        disable: Disable progress bar

    Yields:
        Items from the iterable
    """
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
        yield from pbar


def track_range(
    n: int,
    desc: str = "Processing",
    disable: bool = False,
) -> Generator[int, None, None]:
    """Track progress of a range.

    Args:
        n: Range size (0 to n-1)
        desc: Description prefix
        disable: Disable progress bar

    Yields:
        Integers from 0 to n-1
    """
    with trange(
        n,
        desc=desc,
        disable=disable,
        file=sys.stdout,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        yield from pbar


class MultiProgress:
    """Manage multiple progress bars simultaneously.

    Useful for parallel processing with separate progress bars.

    Example:
        mp = MultiProgress()
        bar1 = mp.add_bar("download", total=100)
        bar2 = mp.add_bar("process", total=50)
        # Update independently
        mp.close_all()
    """

    def __init__(self) -> None:
        self._bars: dict[str, ProgressBar] = {}

    def add_bar(
        self,
        key: str,
        total: int | None = None,
        desc: str = "Processing",
        unit: str = "items",
        position: int | None = None,
    ) -> ProgressBar:
        """Add a new progress bar.

        Args:
            key: Unique identifier for the bar
            total: Total items
            desc: Description
            unit: Unit name
            position: Display position (auto-assigned if None)

        Returns:
            Created ProgressBar instance
        """
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
        """Get a progress bar by key.

        Args:
            key: Bar identifier

        Returns:
            ProgressBar or None if not found
        """
        return self._bars.get(key)

    def remove_bar(self, key: str) -> None:
        """Remove and close a progress bar.

        Args:
            key: Bar identifier
        """
        if key in self._bars:
            self._bars[key].close()
            del self._bars[key]

    def close_all(self) -> None:
        """Close all progress bars."""
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()

    def __enter__(self) -> MultiProgress:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close_all()


class Spinner:
    """Simple spinner for indeterminate progress.

    Displays animated spinner while operation is in progress.
    Useful when total is unknown.

    Example:
        with Spinner("Loading..."):
            do_work()
    """

    SPINNERS = {
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "line": ["|", "/", "-", "\\"],
        "arrow": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        "bounce": ["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"],
    }

    def __init__(self, desc: str = "Processing", style: str = "dots", disable: bool = False) -> None:
        """Initialize spinner.

        Args:
            desc: Description text
            style: Animation style (dots, line, arrow, bounce)
            disable: Disable spinner output
        """
        self.desc = desc
        self.style = style
        self.disable = disable
        self._index = 0
        self._running = False
        self._thread: threading.Timer | None = None

    def start(self) -> None:
        """Start the spinner animation."""
        if self.disable:
            return
        self._running = True
        self._print_frame()

    def stop(self, final_msg: str | None = None) -> None:
        """Stop the spinner.

        Args:
            final_msg: Optional message to display after stopping
        """
        if self.disable:
            return
        self._running = False
        if self._thread:
            self._thread.cancel()
        if final_msg:
            pass
        else:
            pass

    def _print_frame(self) -> None:
        """Print current animation frame."""
        if not self._running:
            return
        frames = self.SPINNERS.get(self.style, self.SPINNERS["dots"])
        frames[self._index % len(frames)]
        self._index += 1
        if self._running:
            self._thread = threading.Timer(0.1, self._print_frame)
            self._thread.daemon = True
            self._thread.start()

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1.5s", "2m 30s", "1h 15m")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def format_bytes(bytes_val: int) -> str:
    """Format bytes in human-readable format.

    Args:
        bytes_val: Size in bytes

    Returns:
        Formatted string (e.g., "1.5KB", "2.3MB", "1.2GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}PB"