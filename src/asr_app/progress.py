import sys
import threading
from abc import ABC
from typing import Union

import tqdm

_hooked = False
_thread_local = threading.local()


class _CustomProgressBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n
        self.update(n=0)

    def update(self, n=1):
        super().update(n)
        self._current += n

        listeners = _get_thread_local_listeners()

        for listener in listeners:
            listener.on_progress(self._current, self.total)


def _get_thread_local_listeners():
    if not hasattr(_thread_local, 'listeners'):
        _thread_local.listeners = []
    return _thread_local.listeners


def init_progress_hook():
    global _hooked

    if _hooked:
        return

    # Inject into tqdm.tqdm of Whisper, so we can see progress
    transcribe_module = sys.modules['whisper.transcribe']
    transcribe_module.tqdm.tqdm = _CustomProgressBar
    _hooked = True


class ProgressListener(ABC):
    def on_progress(self, current: Union[int, float], total: Union[int, float]):
        self.total = total

    def on_finished(self):
        pass


class ProgressListenerHandle:
    def __init__(self, listener: ProgressListener):
        self.listener = listener

    def __enter__(self):
        register_thread_local_progress_listener(self.listener)

    def __exit__(self, exc_type, exc_val, exc_tb):
        unregister_thread_local_progress_listener(self.listener)

        if exc_type is None:
            self.listener.on_finished()


def register_thread_local_progress_listener(progress_listener: ProgressListener):
    # This is a workaround for the fact that the progress bar is not exposed in the API
    init_progress_hook()

    listeners = _get_thread_local_listeners()
    listeners.append(progress_listener)


def unregister_thread_local_progress_listener(progress_listener: ProgressListener):
    listeners = _get_thread_local_listeners()

    if progress_listener in listeners:
        listeners.remove(progress_listener)


def progress_listener_handle(progress_listener: ProgressListener):
    return ProgressListenerHandle(progress_listener)
