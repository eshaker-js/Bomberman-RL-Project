"""Process priority and keeping Windows awake, with the standard library only.

Long training runs on a Windows laptop should get as much of the CPU as
possible, and must not stop because the laptop goes to sleep.

- ``raise_priority()`` sets the current process to the Windows priority
  class "High" (never "Realtime", which can starve the system). Windows
  child processes do not inherit "High" (they start at "Normal"), so every
  process that should run at high priority calls it itself (the subproc env
  workers do, see ``rl.train_ppo.EnvFactory``).
- ``keep_awake()`` is a context manager that asks Windows not to sleep while
  the block runs and releases the request when it ends, also on a crash
  (used by ``rl.run_ppo_seeds``). The display may still turn off.

On other systems both do nothing and say so.

Windows API: ``SetPriorityClass`` and ``SetThreadExecutionState``
(kernel32), called through ``ctypes``.
"""

import contextlib
import sys

HIGH_PRIORITY_CLASS = 0x00000080
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
NOT_WINDOWS = "only supported on Windows; nothing changed"


def _kernel32():
    import ctypes  # Windows only
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.SetPriorityClass.restype = ctypes.c_int
    kernel32.SetThreadExecutionState.argtypes = (ctypes.c_uint32,)
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
    return kernel32


def raise_priority() -> str:
    """Set this process to "High" priority on Windows; return what happened."""
    if sys.platform != "win32":
        return f"high priority is {NOT_WINDOWS}"
    kernel32 = _kernel32()
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS):
        import ctypes
        return f"could not raise the priority (Windows error {ctypes.get_last_error()})"
    return "process priority raised to High"



@contextlib.contextmanager
def keep_awake():
    """Keep Windows from sleeping while the block runs. Yields what happened."""
    if sys.platform != "win32":
        yield f"keeping the computer awake is {NOT_WINDOWS}"
        return
    kernel32 = _kernel32()
    ok = kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield ("Windows is kept awake until the plan finishes" if ok
               else "could not keep Windows awake")
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
