"""Process priority, with the standard library only.

Long training runs on a Windows laptop should get as much of the CPU as
possible.

- ``raise_priority()`` sets the current process to the Windows priority
  class "High" (never "Realtime", which can starve the system). Windows
  child processes do not inherit "High" (they start at "Normal"), so every
  process that should run at high priority calls it itself (the subproc env
  workers do, see ``rl.train_ppo.EnvFactory``).

On other systems it does nothing and says so.

Windows API: ``SetPriorityClass`` (kernel32), called through ``ctypes``.
"""

import sys

HIGH_PRIORITY_CLASS = 0x00000080
NOT_WINDOWS = "only supported on Windows; nothing changed"


def _kernel32():
    import ctypes  # Windows only
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.SetPriorityClass.restype = ctypes.c_int
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

