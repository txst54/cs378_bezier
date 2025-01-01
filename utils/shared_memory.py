import contextlib
import os
import sys
import warnings
from multiprocessing import shared_memory

import numpy as np
import psutil


def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def create_shm(params):
    """Create shared memory"""
    shm = None
    try:
        shm = shared_memory.SharedMemory(create=True, size=params.nbytes)
        np.ndarray(params.shape, dtype=params.dtype, buffer=shm.buf)[:] = params
        return shm.name
    finally:
        if 'shm' in locals() and shm is not None:
            shm.close()


@contextlib.contextmanager
def get_shm(shm_name, shape, dtype):
    """Safely retrieve parameters from shared memory."""
    shm = None
    try:
        if not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape)}")
        shm = shared_memory.SharedMemory(name=shm_name)
        buf = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        if buf is None or buf.size == 0:
            raise ValueError(f"Shared memory buffer {shm_name} of shape {shape} empty or not properly initialized.")
        yield buf
    except Exception as e:
        print(f"Error retrieving params for {shm_name} of shape {shape}: {e}")
        return None
    finally:
        if 'shm' in locals() and shm is not None:
            shm.close()


def set_shm(shm_name, val, shape, dtype):
    with get_shm(shm_name, shape, dtype) as buf:
        buf[:] = val


def clean_shm(shm_name):
    shm = shared_memory.SharedMemory(name=shm_name)
    shm.close()
    shm.unlink()


def init_worker():
    warnings.filterwarnings("ignore", message=".*Falling back to cpu.*")
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')