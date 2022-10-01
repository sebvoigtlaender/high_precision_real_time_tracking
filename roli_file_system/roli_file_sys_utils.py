from typing import Any, Mapping
import ctypes
import numpy as np


_shared_lib = ctypes.CDLL('/home/sebastian/libroli_fse.so')

_shared_lib.roli_fse_open.argtypes = [ctypes.c_char_p,]
_shared_lib.roli_fse_open.restype = ctypes.c_void_p
_shared_lib.roli_fse_get_width.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_shared_lib.roli_fse_get_height.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_shared_lib.roli_fse_get_height.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_shared_lib.roli_fse_get_n_frames.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
_shared_lib.roli_fse_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_shared_lib.roli_fse_close.argtypes = [ctypes.c_void_p,]


def file_open(filename):
    return _shared_lib.roli_fse_open(filename)

def file_get_width(handle):
    width = ctypes.POINTER(ctypes.c_uint32)(ctypes.c_uint32())
    _shared_lib.roli_fse_get_width(handle, width)
    return width.contents.value

def file_get_height(handle):
    height = ctypes.POINTER(ctypes.c_uint32)(ctypes.c_uint32())
    _shared_lib.roli_fse_get_height(handle, height)
    return height.contents.value

def file_get_n_frames(handle):
    n_frames = ctypes.POINTER(ctypes.c_uint64)(ctypes.c_uint64())
    _shared_lib.roli_fse_get_n_frames(handle, n_frames)
    n_frames = n_frames.contents.value-1
    return n_frames

def file_read(handle, idx):
    x = np.ndarray((file_get_width(handle), file_get_height(handle)), dtype=np.uint16)
    p, read_only_flag = x.__array_interface__['data']
    _shared_lib.roli_fse_read(handle, p, idx)
    return x

def file_close(handle):
    return _shared_lib.roli_fse_close(handle)

def get_path_to_file(path_dict: Mapping[str, Any], idx: int, file_type: str = 'ir.roli') -> str:
    path_to_file = '/nfs/data{}/{}/data_raw/{}/{}'.format(path_dict[idx]['server_id'], path_dict[idx]['user'], path_dict[idx]['date'], file_type)
    return path_to_file

def get_path_to_result(path_dict: Mapping[str, Any], idx: int) -> str:
    path_to_result = 'results/test_result_{}_{}'.format(path_dict[idx]['user'], path_dict[idx]['date'])
    return path_to_result