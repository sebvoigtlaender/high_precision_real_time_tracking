import pickle

def get_path_to_file(path_dict, idx):
    '''
    Return path to data file

    Args:
        path_dict: dict containing the paths to the data files
        idx: key to specific path

    Returns: 
        path_to_file: path to data file
    '''
    path_to_file = path_dict[idx]
    return path_to_file

def get_path_to_result(path_dict, idx):
    '''
    Return path to result file. The arguments are given for backward compatibility with our internal file system, but they are not used in the basic version presented here.

    Args:
        path_dict: dict containing the paths to the data files
        idx: key to specific path

    Returns: 
        path_to_result: path to result file
    '''
    path_to_result = 'results/result'
    return path_to_result

def file_open(path_to_file):
    '''
    Return path to file. This function is used for backward compatibility with our internal file system, but it is not necessary for the basic version presented here.
    '''
    return path_to_file

def file_get_n_frames(path_to_file):
    '''
    Return number of frames in dataset.

    Args:
        path_to_file: path to data file

    Returns: 
        n_frames: number of frames in dataset
    '''
    train_data = pickle.load(open(f'{path_to_file}', 'rb'))['data']
    n_frames = len(train_data)
    return n_frames

def file_read(path_to_file, i):
    '''
    Return frame. This function is clumsy and slow when used with pickled data, but we require it for the sake of backward compatibility with our internal file system.
    It is necessary when the path_to_file points to a file not accessible by a python function.

    Args:
        path_to_file: path to data file
        i: frame index

    Returns: 
        x: single frame at index i
    '''
    train_data = pickle.load(open(f'{path_to_file}', 'rb'))['data']
    x = train_data[i]
    return x

def file_close(handle):
    '''
    Close. This function is not used here, but required for backward compatibility with our internal file system.
    '''
    pass