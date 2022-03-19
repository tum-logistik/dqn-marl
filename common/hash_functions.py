import numpy as np

def hash_to_action_index(hash_key, step = 10):
    n = 0
    hash_key_copy = hash_key
    act_inds = []
    while np.mod(hash_key_copy, step) > 0:
        hash_key_copy = np.floor(hash_key_copy/step)
        info_sec = np.mod(hash_key , np.power(step, n+1))
        act_inds.append(int(np.floor(info_sec/np.power(step, n))))
        n += 1
    act_inds.reverse()
    return act_inds

def action_index_to_hash(action_indices, step = 10):
    n = len(action_indices)
    hash_key = 0
    for i in range(n):
        if action_indices[i] > step:
            return False
        hash_key += action_indices[i] * np.power(step, n - i - 1)
    return hash_key

