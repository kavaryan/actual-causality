import time
import threading
import numpy as np
from search_formulation import MonotoQual, RevMonotoQual, SearchSpace, AdditiveBundle, hp_cause_bfs, hp_cause_mm, hp_cause_mm_bundled

class TimeoutError(Exception):
    pass

def av_run_single_experiment(td_thr, v, method, search_space, timeout=30, **kwargs):
    """
    Run a single experiment with the given parameters using threading timeout.
    
    :param td_thr: TD threshold to achieve
    :param v: Initial configuration (numpy array)
    :param method: Method to use ('bfs', 'mm')
    :param search_space: SearchSpace object
    :param timeout: Timeout in seconds
    :param kwargs: Additional arguments
    :return: Dictionary with results
    """
    V = list(range(len(v)))
    
    # Reset simulator time tracking
    search_space.simulator.reset_time()
    
    result = {'success': False, 'timeout': True, 'method': method, **kwargs}
    
    def target():
        nonlocal result
        try:
            if method == 'bfs':
                X = hp_cause_bfs(v, td_thr, search_space)
            elif method == 'mm':
                mms = [RevMonotoQual(i) for i in V]
                X = hp_cause_mm(v, td_thr, mms, search_space)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Get total simulation time for this experiment
            total_time = search_space.simulator.get_total_time()
            
            result.update({
                'success': True,
                'time': total_time,
                'solution': X,
                'timeout': False
            })
            
        except Exception as e:
            result.update({
                'success': False,
                'time': timeout,
                'solution': None,
                'timeout': False,
                'error': str(e)
            })
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Thread is still running, it timed out
        result.update({
            'success': False,
            'time': timeout,
            'solution': None,
            'timeout': True
        })
    
    return result
