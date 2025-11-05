import time
import threading
import numpy as np
from search_formulation import MonotoQual, RevMonotoQual, SearchSpace, AdditiveBundle, hp_cause_bfs, hp_cause_mm, hp_cause_mm_bundled

class TimeoutError(Exception):
    pass

def run_single_experiment(awt_thr, v, method, search_space, timeout=30, **kwargs):
    """
    Run a single experiment with the given parameters using threading timeout.
    
    :param awt_thr: AWT threshold to achieve
    :param v: Initial configuration (numpy array)
    :param method: Method to use ('bfs', 'mm', 'mm_bundled')
    :param search_space: SearchSpace object
    :param timeout: Timeout in seconds
    :param kwargs: Additional arguments (e.g., bundle_size for mm_bundled)
    :return: Dictionary with results
    """
    V = list(range(len(v)))
    
    # Reset simulator time tracking
    if hasattr(search_space.simulate_lifts_func, '__self__'):
        search_space.simulate_lifts_func.__self__.reset_time()
    
    result = {'success': False, 'timeout': True, 'method': method, **kwargs}
    
    def target():
        nonlocal result
        try:
            # tic = time.time()
            sim_time_start = search_space.simulate_lifts_func.__self__.get_total_time() if hasattr(search_space.simulate_lifts_func, '__self__') else 0
            
            if method == 'bfs':
                X = hp_cause_bfs(v, awt_thr, search_space)
            elif method == 'mm':
                mms = [MonotoQual(i) for i in V]
                X = hp_cause_mm(v, awt_thr, mms, search_space)
            elif method == 'mm_bundled':
                bundle_size = kwargs.get('bundle_size', 2)
                mms = [MonotoQual(i) for i in V]
                
                # Create bundles
                bundles = []
                for i in range(0, len(V), bundle_size):
                    bundle_vars = list(range(i, min(i + bundle_size, len(V))))
                    bundles.append(AdditiveBundle(bundle_vars))
                
                X = hp_cause_mm_bundled(V, v, awt_thr, mms, bundles, search_space)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # toc = time.time()
            sim_time_end = search_space.simulate_lifts_func.__self__.get_total_time() if hasattr(search_space.simulate_lifts_func, '__self__') else 0
            
            # total_time = (toc - tic) + (sim_time_end - sim_time_start)
            total_time = sim_time_end - sim_time_start
            
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
