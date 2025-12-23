from multiprocessing import Process, Queue
from typing import Callable, Any, Tuple, Dict

def call_with_timeout(
    fn: Callable,
    timeout: float,
    args: Tuple = (),
    kwargs: Dict = None,
):
    if kwargs is None:
        kwargs = {}

    q = Queue()

    def worker():
        try:
            q.put((True, fn(*args, **kwargs)))
        except Exception as e:
            q.put((False, e))

    p = Process(target=worker)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Function call exceeded {timeout} seconds")

    success, payload = q.get()
    if success:
        return payload
    else:
        raise payload

