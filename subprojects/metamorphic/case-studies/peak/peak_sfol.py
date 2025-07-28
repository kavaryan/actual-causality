from matplotlib import pyplot as plt
import numpy as np
import operator
from typing import Tuple, Literal, TypeAlias
# from scipy.signal import find_peaks

# Define a type alias for relational operator strings
RelOp: TypeAlias = Literal['=', '!=', '<', '<=', '>', '>=']

# Map string operators to Python operator functions
REL_OPS = {
    '=': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
}

def check_peak(
    time: np.ndarray,
    signal: np.ndarray,
    a1_cond: Tuple[RelOp, float],
    a2_cond: Tuple[RelOp, float],
    sp1_cond: Tuple[RelOp, float],
    sp2_cond: Tuple[RelOp, float],
    w_cond: Tuple[RelOp, float]
) -> dict | None:
    """
    Check if the signal contains a peak pattern satisfying all constraints.

    Returns a dictionary of peak properties if found, otherwise None.
    """
    for i in range(1, len(signal) - 2):
        for j in range(i + 1, len(signal) - 1):
            for k in range(j + 1, len(signal)):
                VP1, PP, VP2 = i, j, k

                # Local min at VP1
                # FIXME: check local min at the whole [VP1,PP]
                if not (signal[VP1] < signal[VP1 - 1] and signal[VP1] < signal[VP1 + 1]):
                    continue

                # Local max at PP
                # FIXME: check local max at the whole [VP1,VP2]
                if not (signal[PP] > signal[PP - 1] and signal[PP] > signal[PP + 1]):
                    continue

                # Local min at VP2
                # FIXME: check local min at the whole [PP,VP2]
                if not (signal[VP2] < signal[VP2 - 1] and signal[VP2] < signal[VP2 - 1]):
                    continue

                # Compute features
                a1 = signal[PP] - signal[VP1]
                a2 = signal[PP] - signal[VP2]
                sp1 = a1 / (time[PP] - time[VP1]) if time[PP] != time[VP1] else np.inf
                sp2 = a2 / (time[PP] - time[VP2]) if time[PP] != time[VP2] else np.inf
                w = time[VP2] - time[VP1]

                # Apply relational conditions
                if a1_cond and not REL_OPS[a1_cond[0]](a1, a1_cond[1]):
                    continue
                if a2_cond and not REL_OPS[a2_cond[0]](a2, a2_cond[1]):
                    continue
                if sp1_cond and not REL_OPS[sp1_cond[0]](sp1, sp1_cond[1]):
                    continue
                if sp2_cond and not REL_OPS[sp2_cond[0]](sp2, sp2_cond[1]):
                    continue
                if w_cond and not REL_OPS[w_cond[0]](w, w_cond[1]):
                    continue

                return {
                    "a1": a1,
                    "a2": a2,
                    "sp1": sp1,
                    "sp2": sp2,
                    "w": w,
                    "VP1_index": VP1,
                    "PP_index": PP,
                    "VP2_index": VP2,
                    "VP1_time": time[VP1],
                    "PP_time": time[PP],
                    "VP2_time": time[VP2],
                    "VP1_value": signal[VP1],
                    "PP_value": signal[PP],
                    "VP2_value": signal[VP2],
                }

    return None


def plot_signal_with_peak(t, s, result, title):
    plt.figure(figsize=(8, 4))
    plt.plot(t, s, label="Signal")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.grid(True)

    if result:
        vp1 = result["VP1_index"]
        pp = result["PP_index"]
        vp2 = result["VP2_index"]
        plt.plot(t[[vp1, pp, vp2]], s[[vp1, pp, vp2]], 'ro')

        for name in ["VP1", "PP", "VP2"]:
            idx = result[f"{name}_index"]
            val = result[f"{name}_value"]
            time_val = result[f"{name}_time"]
            plt.annotate(f"{name}\n({time_val:.2f}, {val:.2f})",
                            (time_val, val), textcoords="offset points", xytext=(-10,10), ha='center')

        plt.legend()
    else:
        plt.text(0.5, 0.5, "No peak detected", transform=plt.gca().transAxes,
                    fontsize=12, ha='center', color='red')

    plt.tight_layout()
    plt.show()