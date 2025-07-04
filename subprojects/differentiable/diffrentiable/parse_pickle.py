from collections import defaultdict
import pickle
import numpy as np

    
def parse_pickle(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)
        # print(data[8][0])

    d1 = defaultdict(list)
    d2 = defaultdict(list)
    for n, v in data.items():
        for i, x in enumerate(v):
            d1[n].append({"time": x["delta_hp"], "cause": x["a_cause_hp"]})
            d2[n].append({"time": x["delta_hp_nn"], "cause": x["a_cause_hp_nn"]})

    for n in d1.keys():
        diffs = np.array([len(y['cause']['cf_X']) for y in d2[n]])
        print(diffs)

    for ni, n in enumerate(d1.keys()):
        diffs = np.array([x['cause']['o_cf']for x in d2[n]])
        print(diffs)

def parse_pickle_2(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)
        # print(data[8][0])

    d1 = defaultdict(list)
    d2 = defaultdict(list)
    for n, v in data.items():
        for i, x in enumerate(v):
            d1[n].append({"time": x["delta_hp"], "cause": x["a_cause_hp"], "o_thr": x["o_thr"]})
            d2[n].append({"time": x["delta_hp_nn"], "cause": x["a_cause_hp_nn"], "o_thr": x["o_thr"]})

    for n in d1.keys():
        diffs = np.array([len(y['cause']['cf_X']) for y in d2[n]])
        print(diffs)

    for ni, n in enumerate(d1.keys()):
        o_cfs = np.array([x['cause']['o_cf']for x in d2[n]])
        o_thrs = np.array([x['o_thr']for x in d2[n]])
        os_ = np.array([x['o'] for x in data[n]])
        print(list(zip(o_thrs, o_cfs, os_)))

if __name__ == "__main__":
    import sys
    fn = sys.argv[1]
    parse_pickle_2(fn)