import json
import numpy as np
from boia_utils.utils import HashableDict


def serialize_np(obj):
    """
    Recursively convert NumPy arrays and values to lists and primitive types within a nested object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: serialize_np(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_np(item) for item in obj]
    elif isinstance(obj, HashableDict):
        return {key: serialize_np(value) for key, value in obj.items()}
    else:
        return obj


def save_status_log(status_log, filename):
    """Save the status log to a JSON file."""

    with open(filename, "w") as f:
        json.dump(serialize_np(status_log), f, indent=2)


def load_status_log(filename):
    """Load the status log from a JSON file."""
    with open(filename, "r") as f:
        status_log = json.load(f)
    return status_log


def check_all_done(status_dict):
    def _check_single(name_done, name_conf):
        to_rtn = True
        for i in range(len(status_dict[name_done])):
            if status_dict[name_done][i] < status_dict[name_conf]["num"][i]:
                to_rtn = False
                break
        return to_rtn

    train_status = _check_single("train_done", "train_confs")
    val_status = _check_single("val_done", "val_confs")
    test_status = _check_single("test_done", "test_confs")
    ood_status = _check_single("ood_done", "ood_confs")

    if train_status:
        print("Train done")

    if val_status:
        print("Val done")

    if test_status:
        print("Test done")

    if ood_status:
        print("Ood done")

    return train_status and val_status and test_status and ood_status
