"""Basic stopping history consistency checks, reused from the paused adapter."""
import math
import re

import numpy as np

from songpop.phase2 import tabm


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def sha256(value, name):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def stopping_history(receipt, *, max_epochs, patience, tolerance, batch_size, fitting_rows):
    """Check the legacy loss/epoch policy and exact optimizer-update accounting."""
    n = positive_integer(receipt.get("epochs_evaluated"), "epochs_evaluated")
    losses = receipt.get("validation_losses")
    if (not isinstance(losses, list) or len(losses) != n or n > max_epochs
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                   for v in losses)):
        raise ValueError("validation_losses must be finite nonnegative losses for exactly the evaluated epochs")
    best = positive_integer(receipt.get("best_epoch"), "best_epoch")
    chosen = positive_integer(receipt.get("chosen_epoch"), "chosen_epoch")
    stopped, expected = tabm.stop_rule(losses, tolerance, patience)
    if best != int(np.argmin(losses)) + 1 or chosen != expected:
        raise ValueError("saved best/chosen epoch disagrees with the declared stopping rule")
    if any(tabm.stop_rule(losses[:end], tolerance, patience)[0] for end in range(1, n)):
        raise ValueError("stopping history continues after its declared patience was exhausted")
    if receipt.get("stopped_by") != ("patience" if stopped else "max_epochs") or (not stopped and n != max_epochs):
        raise ValueError("stopped_by disagrees with the stopping history or epoch cap")
    n_train = positive_integer(receipt.get("inner_train_rows"), "inner_train_rows")
    n_valid = positive_integer(receipt.get("inner_validation_rows"), "inner_validation_rows")
    if n_train + n_valid != fitting_rows:
        raise ValueError("stopping partition row counts do not cover the fitting population")
    for key, expected_updates in (("stopping_updates", math.ceil(n_train / batch_size) * n),
                                  ("final_updates", math.ceil(fitting_rows / batch_size) * chosen)):
        if positive_integer(receipt.get(key), key) != expected_updates:
            raise ValueError(f"{key} disagrees with the declared epoch-to-update policy")
