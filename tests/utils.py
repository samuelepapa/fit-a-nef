from flax.core.frozen_dict import FrozenDict


def dict_equal(d1, d2, equal_fn):
    if isinstance(d1, (dict, FrozenDict)) and isinstance(d2, (dict, FrozenDict)):
        if set(d1.keys()) != set(d2.keys()):
            return False
        for key in d1.keys():
            if not dict_equal(d1[key], d2[key], equal_fn):
                return False
        return True
    else:
        return equal_fn(d1, d2)
