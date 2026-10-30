def valid_seed(seed: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(s % 2**32 for s in seed)
