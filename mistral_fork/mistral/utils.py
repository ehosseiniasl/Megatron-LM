def ensure_divisibility(numerator: int, denominator: int) -> None:
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )
