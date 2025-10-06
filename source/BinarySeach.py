from loguru import logger
import numpy as np

def binary_search_for_target_value(lower_value, upper_value,
                                   func,
                                   target_result: float = 0,
                                   max_iters=40,
                                   precision=1E-9, floats_allowed=True):
    # debugging now inhibition levels
    lower_result = func(lower_value)
    upper_result = func(upper_value)
    logger.debug(f"Lower value: {lower_value} -> {lower_result}")
    logger.debug(f"Upper value: {upper_value} -> {upper_result}")

    up_reached = False
    lower_reached = False

    if lower_result > target_result: raise ValueError(
        f"Lower value: {lower_value} should produce a result < {target_result}. However, result was {lower_result}")
    if upper_result < target_result: raise ValueError(
        f"Upper value: {upper_value} should produce a result > {target_result}. However, result was {upper_result}")
    '''
    if lower_result >= target_result:
        return 0, 0
    if upper_result <= target_result:
        return 0, 0
    '''
    for i in range(max_iters):
        middle_value = 0.5 * (lower_value + upper_value)
        if not floats_allowed:
            middle_value = int(np.round(middle_value))
        current_result = func(middle_value)

        #logger.debug(f"Current value: {middle_value} -> {current_result}")
        if current_result > target_result:
            delta = upper_value - middle_value
            upper_value = middle_value
            up_reached = abs(delta) < precision
        else:
            delta = lower_value - middle_value
            lower_value = middle_value
            lower_reached = abs(delta) < precision
        if floats_allowed:
            if lower_reached and up_reached:
                logger.debug(f"Break at {i} due to total change {delta}")
                break
        else:
            next_middle_value = int(np.round((lower_value + upper_value) / 2))
            if middle_value == next_middle_value:
                logger.debug(f"Break at {i} because next value ({next_middle_value}) will  eq {middle_value}. In integers, no advance possible. "
                             f"Current values: ({lower_value, upper_value})")
                break

    return lower_value, upper_value