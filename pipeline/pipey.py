from enum import Enum
from functools import reduce

modifier = Enum('modifier', 'map filter reduce window')  # TODO maybe add async versions


__pipeline_functions = {
    modifier.map: map,
    modifier.filter: filter,
    modifier.reduce: reduce,
    modifier.window: lambda f, x: f(x)
}


def apply_pipeline(input_iterable, pipeline):
    if len(pipeline) == 0:
        return input_iterable

    function_to_apply, function_type = pipeline.pop(0)
    assert len(pipeline) == 0 or function_type != modifier.reduce

    applied_function = __pipeline_functions[function_type](function_to_apply, input_iterable)
    return apply_pipeline(applied_function, pipeline)
