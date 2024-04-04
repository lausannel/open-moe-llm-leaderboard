import functools
from lm_eval.api.metrics import mean


def process_results_decorator(func):
    # This decorator processes the results of a task before passing them to the original process_results function
    @functools.wraps(func)
    def wrapper(self, doc, results, *args, **kwargs):
        # We process the results here
        processed_results = [r[0] for r in results]

        latency = sum([r[1] for r in results]) / len(results)
        print(f"Average latency: {latency}")

        # Now call the original process_results with the processed results
        result_dict = func(self, doc, processed_results, *args, **kwargs)
        result_dict["latency"] = latency
        return result_dict
    return wrapper


def aggregation_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        aggregation_list = func(self, *args, **kwargs)
        aggregation_list["latency"] = mean
        return aggregation_list
    return wrapper


def higher_is_better_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        higher_is_better_dict = func(self, *args, **kwargs)
        higher_is_better_dict["latency"] = False
        return higher_is_better_dict
    return wrapper


def measure_system_metrics(cls):
    method_decorators = {
        'process_results': [process_results_decorator],
        'aggregation': [aggregation_decorator],
        'higher_is_better': [higher_is_better_decorator],
    }
    for method_name, decorators in method_decorators.items():
        if callable(getattr(cls, method_name, None)):
            original_method = getattr(cls, method_name)
            for decorator in reversed(decorators):
                original_method = decorator(original_method)
            setattr(cls, method_name, original_method)
    return cls
