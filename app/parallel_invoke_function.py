from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Tuple

def parallel_invoke_function(function: Callable[..., Any], variable_args: List[Any], **constant_args: Any) -> List[Any]:
    """
    Generic function to invoke a custom function in parallel.

    Parameters
    ----------
    variable_args : List[Any]
        List of arguments to process, varying for each invocation.
    function : Callable[..., Any]
        A user-defined function to process a single variable argument. The function must accept
        the variable argument as its first argument and any additional keyword arguments provided
        to `parallel_invoke_function`.
    **constant_args : dict
        Additional arguments that remain constant for each invocation of the custom function.

    Returns
    -------
    List[Any]
        Processed results for each variable argument, in the same order as input.
    """
    MAX_THREADS = 5
    worker = partial(function, **constant_args)

    total_items = len(variable_args)
    results = [None] * total_items

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_index = {
            executor.submit(worker, data_item): index
            for index, data_item in enumerate(variable_args)
        }

        completed_count = 0
        for future in as_completed(future_to_index):
            original_index = future_to_index[future]
            completed_count += 1
            
            try:
                result = future.result(timeout=300)
                results[original_index] = result
                print(f"Processed {completed_count}/{total_items} items")
            except Exception as e:
                print(f"Error processing item at index {original_index}: {str(e)}")

    return results