from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Tuple
import time
import logging

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
    
    batch_start_time = time.time()
    logging.warning(f"Starting parallel processing of {total_items} items with {MAX_THREADS} threads")

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # Submit all tasks and track their original indices
        future_to_index = {
            executor.submit(worker, data_item): index
            for index, data_item in enumerate(variable_args)
        }
        future_to_start_time = {future: time.time() for future in future_to_index}

        completed_count = 0
        slow_items = []
        
        for future in as_completed(future_to_index, timeout=600):  # 10 minute overall timeout
            original_index = future_to_index[future]
            start_time = future_to_start_time[future]
            completed_count += 1
            
            try:
                result = future.result(timeout=120)  # 2 minute per-task timeout
                processing_time = time.time() - start_time
                results[original_index] = result
                
                if processing_time > 30:  # Flag slow items (>30 seconds)
                    slow_items.append((original_index, processing_time))
                    logging.warning(f"SLOW ITEM: Index {original_index} took {processing_time:.1f}s")
                
                logging.warning(f"Completed {completed_count}/{total_items} items (index {original_index}, {processing_time:.1f}s)")
                
            except Exception as e:
                processing_time = time.time() - start_time
                logging.error(f"ERROR: Item at index {original_index} failed after {processing_time:.1f}s: {str(e)}")
                results[original_index] = None

    total_batch_time = time.time() - batch_start_time
    avg_time = total_batch_time / total_items if total_items > 0 else 0
    
    logging.warning(f"Batch complete: {total_batch_time:.1f}s total, {avg_time:.1f}s average per item")
    if slow_items:
        logging.warning(f"Found {len(slow_items)} slow items (>30s): {slow_items}")

    return results