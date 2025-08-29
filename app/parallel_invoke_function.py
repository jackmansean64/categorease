from functools import partial
from typing import Callable, List, Any, Tuple
import time
import logging
import eventlet
from eventlet.green import threading

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
    logging.warning(f"Starting parallel processing of {total_items} items with {MAX_THREADS} green threads")

    pool = eventlet.GreenPool(MAX_THREADS)
    
    greenthread_start_times = {}
    completed_count = 0
    slow_items = []
    
    logging.warning(f"Submitted all {total_items} tasks to green thread pool")
    
    def process_with_timing(index_and_item):
        index, item = index_and_item
        start_time = time.time()
        try:
            result = worker(item)
            processing_time = time.time() - start_time
            
            nonlocal completed_count
            completed_count += 1
            
            if processing_time > 30:  # Flag slow items (>30 seconds)
                slow_items.append((index, processing_time))
                logging.warning(f"SLOW ITEM: Index {index} took {processing_time:.1f}s")
            
            logging.warning(f"Completed {completed_count}/{total_items} items (index {index}, {processing_time:.1f}s)")
            return index, result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"ERROR: Item at index {index} failed after {processing_time:.1f}s: {str(e)}")
            return index, None
    
    # Process all items and collect results
    indexed_items = [(i, item) for i, item in enumerate(variable_args)]
    
    # Use pool.imap to process items with a timeout
    try:
        with eventlet.Timeout(600):  # Overall timeout of 10 minutes
            for index, result in pool.imap(process_with_timing, indexed_items):
                results[index] = result
    except eventlet.Timeout:
        logging.error("Batch processing timed out after 10 minutes")
    finally:
        pool.waitall()  # Wait for any remaining greenthreads to complete

    total_batch_time = time.time() - batch_start_time
    avg_time = total_batch_time / total_items if total_items > 0 else 0
    
    logging.warning(f"Batch complete: {total_batch_time:.1f}s total, {avg_time:.1f}s average per item")
    if slow_items:
        logging.warning(f"Found {len(slow_items)} slow items (>30s): {slow_items}")

    return results