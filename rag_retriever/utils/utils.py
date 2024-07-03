import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Generator, List

from loguru import logger


def run_in_thread_pool(
    func: Callable,
    params: List[Dict] = [],
) -> Generator:
    """
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    """
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            tasks.append(pool.submit(func, **kwargs))

        for obj in as_completed(tasks):
            try:
                yield obj.result()
            except Exception as e:
                logger.error(f"error in sub thread: {e}", exc_info=True)


def run_in_process_pool(
    func: Callable,
    params: List[Dict] = [],
) -> Generator:
    """
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    """
    tasks = []
    max_workers = None
    if sys.platform.startswith("win"):
        max_workers = min(
            mp.cpu_count(), 60
        )  # max_workers should not exceed 60 on windows
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for kwargs in params:
            tasks.append(pool.submit(func, **kwargs))

        for obj in as_completed(tasks):
            try:
                yield obj.result()
            except Exception as e:
                logger.error(f"error in sub process: {e}", exc_info=True)
