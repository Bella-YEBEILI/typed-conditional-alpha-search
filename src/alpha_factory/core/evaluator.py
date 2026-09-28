from concurrent.futures import ThreadPoolExecutor,as_completed
from typing import Union
from .factor import Factor
from .parser import FormulaParser
from .value_engine import ValueEngine
from .post_processor import PostProcessor
from .result_engine import ResultEngine
from .selector import Selector
from ..log_and_time.terminal_progress import TerminalProgress


class Evaluator:
    def __init__(
        self,
        parser:FormulaParser,
        value_engine:ValueEngine,
        post_processor:PostProcessor,
        result_engine:ResultEngine,
        selector:Selector,
    ):
        self.parser = parser
        self.value_engine = value_engine
        self.post_processor = post_processor
        self.result_engine = result_engine
        self.selector = selector

    def evaluate(self,factor:Factor)->Factor:
        factor.node = self.parser.parse(factor.spec.formula)
        value = self.value_engine.calc(factor.node.root)
        value = self.post_processor.process(value,factor.spec.decay,factor.spec.neutralize)
        factor.result = self.result_engine.calc(value)
        factor.result = self.selector.select(factor.result)
        return factor

    def cache_values(self,
                     factors:list[Factor],
                     threads:int=9,
                     progress_label:Union[str,None]=None):
        def run_one(factor):
            if factor.node.root is None:
                factor.node = self.parser.parse(factor.spec.formula)
            self.value_engine.cache_value(factor.node.root)

        progress = TerminalProgress(progress_label,len(factors),kind="cache",show_counts=False).open() if progress_label is not None else None
        try:
            if threads<=1:
                for factor in factors:
                    run_one(factor)
                    if progress is not None:
                        progress.update()
                return

            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = [executor.submit(run_one,factor) for factor in factors]
                for future in as_completed(futures):
                    future.result()
                    if progress is not None:
                        progress.update()
        finally:
            if progress is not None:
                progress.finish()

    def multi_evaluate(self,
                       factors:list[Factor],
                       threads:int=9,
                       progress_label:Union[str,None]=None)->list[Factor]:
        progress = TerminalProgress(progress_label,len(factors),kind="batch",show_counts=True).open() if progress_label is not None else None
        try:
            if threads<=1:
                results = []
                for factor in factors:
                    factor = self.evaluate(factor)
                    results.append(factor)
                    if progress is not None:
                        progress.update(factor)
                return results

            results = [None]*len(factors)
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {
                    executor.submit(self.evaluate,factor):i
                    for i,factor in enumerate(factors)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    factor = future.result()
                    results[i] = factor
                    if progress is not None:
                        progress.update(factor)
            return results
        finally:
            if progress is not None:
                progress.finish()
