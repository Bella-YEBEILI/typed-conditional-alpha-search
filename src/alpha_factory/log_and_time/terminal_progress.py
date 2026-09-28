import time


class TerminalProgress:
    def __init__(self,label:str,total:int,kind:str="batch",show_counts:bool=True,width:int=24):
        self.label = label
        self.total = total
        self.kind = kind
        self.show_counts = show_counts
        self.width = width
        self.done = 0
        self.candidate = 0
        self.seed = 0
        self.start = None
        self.last_len = 0

    def open(self):
        self.start = time.perf_counter()
        return self

    def update(self,factor=None):
        self.done += 1
        if factor is not None and self.show_counts:
            if factor.result.is_candidate==True:
                self.candidate += 1
            if factor.result.is_seed==True:
                self.seed += 1
        self._print()

    def finish(self):
        if self.done==0:
            self._print(end=True)
            return
        self._print(end=True)

    def _print(self,end:bool=False):
        count = self.total
        pct = self.done/count if count>0 else 1.0
        filled = int(self.width*pct)
        bar = "#"*filled+"."*(self.width-filled)
        line = (
            f"\r[{self.kind}] {self.label} "
            f"{self.done}/{count} {pct*100:5.1f}% [{bar}]"
        )
        if self.show_counts:
            line += f" C={self.candidate} S={self.seed} A={self.done}"
        if end and self.start is not None:
            line += f" sec={time.perf_counter()-self.start:.2f}"
        pad = " "*max(0,self.last_len-len(line))
        print(line+pad,end="\n" if end else "",flush=True)
        self.last_len = len(line)
