"""
QuantaAlpha custom factor workspace.

This workspace overrides the upstream experiment workspace with project-level
factor templates and initializes an empty git repo in the workspace to suppress
noisy recorder git output.
"""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from rdagent.log import rdagent_logger as logger

from quantaalpha.factors.upstream_factor_api import UpstreamFactorWorkspace
from quantaalpha.utils.factor_local_env import FactorLocalEnv

class FactorWorkspace(UpstreamFactorWorkspace):
    """Inject local templates and keep local execution behavior stable for factor workspaces."""

    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(template_folder_path, *args, **kwargs)

    def before_execute(self) -> None:
        """Init an empty git repo in the workspace to suppress recorder git warnings."""
        super().before_execute()
        git_dir = self.workspace_path / '.git'
        if not git_dir.exists():
            try:
                subprocess.run(
                    ['git', 'init'],
                    cwd=str(self.workspace_path),
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

    def execute(
        self,
        config_name: str = 'conf.yaml',
        run_env: dict | None = None,
        *args,
        **kwargs,
    ) -> str:
        legacy_config_name = kwargs.pop(('q' + 'lib_config_name'), None)
        if legacy_config_name:
            config_name = legacy_config_name
        if run_env is None:
            run_env = {}
        run_env.setdefault("QUANTAALPHA_QUIET_CONSOLE", "1")
        run_env.setdefault("QUANTAALPHA_QUIET_SUBPROCESS", "1")
        run_env.setdefault("QUANTAALPHA_DISABLE_TQDM", "1")
        use_local = kwargs.pop('use_local', True)
        if not use_local:
            return super().execute(config_name, run_env, *args, **kwargs)

        env = FactorLocalEnv()
        env.prepare()

        workspace_path_str = str(self.workspace_path)

        execute_log = env.run(
            entry=f'qrun {config_name}',
            local_path=workspace_path_str,
            env=run_env,
        )
        logger.log_object(execute_log, tag='Factor_execute_log')

        env.run(
            entry=f'{sys.executable} read_exp_res.py',
            local_path=workspace_path_str,
            env=run_env,
        )

        quantitative_backtesting_chart_path = self.workspace_path / 'ret.pkl'
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag='Quantitative Backtesting Chart')
        else:
            logger.error('No result file found.')
            return None, execute_log

        result_csv_path = self.workspace_path / 'factor_res.csv'
        legacy_result_csv_path = self.workspace_path / (('q' + 'lib') + '_res.csv')
        if not result_csv_path.exists() and legacy_result_csv_path.exists():
            result_csv_path = legacy_result_csv_path
        if result_csv_path.exists():
            pattern = (
                r'(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|'
                r'best score: -[0-9\.]+ @ \d+ epoch)'
            )
            matches = re.findall(pattern, execute_log)
            execute_log = '\n'.join(matches)
            return pd.read_csv(result_csv_path, index_col=0).iloc[:, 0], execute_log
        logger.error(f'File {result_csv_path} does not exist.')
        return None, execute_log


