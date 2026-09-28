from quantaalpha.components.runner import CachedRunner
from quantaalpha.core.exception import ModelEmptyError
from quantaalpha.core.utils import cache_with_pickle
from quantaalpha.contrib.model.experiment import ModelResearchExperiment


class ModelRunner(CachedRunner[ModelResearchExperiment]):
    """Execute optional model research experiments in the project workspace."""

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: ModelResearchExperiment) -> ModelResearchExperiment:
        if exp.sub_workspace_list[0].code_dict.get('model.py') is None:
            raise ModelEmptyError('model.py is empty')
        exp.experiment_workspace.inject_code(**{'model.py': exp.sub_workspace_list[0].code_dict['model.py']})

        env_to_use = {'PYTHONPATH': './'}
        if exp.sub_tasks[0].model_type == 'TimeSeries':
            env_to_use.update({'dataset_cls': 'TSDatasetH', 'step_len': 20, 'num_timesteps': 20})
        elif exp.sub_tasks[0].model_type == 'Tabular':
            env_to_use.update({'dataset_cls': 'DatasetH'})

        result = exp.experiment_workspace.execute(config_name='conf.yaml', run_env=env_to_use)
        exp.result = result
        return exp
