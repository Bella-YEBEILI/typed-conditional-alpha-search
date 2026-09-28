import pickle
from pathlib import Path

from quantaalpha.coder.costeer.config import CoSTEERSettings
from quantaalpha.coder.costeer.evolvable_subjects import EvolvingItem
from quantaalpha.coder.costeer.evolving_agent import FilterFailedRAGEvoAgent
from quantaalpha.coder.costeer.knowledge_management import (
    CoSTEERKnowledgeBaseV1,
    CoSTEERKnowledgeBaseV2,
    CoSTEERRAGStrategyV1,
    CoSTEERRAGStrategyV2,
)
from quantaalpha.core.developer import Developer
from quantaalpha.core.evaluation import Evaluator
from quantaalpha.core.evolving_agent import EvolvingStrategy
from quantaalpha.core.experiment import Experiment
from quantaalpha.log import logger


class CoSTEER(Developer[Experiment]):
    def __init__(
        self,
        settings: CoSTEERSettings,
        eva: Evaluator,
        es: EvolvingStrategy,
        evolving_version: int,
        *args,
        with_knowledge: bool = True,
        with_feedback: bool = True,
        knowledge_self_gen: bool = True,
        filter_final_evo: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_loop = settings.max_loop
        self.knowledge_base_path = (
            Path(settings.knowledge_base_path) if settings.knowledge_base_path is not None else None
        )
        self.new_knowledge_base_path = (
            Path(settings.new_knowledge_base_path) if settings.new_knowledge_base_path is not None else None
        )

        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen
        self.filter_final_evo = filter_final_evo
        self.evolving_strategy = es
        self.evaluator = eva
        self.evolving_version = evolving_version

        # init knowledge base
        self.knowledge_base = self.load_or_init_knowledge_base(
            former_knowledge_base_path=self.knowledge_base_path,
            component_init_list=[],
        )
        # init rag method
        self.rag = (
            CoSTEERRAGStrategyV2(self.knowledge_base, settings=settings)
            if self.evolving_version == 2
            else CoSTEERRAGStrategyV1(self.knowledge_base, settings=settings)
        )

    def load_or_init_knowledge_base(self, former_knowledge_base_path: Path = None, component_init_list: list = []):
        if former_knowledge_base_path is not None and former_knowledge_base_path.exists():
            knowledge_base = pickle.load(open(former_knowledge_base_path, "rb"))
            if self.evolving_version == 1 and not isinstance(knowledge_base, CoSTEERKnowledgeBaseV1):
                raise ValueError("The former knowledge base is not compatible with the current version")
            elif self.evolving_version == 2 and not isinstance(
                knowledge_base,
                CoSTEERKnowledgeBaseV2,
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
        else:
            knowledge_base = (
                CoSTEERKnowledgeBaseV2(
                    init_component_list=component_init_list,
                )
                if self.evolving_version == 2
                else CoSTEERKnowledgeBaseV1()
            )
        return knowledge_base

    def develop(self, exp: Experiment) -> Experiment:

        # init intermediate items
        experiment = EvolvingItem.from_experiment(exp)

        self.evolve_agent = FilterFailedRAGEvoAgent(
            max_loop=self.max_loop,
            evolving_strategy=self.evolving_strategy,
            rag=self.rag,
            with_knowledge=self.with_knowledge,
            with_feedback=self.with_feedback,
            knowledge_self_gen=self.knowledge_self_gen,
        )

        experiment = self.evolve_agent.multistep_evolve(
            experiment,
            self.evaluator,
            filter_final_evo=self.filter_final_evo,
        )

        # save new knowledge base
        if self.new_knowledge_base_path is not None:
            pickle.dump(self.knowledge_base, open(self.new_knowledge_base_path, "wb"))
            logger.info(f"New knowledge base saved to {self.new_knowledge_base_path}")

        passed_tasks = []
        passed_workspaces = []
        skipped_tasks = []
        selection_applied = bool(getattr(experiment, "implementation_selection_applied", False))
        for idx, task in enumerate(list(getattr(experiment, "sub_tasks", []) or [])):
            workspace = (
                experiment.sub_workspace_list[idx]
                if idx < len(getattr(experiment, "sub_workspace_list", []) or [])
                else None
            )
            passed = bool(getattr(task, "factor_implementation", False)) or (
                selection_applied and workspace is not None
            )
            if passed and workspace is not None:
                try:
                    task.factor_implementation = True
                except Exception:
                    pass
                passed_tasks.append(task)
                passed_workspaces.append(workspace)
            else:
                skipped_tasks.append(str(getattr(task, "factor_name", getattr(task, "name", f"task_{idx}"))))

        if passed_tasks:
            logger.info(
                "Passing coder-approved factors to TQ runner: "
                + ", ".join(str(getattr(task, "factor_name", getattr(task, "name", "unknown"))) for task in passed_tasks)
            )

        if skipped_tasks:
            logger.warning(
                "Discarding generated factor implementations that did not pass coder final_decision "
                "before TQ runner: " + ", ".join(skipped_tasks)
            )

        exp.sub_tasks = passed_tasks
        exp.sub_workspace_list = passed_workspaces
        exp.implementation_selection_applied = True
        return exp
