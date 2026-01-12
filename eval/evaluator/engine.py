from typing import Callable, Dict, List, Optional, Union

from eval.config.eval import EvalEngineConfig
from eval.evaluator.evaluator import build_evaluator
from eval.interfaces import Evaluator, Reporter
from eval.report.reporter import build_reporter
from eval.types import EvalContext, EvalReport, EvalResult


class EvalEngine:
    def __init__(
        self,
        evaluators: List[Evaluator],
        reporters: List[Reporter] ,
        strict: bool = True,
        on_error: Optional[Callable[[str, Exception, EvalContext], None]] = None,
    ):
        self._evaluators = list(evaluators)
        self._strict = strict
        self._on_error = on_error
        self._reporters = reporters  # override ctx.run.reporter if provided

        # validate evaluator name uniqueness early (prevents score dict collisions)
        names = [e.name for e in self._evaluators]
        duplicate = {n for n in names if names.count(n) > 1}
        if duplicate:
            raise ValueError(f"Duplicate evaluator names: {sorted(duplicate)}")

    @property
    def evaluators(self) -> List[Evaluator]:
        return self._evaluators

    def run(self, context: EvalContext) -> EvalReport:
        results: Dict[str, EvalResult] = dict()
        errors: Dict[str, str] = dict()

        for ev in self.evaluators:
            try:
                r = ev.unit_evaluate(context)
                results[ev.name] = r
            except Exception as ex:
                # store error string (keep it concise; full trace can go to logger)
                errors[ev.name] = f"{type(ex).__name__}: {ex}"

                if self._on_error is not None:
                    try:
                        self._on_error(ev.name, ex, context)
                    except Exception:
                        # never let error handler crash engine
                        pass

                if self._strict:
                    report = self._build_report(context, results, errors, ok=False)
                    self._maybe_report(context, report)
                    raise  # preserve original exception
                # non-strict: continue evaluating others
        
        ok = (len(errors) == 0)
        report = self._build_report(context, results, errors, ok=ok)
        self._maybe_report(context, report)
        return report

    def batch_run(self, contexts):
        reports = list() 
        for context in contexts: 
            report = self.run(context)
            reports.append(report)
        
        return reports 

    def _build_report(
        self,
        context: EvalContext,
        results: Dict[str, EvalResult],
        errors: Dict[str, str],
        ok: bool,
    ) -> EvalReport:
        scores: Dict[str, Union[int, float, None]] = dict() 
        for r_name in results: 
            score = results[r_name].metric.score
            if score is not None: 
                scores[r_name] = int(score)
            else: 
                scores[r_name] = None 

        score_mean: Optional[float] = None
        if scores:
            valid_scores: List[Union[int, float]] = list() 
            for score in scores.values():
                if score is not None:
                    valid_scores.append(score)

            if len(valid_scores) > 0: 
                score_mean = sum(valid_scores) / len(valid_scores)
            else:
                score_mean = None

        return EvalReport(
            run_id=context.run_id,
            sample_id=context.sample.sample_id,
            results=results,
            scores=scores,
            ok=ok,
            errors=dict(errors),
            score_mean=score_mean,
        )

    def _maybe_report(self, context: EvalContext, report: EvalReport) -> None:
        try:
            for reporter in self._reporters:
                reporter.report_one(context, report)
        except Exception:
            if self._strict:
                raise



def build_eval_engine(config: EvalEngineConfig):
    evaluators = [
        build_evaluator(evaluator_config)
        for evaluator_config in config.evaluators
    ]
    reporters = [
        build_reporter(reporter_config)
        for reporter_config in config.reporters
    ]

    return EvalEngine(
        evaluators = evaluators, 
        reporters = reporters, 
        strict = config.strict
    )
