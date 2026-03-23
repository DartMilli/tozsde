from app.analysis.decision_quality_analyzer import DecisionQualityAnalyzer
from app.analysis.confidence_calibrator import ConfidenceCalibrator
from app.analysis.wf_stability_analyzer import WalkForwardStabilityAnalyzer
from app.analysis.safety_stress_tester import SafetyStressTester
from app.analysis.validation_report_builder import ValidationReportBuilder
from app.application.use_cases.result import ok, UseCaseResult


class RunPhase5ValidationUseCase:
    def __init__(self, data_manager):
        self._data_manager = data_manager

    def run(
        self,
        ticker: str = None,
        start_date: str = None,
        end_date: str = None,
        scenario: str = "elevated_volatility",
        include_calibration: bool = True,
        repeat: int = 1,
        compare_repeat: bool = False,
    ) -> UseCaseResult:
        try:
            self._data_manager.initialize_tables()
        except Exception:
            pass

        builder = ValidationReportBuilder()
        reports = []

        for _ in range(repeat):
            DecisionQualityAnalyzer().analyze(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            if include_calibration:
                ConfidenceCalibrator().compute(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                )

            if ticker:
                WalkForwardStabilityAnalyzer().analyze(ticker=ticker)

                if start_date and end_date:
                    SafetyStressTester().run(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        scenario=scenario,
                    )

            report = builder.build()
            reports.append(report)

        return ok(
            "run_phase5_validation",
            data={
                "reports": reports,
                "repeat_match": (
                    reports[-1] == reports[-2]
                    if compare_repeat and len(reports) >= 2
                    else None
                ),
            },
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            scenario=scenario,
            include_calibration=include_calibration,
            repeat=repeat,
            compare_repeat=compare_repeat,
        )
