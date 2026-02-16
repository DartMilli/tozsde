"""
validation_runner.py

Automatikus rendszer-validációs orkesztrátor.

Használat:
    python validation_runner.py --mode full
    python validation_runner.py --mode quick
    python validation_runner.py --mode shadow
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone

# Feltételezett modulok (implementálandók a validation/ mappában)
# Ezeket a Copilot segítségével kell majd kitölteni
from app.validation.scoring import compute_quant_score
from app.validation.report_builder import build_markdown_report
from app.data_access.data_manager import DataManager
from app.validation.errors import (
    ValidationError,
    EngineLogicError,
    DeploymentBlockedException,
)
from app.backtesting.execution_utils import seed_deterministic


class ValidationRunner:
    def __init__(self, mode: str):
        self.mode = mode
        self.results = {}

    def run_bias_tests(self):
        print("Running bias tests...")
        from app.validation.bias_tests import run_bias_tests

        self.results["bias"] = run_bias_tests()

    def run_rl_stress(self):
        print("Running RL stress tests...")
        from app.validation.rl_stress_tests import run_rl_stress_tests

        self.results["rl_stability"] = run_rl_stress_tests()

    def run_walk_forward_analysis(self):
        print("Running walk-forward analysis...")
        from app.validation.wf_analysis import run_walk_forward_analysis

        self.results["walk_forward"] = run_walk_forward_analysis()

    def run_ga_robustness(self):
        print("Running GA robustness tests...")
        from app.validation.ga_robustness import run_ga_robustness_tests

        self.results["ga_robustness"] = run_ga_robustness_tests()

    def run_shadow_comparison(self):
        print("Running shadow comparison...")
        from app.validation.shadow_compare import run_shadow_comparison

        self.results["shadow"] = run_shadow_comparison()

    def run_risk_stress(self):
        print("Running risk stress tests...")
        from app.validation.risk_stress import run_risk_stress_tests

        self.results["risk"] = run_risk_stress_tests()

    def run_execution_sensitivity(self):
        print("Running execution sensitivity analysis...")
        from app.validation.execution_sensitivity import run_execution_sensitivity

        self.results["execution_sensitivity"] = run_execution_sensitivity()

    def compute_final_score(self):
        print("Computing Quant Stability Score...")
        score_data = compute_quant_score(self.results)
        self.results["final_score"] = score_data

    def _update_engine_status(self):
        engine_integrity = {
            "status": "ENGINE_VALID",
            "issues": [],
        }
        bias = self.results.get("bias", {})
        bias_integrity = bias.get("engine_integrity") or {}
        if bias_integrity.get("trade_count_match") is False:
            engine_integrity["status"] = "ENGINE_INVALID"
            engine_integrity["issues"].append("trade_count_mismatch")
        if bias_integrity.get("trade_index_match") is False:
            engine_integrity["status"] = "ENGINE_INVALID"
            engine_integrity["issues"].append("trade_index_mismatch")

        shadow = self.results.get("shadow", {})
        equity_diff = shadow.get("equity_diff")
        if isinstance(equity_diff, (int, float)) and equity_diff != 0:
            engine_integrity["status"] = "ENGINE_INVALID"
            engine_integrity["issues"].append("equity_diff")

        self.results["engine_integrity"] = engine_integrity

        validation_status = "RESEARCH_ONLY"
        if engine_integrity["status"] == "ENGINE_INVALID":
            validation_status = "ENGINE_INVALID"
        else:
            relative_gap = bias.get("relative_gap")
            if isinstance(relative_gap, (int, float)) and abs(relative_gap) > 0.5:
                validation_status = "STRUCTURAL_EXECUTION_DRIFT"
                exec_sens = self.results.get("execution_sensitivity", {})
                if isinstance(exec_sens, dict):
                    exec_sens["warning_level"] = "HIGH"
                    self.results["execution_sensitivity"] = exec_sens
            else:
                validation_status = "ENGINE_VALID"

        self.results["validation_status"] = validation_status

    def execute(self):
        print(f"Validation mode: {self.mode}")
        os.environ["VALIDATION_MODE"] = self.mode
        os.environ["VALIDATION_DISABLE_SAFETY"] = "true"
        os.environ["VALIDATION_DISABLE_POLICY"] = "true"
        DataManager().initialize_tables()
        seed_deterministic(42)
        random.seed(42)

        try:
            if self.mode in ["quick", "full"]:
                self.run_bias_tests()
                self.run_execution_sensitivity()

            if self.mode == "full":
                self.run_walk_forward_analysis()
                self.run_ga_robustness()
                self.run_rl_stress()
                self.run_risk_stress()

            if self.mode in ["shadow", "full"]:
                self.run_shadow_comparison()

            self._update_engine_status()
            self.compute_final_score()
            if self.mode == "full":
                from app.validation.improvement_check import evaluate_results

                self.results["improvement_check"] = evaluate_results(self.results)
            self._save_reports()
        except (ValidationError, EngineLogicError) as exc:
            self.results["engine_error"] = str(exc)
            self.results["final_score"] = {
                "quant_score": 0,
                "status": "INVALID_ENGINE",
            }
            self._save_reports()
            raise
        except DeploymentBlockedException as exc:
            self.results["deployment_blocked"] = str(exc)
            self._save_reports()
            raise

    def _save_reports(self):
        os.makedirs("validation_reports", exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        json_path = f"validation_reports/validation_{timestamp}.json"
        md_path = f"validation_reports/validation_{timestamp}.md"

        # JSON mentés
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)

        # Markdown riport generálás
        md_content = build_markdown_report(self.results)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"Saved JSON report: {json_path}")
        print(f"Saved Markdown report: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Quant Validation Runner")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "shadow"],
        default="quick",
        help="Validation mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = ValidationRunner(mode=args.mode)
    runner.execute()
