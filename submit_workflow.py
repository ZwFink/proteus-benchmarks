#!/usr/bin/env python3
"""
Submit Proteus benchmark workflows to Flux.

This utility reads a YAML configuration that captures runtime defaults
and then emits `flux submit` jobs that wrap calls to `driver.py`.  Use
CLI overrides for quick what-if runs; provide `--dry-run` to inspect
commands without submitting.
"""
from __future__ import annotations

import datetime
import os
import shlex
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import click

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "PyYAML is required to load workflow configs. "
        "Install it via `python -m pip install pyyaml` and re-run."
    ) from exc

try:
    from sh import ErrorReturnCode, flux
except ImportError:  # pragma: no cover - runtime guard
    flux = None

    class ErrorReturnCode(Exception):
        """Fallback so type hints remain valid when sh is missing."""


DEFAULT_CONFIG_PATH = "workflow.yaml"
RESULTS_ROOT = Path("results")
SUPPORTED_PROFMODES = ("direct", "profiler", "metrics")


class ConfigError(RuntimeError):
    """Raised when the workflow configuration is invalid."""


def _resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _as_list(value: Optional[Sequence[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _timestamped_results_dir() -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return (RESULTS_ROOT / stamp).resolve()


def _result_suffix(prof_mode: str) -> str:
    if prof_mode == "direct":
        return "direct"
    if prof_mode == "profiler":
        return "profiler"
    if prof_mode == "metrics":
        return "profiler-metrics"
    raise ConfigError(f"Unsupported profiling mode {prof_mode}")


def _format_flux_command(args: Sequence[str]) -> str:
    return "flux " + " ".join(shlex.quote(part) for part in args)


@dataclass
class RunSpec:
    descriptor: Path
    runconfig: Path
    results_dir: Path
    benchmark: str
    exec_mode: str
    machine: str
    prof_mode: str
    reps: int
    compiler: Path
    proteus_path: Path
    driver_script: Path
    driver_extra_args: List[str]
    flux_flags: List[str]
    flux_env: Dict[str, str]
    cwd: Path

    def label(self) -> str:
        return f"{self.machine}:{self.exec_mode}:{self.prof_mode}:{self.benchmark}"

    def result_csv(self) -> Path:
        return self.results_dir / (
            f"{self.machine}-{self.benchmark}-{self.exec_mode}-results-"
            f"{_result_suffix(self.prof_mode)}.csv"
        )

    def stdouterr_log(self) -> Path:
        return self.results_dir / (
            f"{self.machine}-{self.benchmark}-{self.exec_mode}-stdouterr.txt"
        )

    def driver_command(self) -> List[str]:
        cmd = [
            "python3",
            str(self.driver_script),
            "-t",
            str(self.descriptor),
            "-c",
            str(self.compiler),
            "-j",
            str(self.proteus_path),
            "-x",
            self.exec_mode,
            "-m",
            self.machine,
            "-r",
            str(self.reps),
            "--results-dir",
            str(self.results_dir),
            "--runconfig",
            str(self.runconfig),
            "-b",
            self.benchmark,
        ]
        cmd.extend(self.driver_extra_args)
        return cmd


def load_workflow_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError("Workflow config must be a mapping at the top level.")
    return data


def build_run_specs(
    config_data: Dict,
    config_path: Path,
    descriptor_override: Optional[str],
    exec_modes_override: Sequence[str],
    benchmarks_override: Sequence[str],
    machines_override: Sequence[str],
    prof_modes_override: Sequence[str],
    result_root_override: Optional[str],
    reps_override: Optional[int],
) -> tuple[List[RunSpec], Path]:
    base = config_path.parent.resolve()
    proteus_cfg = config_data.get("proteus") or {}
    driver_cfg = config_data.get("driver") or {}
    workflow_cfg = config_data.get("workflow") or {}
    flux_cfg = config_data.get("flux") or {}

    compiler = proteus_cfg.get("cc") or os.environ.get("PROTEUS_CC")
    proteus_path = proteus_cfg.get("path") or os.environ.get("PROTEUS_PATH")
    if not compiler:
        raise ConfigError("Proteus compiler path is missing (set proteus.cc or PROTEUS_CC).")
    if not proteus_path:
        raise ConfigError("Proteus install path is missing (set proteus.path or PROTEUS_PATH).")
    compiler_path = (
        _resolve_path(base, compiler) if isinstance(compiler, str) else Path(compiler)
    )
    proteus_install_path = (
        _resolve_path(base, proteus_path)
        if isinstance(proteus_path, str)
        else Path(proteus_path)
    )
    if compiler_path is None or not compiler_path.exists():
        raise ConfigError(f"Compiler path {compiler!r} does not exist.")
    if proteus_install_path is None or not proteus_install_path.exists():
        raise ConfigError(f"Proteus path {proteus_path!r} does not exist.")

    descriptor_value = descriptor_override or driver_cfg.get("descriptor")
    if not descriptor_value:
        raise ConfigError("Driver descriptor is missing (set driver.descriptor or use --descriptor).")
    descriptor = _resolve_path(base, descriptor_value)
    if descriptor is None or not descriptor.exists():
        raise ConfigError(f"Descriptor {descriptor_value!r} does not exist.")

    runconfigs_cfg = driver_cfg.get("runconfigs") or driver_cfg.get("runconfig")
    if not isinstance(runconfigs_cfg, dict) or not runconfigs_cfg:
        raise ConfigError("Provide driver.runconfigs mapping of execution mode -> TOML path.")

    runconfig_paths = {
        mode: _resolve_path(base, path) for mode, path in runconfigs_cfg.items()
    }
    missing_runconfigs = [mode for mode, path in runconfig_paths.items() if path is None or not path.exists()]
    if missing_runconfigs:
        raise ConfigError(f"Missing runconfig files for modes: {', '.join(missing_runconfigs)}")

    driver_script_cfg = driver_cfg.get("script")
    driver_script = (
        _resolve_path(base, driver_script_cfg)
        if driver_script_cfg
        else (Path(__file__).resolve().parent / "driver.py")
    )
    if not driver_script.exists():
        raise ConfigError(f"driver.py not found at {driver_script}")

    results_dir_value = result_root_override or driver_cfg.get("results_dir")
    results_dir = (
        _resolve_path(base, results_dir_value) if results_dir_value else _timestamped_results_dir()
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine workflow axes.
    benchmarks = _as_list(benchmarks_override) or _as_list(workflow_cfg.get("benchmarks"))
    if not benchmarks:
        raise ConfigError("No benchmarks specified (set workflow.benchmarks or pass --benchmark).")

    exec_modes = _as_list(exec_modes_override) or _as_list(workflow_cfg.get("execution_modes"))
    if not exec_modes:
        exec_modes = list(runconfig_paths.keys())
    machines = _as_list(machines_override) or _as_list(workflow_cfg.get("machines"))
    if not machines:
        raise ConfigError("No machines specified (set workflow.machines or pass --machine).")
    prof_modes = _as_list(prof_modes_override) or _as_list(workflow_cfg.get("prof_modes"))
    if not prof_modes:
        prof_modes = ["direct"]

    for prof in prof_modes:
        if prof not in SUPPORTED_PROFMODES:
            raise ConfigError(f"Profiling mode {prof!r} is not supported.")

    reps = reps_override or driver_cfg.get("reps") or driver_cfg.get("iterations")
    if reps is None:
        raise ConfigError("Repetitions for driver are missing (set driver.reps or pass --reps).")

    driver_extra_args = _as_list(driver_cfg.get("extra_args"))
    driver_extra_env = {str(k): str(v) for k, v in (driver_cfg.get("env") or {}).items()}

    flux_flags = _as_list(flux_cfg.get("options"))
    flux_env = {str(k): str(v) for k, v in (flux_cfg.get("env") or {}).items()}
    flux_cwd = _resolve_path(base, flux_cfg.get("cwd")) or Path.cwd()

    mode_overrides = workflow_cfg.get("mode_overrides") or {}

    run_specs: List[RunSpec] = []
    for exec_mode, machine, prof_mode, benchmark in product(exec_modes, machines, prof_modes, benchmarks):
        if exec_mode not in runconfig_paths:
            raise ConfigError(f"Execution mode {exec_mode!r} is not defined in driver.runconfigs.")
        runconfig = runconfig_paths[exec_mode]
        overrides = mode_overrides.get(exec_mode, {})
        mode_driver_args = driver_extra_args + _as_list(overrides.get("driver_extra_args"))
        mode_flux_flags = flux_flags + _as_list(overrides.get("flux_options"))
        mode_flux_env = {
            **flux_env,
            **{str(k): str(v) for k, v in (overrides.get("flux_env") or {}).items()},
        }
        env_updates = {
            **mode_flux_env,
            **driver_extra_env,
            "PROTEUS_PATH": str(proteus_install_path),
            "PROTEUS_CC": str(compiler_path),
        }

        run_specs.append(
            RunSpec(
                descriptor=descriptor,
                runconfig=runconfig,
                results_dir=results_dir,
                benchmark=benchmark,
                exec_mode=exec_mode,
                machine=machine,
                prof_mode=prof_mode,
                reps=int(reps),
                compiler=compiler_path,
                proteus_path=proteus_install_path,
                driver_script=driver_script,
                driver_extra_args=mode_driver_args,
                flux_flags=mode_flux_flags,
                flux_env={str(k): str(v) for k, v in env_updates.items()},
                cwd=flux_cwd,
            )
        )

    return run_specs, results_dir


def submit_run(
    run: RunSpec,
    dry_run: bool,
    force: bool,
    retries: int,
    verbose: bool,
) -> Dict[str, Optional[str]]:
    result_csv = run.result_csv()
    if result_csv.exists() and result_csv.stat().st_size > 0 and not force:
        return {"status": "skipped", "reason": "results-exist", "run": run}

    stdouterr_path = run.stdouterr_log()

    sanitized_flags: List[str] = []
    skip_next = False
    for index, flag in enumerate(run.flux_flags):
        if skip_next:
            skip_next = False
            continue
        if flag in {"--output", "-o", "--error", "-e"}:
            skip_next = True
            continue
        if flag.startswith("--output=") or flag.startswith("--error="):
            continue
        if flag.startswith("-o") and flag != "-o":
            continue
        if flag.startswith("-e") and flag != "-e":
            continue
        sanitized_flags.append(flag)

    driver_cmd = run.driver_command()
    parts: List[str] = []
    parts.append(shlex.join(driver_cmd))
    activation_cmd = " && ".join(p for p in parts if p)

    flux_command = [
        "submit",
        *sanitized_flags,
        "--output",
        str(stdouterr_path),
        "--error",
        str(stdouterr_path),
        "bash",
        "-lc",
        activation_cmd,
    ]
    command_str = _format_flux_command(flux_command)

    if dry_run:
        if verbose:
            click.echo(f"[DRY-RUN] {command_str}")
        return {"status": "dry-run", "command": command_str, "run": run}

    if flux is None:
        raise ConfigError("The `sh` library is required to submit jobs (pip install sh).")

    attempt = 0
    last_error: Optional[str] = None
    attempts_total = retries + 1
    while attempt <= retries:
        try:
            env = os.environ.copy()
            env.update(run.flux_env)
            if verbose:
                attempt_suffix = (
                    f" (attempt {attempt + 1}/{attempts_total})" if attempts_total > 1 else ""
                )
                click.echo(f"Submitting: {command_str}{attempt_suffix}")
            submission = flux(
                *flux_command,
                _env=env,
                _cwd=str(run.cwd),
            )
            jobid = str(submission).strip()
            return {"status": "submitted", "jobid": jobid, "command": command_str, "run": run}
        except ErrorReturnCode as exc:  # pragma: no cover - flux handles this
            last_error = exc.stderr.decode() if hasattr(exc, "stderr") else str(exc)
            attempt += 1
            if attempt > retries:
                return {
                    "status": "failed",
                    "error": last_error or "flux submit failed",
                    "command": command_str,
                    "run": run,
                }

    return {
        "status": "failed",
        "error": last_error or "Unknown submission failure",
        "command": command_str,
        "run": run,
    }


def _format_run_summary(result: Dict[str, Optional[str]]) -> str:
    run: RunSpec = result["run"]
    prefix = f"[{result['status'].upper():8}] {run.label()}"
    if result["status"] == "submitted":
        return f"{prefix} -> job {result.get('jobid')}"
    if result["status"] == "dry-run":
        return f"{prefix} (dry run) {result.get('command')}"
    if result["status"] == "skipped":
        return f"{prefix} (skip: {result.get('reason')})"
    if result["status"] == "failed":
        return f"{prefix} (failed: {result.get('error')})"
    return prefix


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_CONFIG_PATH,
    help="YAML file describing the workflow defaults.",
)
@click.option(
    "--descriptor",
    "descriptor_override",
    type=click.Path(path_type=str, dir_okay=False),
    help="Override benchmark descriptor path.",
)
@click.option("--exec-mode", "exec_modes_override", multiple=True, help="Restrict to one or more execution modes.")
@click.option("--benchmark", "benchmarks_override", multiple=True, help="Restrict to specific benchmarks.")
@click.option("--machine", "machines_override", multiple=True, help="Restrict to specific machines.")
@click.option("--prof-mode", "prof_modes_override", multiple=True, help="Restrict to specific profiling modes.")
@click.option(
    "--result-root",
    "result_root_override",
    type=click.Path(path_type=str),
    help="Override the results directory.",
)
@click.option("--reps", "reps_override", type=int, help="Override repetitions passed to driver.py.")
@click.option("--dry-run", is_flag=True, help="Print flux commands without submitting them.")
@click.option("--force", is_flag=True, help="Submit even if results already exist.")
@click.option("--retries", default=0, show_default=True, type=int, help="Retry failed submissions.")
@click.option("--verbose", "-v", is_flag=True, help="Print flux submit commands as they run.")
def main(
    config_path: Path,
    descriptor_override: Optional[str],
    exec_modes_override: Sequence[str],
    benchmarks_override: Sequence[str],
    machines_override: Sequence[str],
    prof_modes_override: Sequence[str],
    result_root_override: Optional[str],
    reps_override: Optional[int],
    dry_run: bool,
    force: bool,
    retries: int,
    verbose: bool,
) -> None:
    """Submit driver.py workflows to Flux."""
    try:
        config_data = load_workflow_config(config_path)
        run_specs, results_dir = build_run_specs(
            config_data=config_data,
            config_path=config_path,
            descriptor_override=descriptor_override,
            exec_modes_override=exec_modes_override,
            benchmarks_override=benchmarks_override,
            machines_override=machines_override,
            prof_modes_override=prof_modes_override,
            result_root_override=result_root_override,
            reps_override=reps_override,
        )
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    click.echo(f"Results directory: {results_dir}")
    if dry_run:
        click.echo("Dry run enabled; no jobs will be queued.")

    outcomes: List[Dict[str, Optional[str]]] = []
    for run in run_specs:
        result = submit_run(
            run,
            dry_run=dry_run,
            force=force,
            retries=retries,
            verbose=verbose,
        )
        outcomes.append(result)
        click.echo(_format_run_summary(result))

    submitted = sum(1 for outcome in outcomes if outcome["status"] == "submitted")
    skipped = sum(1 for outcome in outcomes if outcome["status"] == "skipped")
    dry = sum(1 for outcome in outcomes if outcome["status"] == "dry-run")
    failed = sum(1 for outcome in outcomes if outcome["status"] == "failed")

    click.echo(
        f"Summary: submitted={submitted}, skipped={skipped}, dry_run={dry}, failed={failed}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
