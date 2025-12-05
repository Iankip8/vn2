#!/usr/bin/env python3
"""
Run all hypothesis validation tests and generate consolidated report.

This script:
1. Runs H1 (Jensen Gap) test
2. Runs H2 (Stockout Awareness) test
3. Runs H3 (SURD Effect) test
4. Compiles H4 (Lead Time) documentation
5. Generates consolidated hypothesis validation report

Usage:
    python scripts/run_hypothesis_validation.py \
        --checkpoints-dir models/checkpoints_h3 \
        --demand-path data/processed/demand_long.parquet \
        --output-dir reports/hypothesis_tests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def run_test(script_name: str, args: list, output_dir: Path) -> dict:
    """Run a hypothesis test script and return results."""
    cmd = [sys.executable, f"scripts/{script_name}"] + args
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print('='*70)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        success = result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        success = False
    
    return {'script': script_name, 'success': success}


def compile_report(output_dir: Path) -> None:
    """Compile consolidated hypothesis validation report."""
    
    report = []
    report.append("# Hypothesis Validation Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")
    
    # Summary table
    report.append("## Summary\n")
    report.append("| Hypothesis | Description | Verdict |")
    report.append("|------------|-------------|---------|")
    
    verdicts = {}
    
    # H1
    h1_path = output_dir / 'h1_jensen_gap_results.csv'
    if h1_path.exists():
        df = pd.read_csv(h1_path)
        supported = df['hypothesis_supported'].sum() if 'hypothesis_supported' in df.columns else 0
        total = len(df[~df.get('error', pd.Series(dtype=str)).notna()])
        verdict = "SUPPORTED" if supported > total/2 else "NOT SUPPORTED"
        verdicts['H1'] = verdict
        report.append(f"| H1 | Jensen Gap: Density-aware SIP better than point+SL | {verdict} |")
    else:
        report.append("| H1 | Jensen Gap | NOT RUN |")
    
    # H2
    h2_path = output_dir / 'h2_stockout_awareness_results.csv'
    if h2_path.exists():
        df = pd.read_csv(h2_path)
        if 'mean_delta' in df.columns and 'significant' in df.columns:
            verdict = "SUPPORTED" if df['mean_delta'].iloc[0] > 0 and df['significant'].iloc[0] else "NOT SUPPORTED"
        else:
            verdict = "INCONCLUSIVE"
        verdicts['H2'] = verdict
        report.append(f"| H2 | Stockout Awareness: Interval targets better | {verdict} |")
    else:
        report.append("| H2 | Stockout Awareness | NOT RUN |")
    
    # H3
    h3_path = output_dir / 'h3_surd_effect_results.csv'
    if h3_path.exists():
        df = pd.read_csv(h3_path)
        if 'hypothesis_supported' in df.columns:
            verdict = "SUPPORTED" if df['hypothesis_supported'].iloc[0] else "NOT SUPPORTED"
        else:
            verdict = "INCONCLUSIVE"
        verdicts['H3'] = verdict
        report.append(f"| H3 | SURD Effect: Transforms improve calibration | {verdict} |")
    else:
        report.append("| H3 | SURD Effect | NOT RUN |")
    
    # H4
    h4_path = output_dir / 'h4_lead_time_report.md'
    if h4_path.exists():
        verdict = "CONFIRMED"
        verdicts['H4'] = verdict
        report.append(f"| H4 | Lead Time: L=2 vs L=3 implementation bug | {verdict} |")
    else:
        report.append("| H4 | Lead Time | NOT DOCUMENTED |")
    
    report.append("\n---\n")
    
    # Include individual reports
    for test in ['h1', 'h2', 'h3', 'h4']:
        md_path = output_dir / f'{test}_*_report.md'
        for f in output_dir.glob(f'{test}_*_report.md'):
            report.append(f"\n## {f.stem}\n")
            with open(f, 'r') as fp:
                content = fp.read()
                # Skip the title if present (we've already added section header)
                lines = content.split('\n')
                if lines and lines[0].startswith('#'):
                    content = '\n'.join(lines[1:])
                report.append(content)
    
    # Research implications
    report.append("\n---\n")
    report.append("## Research Implications\n")
    
    supported_count = sum(1 for v in verdicts.values() if v == "SUPPORTED")
    total_count = len(verdicts)
    
    if supported_count >= 3:
        report.append("**Overall Assessment: Research objectives ACHIEVED**\n")
        report.append("The majority of hypotheses were validated, indicating that:\n")
        report.append("1. Density-aware optimization provides value over point forecasts\n")
        report.append("2. Stockout-aware training improves shortage predictions\n")
        report.append("3. SURD transforms enhance forecast calibration\n")
    elif supported_count >= 2:
        report.append("**Overall Assessment: Research objectives PARTIALLY ACHIEVED**\n")
        report.append("Some hypotheses were validated but not all. Key learnings:\n")
    else:
        report.append("**Overall Assessment: Research objectives NOT FULLY ACHIEVED**\n")
        report.append("The hypotheses were not strongly supported by the data.\n")
        report.append("This may indicate:\n")
        report.append("- Forecast quality issues dominate methodology differences\n")
        report.append("- Need for larger sample sizes or longer evaluation periods\n")
        report.append("- Different competition characteristics than assumed\n")
    
    # Save report
    report_path = output_dir / 'hypothesis_validation.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nConsolidated report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all hypothesis validation tests")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models/checkpoints_h3"))
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hypothesis_tests"))
    parser.add_argument("--max-skus", type=int, default=300)
    parser.add_argument("--skip-tests", nargs="+", default=[], help="Tests to skip (h1, h2, h3)")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("HYPOTHESIS VALIDATION SUITE")
    print("="*70)
    
    common_args = [
        f"--checkpoints-dir={args.checkpoints_dir}",
        f"--demand-path={args.demand_path}",
        f"--output-dir={args.output_dir}",
        f"--max-skus={args.max_skus}"
    ]
    
    results = []
    
    # H1: Jensen Gap
    if 'h1' not in args.skip_tests:
        result = run_test('test_h1_jensen_gap.py', common_args, args.output_dir)
        results.append(result)
    
    # H2: Stockout Awareness
    if 'h2' not in args.skip_tests:
        result = run_test('test_h2_stockout_awareness.py', common_args, args.output_dir)
        results.append(result)
    
    # H3: SURD Effect
    if 'h3' not in args.skip_tests:
        result = run_test('test_h3_surd_effect.py', common_args, args.output_dir)
        results.append(result)
    
    # Compile consolidated report
    print("\n" + "="*70)
    print("COMPILING CONSOLIDATED REPORT")
    print("="*70)
    compile_report(args.output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("TEST EXECUTION SUMMARY")
    print("="*70)
    
    for r in results:
        status = "✓ PASSED" if r['success'] else "✗ FAILED"
        print(f"  {r['script']}: {status}")
    
    print(f"\nResults directory: {args.output_dir}")
    print("Run complete!")


if __name__ == "__main__":
    main()

