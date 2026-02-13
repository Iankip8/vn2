#!/usr/bin/env python
"""
Data Loading Phase Walkthrough
===============================

This script walks through Phase 1: Data Ingestion step-by-step,
showing what data exists and how it's structured.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rich import print as rprint
from rich.table import Table
from rich.console import Console

console = Console()


def section_header(title):
    """Print a section header"""
    console.rule(f"[bold blue]{title}[/bold blue]")
    print()


def main():
    """Walk through the data loading phase"""
    
    console.print("\n[bold green]═══════════════════════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]     VN2 DATA LOADING PHASE - INTERACTIVE WALKTHROUGH             [/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════════════════════[/bold green]\n")
    
    # ==================================================================
    # STEP 1: Raw Data Files
    # ==================================================================
    section_header("STEP 1: Raw Data Files")
    
    raw_path = Path('data/raw')
    rprint("[cyan]📁 Raw data directory contains competition input files:[/cyan]\n")
    
    raw_files = {
        'Week 0 - 2024-04-08 - Sales.csv': 'Historical sales (weekly) from 2021-2024',
        'Week 0 - In Stock.csv': 'Current on-hand inventory per SKU',
        'Week 0 - Master.csv': 'SKU metadata (product hierarchy, store format)',
        'Week 0 - Submission Template.csv': 'Template defining required SKU ordering',
        'Week 0 - 2024-04-08 - Initial State.csv': 'Initial state (backup)',
    }
    
    for fname, description in raw_files.items():
        fpath = raw_path / fname
        if fpath.exists():
            size = fpath.stat().st_size / 1024
            rprint(f"  ✓ [green]{fname:45s}[/green] ({size:7.1f} KB)")
            rprint(f"    → {description}")
        else:
            rprint(f"  ✗ [red]{fname:45s}[/red] MISSING")
    
    # ==================================================================
    # STEP 2: Load Submission Index
    # ==================================================================
    section_header("STEP 2: Submission Index (SKU List)")
    
    rprint("[cyan]📋 Loading canonical SKU index from submission template...[/cyan]\n")
    
    from vn2.data import submission_index
    idx = submission_index('data/raw')
    
    rprint(f"  [green]Total SKUs: {len(idx):,}[/green]")
    rprint(f"  Index type: MultiIndex (Store, Product)")
    rprint(f"  Example SKUs:")
    for i, (store, product) in enumerate(idx[:5]):
        rprint(f"    {i+1}. Store={store}, Product={product}")
    rprint(f"    ...")
    
    # ==================================================================
    # STEP 3: Load Initial State
    # ==================================================================
    section_header("STEP 3: Initial Inventory State")
    
    rprint("[cyan]📦 Loading initial inventory state...[/cyan]\n")
    
    from vn2.data import load_initial_state
    state = load_initial_state('data/raw', idx)
    
    rprint(f"  [green]State DataFrame:[/green]")
    rprint(f"    Shape: {state.shape[0]:,} SKUs × {state.shape[1]} columns")
    rprint(f"    Columns: {list(state.columns)}")
    rprint(f"\n  [yellow]Inventory Statistics:[/yellow]")
    rprint(f"    Total on-hand: {state['on_hand'].sum():,.0f} units")
    rprint(f"    Mean on-hand: {state['on_hand'].mean():.2f} units per SKU")
    rprint(f"    Max on-hand: {state['on_hand'].max():.0f} units")
    rprint(f"    SKUs with zero inventory: {(state['on_hand'] == 0).sum():,} ({(state['on_hand'] == 0).mean()*100:.1f}%)")
    
    rprint(f"\n  [yellow]Sample state (first 5 SKUs):[/yellow]")
    rprint(state.head())
    
    # ==================================================================
    # STEP 4: Load Master Data
    # ==================================================================
    section_header("STEP 4: Master Data (SKU Metadata)")
    
    rprint("[cyan]🏷️  Loading master product/store hierarchy...[/cyan]\n")
    
    from vn2.data import load_master
    master = load_master('data/raw')
    
    rprint(f"  [green]Master DataFrame:[/green]")
    rprint(f"    Shape: {master.shape[0]:,} SKUs × {master.shape[1]} columns")
    rprint(f"    Columns: {list(master.columns)}")
    
    # Show hierarchy levels
    rprint(f"\n  [yellow]Hierarchy Levels:[/yellow]")
    for col in master.columns:
        n_unique = master[col].nunique()
        rprint(f"    {col:20s}: {n_unique:4d} unique values")
    
    rprint(f"\n  [yellow]Sample master data:[/yellow]")
    rprint(master.head())
    
    # ==================================================================
    # STEP 5: Load Sales History
    # ==================================================================
    section_header("STEP 5: Sales History")
    
    rprint("[cyan]📈 Loading historical sales data...[/cyan]\n")
    
    from vn2.data import load_sales
    sales = load_sales('data/raw')
    
    rprint(f"  [green]Sales DataFrame:[/green]")
    rprint(f"    Shape: {sales.shape[0]:,} SKUs × {sales.shape[1]} columns")
    rprint(f"    First few columns: {list(sales.columns[:5])}")
    
    # Identify date columns
    date_cols = [c for c in sales.columns if c not in ['Store', 'Product']]
    rprint(f"    Date columns: {len(date_cols)} weeks")
    rprint(f"    Date range: {date_cols[0]} to {date_cols[-1]}")
    
    # Statistics
    sales_data = sales[date_cols]
    rprint(f"\n  [yellow]Sales Statistics:[/yellow]")
    rprint(f"    Total sales: {sales_data.sum().sum():,.0f} units")
    rprint(f"    Mean weekly sales per SKU: {sales_data.mean(axis=1).mean():.2f} units")
    rprint(f"    Weeks with zero sales: {(sales_data == 0).sum().sum():,} ({(sales_data == 0).mean().mean()*100:.1f}%)")
    
    # ==================================================================
    # STEP 6: Check Interim Data
    # ==================================================================
    section_header("STEP 6: Interim Data (Post-Ingestion)")
    
    interim_path = Path('data/interim')
    
    rprint("[cyan]💾 Checking what's been saved to interim/...[/cyan]\n")
    
    if interim_path.exists():
        interim_files = list(interim_path.glob('*.parquet'))
        if interim_files:
            rprint(f"  [green]Found {len(interim_files)} interim file(s):[/green]\n")
            for fpath in sorted(interim_files):
                size_mb = fpath.stat().st_size / 1024 / 1024
                rprint(f"    ✓ {fpath.name:30s} ({size_mb:6.2f} MB)")
                
                # Load and show summary
                df = pd.read_parquet(fpath)
                rprint(f"      Shape: {df.shape[0]:,} × {df.shape[1]}")
        else:
            rprint("  [yellow]No interim files found. Run: ./go ingest --raw data/raw --out data/interim[/yellow]")
    else:
        rprint("  [yellow]Interim directory doesn't exist yet.[/yellow]")
    
    # ==================================================================
    # STEP 7: Check Processed Data (EDA)
    # ==================================================================
    section_header("STEP 7: Processed Data (From EDA)")
    
    processed_path = Path('data/processed')
    
    rprint("[cyan]🔬 Checking processed data from EDA notebook...[/cyan]\n")
    
    if processed_path.exists():
        processed_files = list(processed_path.glob('*.parquet'))
        if processed_files:
            rprint(f"  [green]Found {len(processed_files)} processed file(s):[/green]\n")
            for fpath in sorted(processed_files):
                size_mb = fpath.stat().st_size / 1024 / 1024
                rprint(f"    ✓ {fpath.name:40s} ({size_mb:6.2f} MB)")
                
                # Show details for key files
                if fpath.name == 'demand_long.parquet':
                    df = pd.read_parquet(fpath)
                    rprint(f"      Shape: {df.shape[0]:,} observations × {df.shape[1]} columns")
                    rprint(f"      Columns: {list(df.columns)}")
                    if 'in_stock' in df.columns:
                        n_stockout = (~df['in_stock']).sum()
                        pct_stockout = (~df['in_stock']).mean() * 100
                        rprint(f"      Stockouts: {n_stockout:,} ({pct_stockout:.1f}%)")
        else:
            rprint("  [yellow]No processed files found. Run EDA notebook first.[/yellow]")
    else:
        rprint("  [yellow]Processed directory doesn't exist yet.[/yellow]")
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    section_header("Summary & Next Steps")
    
    rprint("[bold green]✅ Data Loading Phase Complete![/bold green]\n")
    
    rprint("[cyan]What we have:[/cyan]")
    rprint("  • Raw data: Sales history, inventory state, SKU metadata")
    rprint("  • Interim data: Parquet files ready for modeling")
    rprint("  • Processed data: Long-format demand with stockout flags\n")
    
    rprint("[cyan]Next steps:[/cyan]")
    rprint("  1. If interim/ is empty:")
    rprint("     → Run: [bold]./go ingest --raw data/raw --out data/interim[/bold]")
    rprint("\n  2. If processed/demand_long.parquet is missing:")
    rprint("     → Run EDA notebook: [bold]notebooks/02_comprehensive_time_series_eda.ipynb[/bold]")
    rprint("\n  3. Then move to Phase 3:")
    rprint("     → Run: [bold]./go impute --n-neighbors 20 --n-jobs -1[/bold]\n")


if __name__ == "__main__":
    main()
