#!/usr/bin/env python3
"""
Orchestrator script for SLURM activities
This script provides commands for downloading and processing Amazon product data
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import logging
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel

# Initialize Typer app and console
app = typer.Typer()
console = Console()

def load_categories(categories_file: str = "all_categories.txt") -> List[str]:
    """
    Load categories from the specified text file.
    
    Args:
        categories_file: Path to the text file containing category names
        
    Returns:
        List of category names (excluding empty lines)
    """
    categories = []
    try:
        with open(categories_file, 'r') as f:
            for line in f:
                category = line.strip()
                if category:  # Skip empty lines
                    categories.append(category)
        return categories
    except FileNotFoundError:
        console.print(f"[red]Error: Categories file '{categories_file}' not found.[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error reading categories file: {e}[/red]")
        sys.exit(1)

def category_download(category: str, work_dir: str, use_slurm: bool = False) -> bool:
    """
    Process a single category download using dataset_builder.py
    
    Args:
        category: Category name to process
        work_dir: Working directory for the dataset builder
        use_slurm: Whether to use SLURM-friendly output
        
    Returns:
        True if successful, False otherwise
    """
    if use_slurm:
        print(f"Downloading data for category: {category}", flush=True)
    else:
        console.print(f"[blue]Downloading data for category: {category}[/blue]")
    
    try:
        # Call dataset_builder.py with the download-huggingface command
        cmd = [
            sys.executable,  # Use current Python interpreter
            "/home/damonp/projects/chimera_proj/amazon-data/lib/dataset_builder.py",
            "download-huggingface",
        ]
        if use_slurm:
            cmd.append("--slurm")
        cmd.extend([
            "--category", category,
            "--work-dir", work_dir
        ])
        
        # Remove empty strings from command
        cmd = [arg for arg in cmd if arg]
        
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        
        if result.returncode == 0:
            if use_slurm:
                print(f"Completed download for category: {category}", flush=True)
                print("----------------------------------------", flush=True)
            else:
                console.print(f"[green]Completed download for category: {category}[/green]")
            return True
        else:
            if use_slurm:
                print(f"Error processing category {category}:", flush=True)
                print(f"STDOUT: {result.stdout}", flush=True)
                print(f"STDERR: {result.stderr}", flush=True)
            else:
                console.print(f"[red]Error processing category {category}:[/red]")
                console.print(f"STDOUT: {result.stdout}")
                console.print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        error_msg = f"Exception occurred while processing category {category}: {e}"
        if use_slurm:
            print(error_msg, flush=True)
        else:
            console.print(f"[red]{error_msg}[/red]")
        return False

def category_processing(category: str, work_dir: str, use_slurm: bool = False, 
                              windowing_strategy: str = "calendar", **kwargs) -> bool:
    """
    Process a single category using dataset_builder.py process-huggingface-only command
    this method submits the subprocess for a single call to dataset_builder module
    
    Args:
        category: Category name to process
        work_dir: Working directory for the dataset builder
        use_slurm: Whether to use SLURM-friendly output
        windowing_strategy: Windowing strategy for processing
        **kwargs: Additional arguments for processing
        
    Returns:
        True if successful, False otherwise
    """
    if use_slurm:
        print(f"Processing data for category: {category}", flush=True)
    else:
        console.print(f"[blue]Processing data for category: {category}[/blue]")

    try:
        # Build command with all arguments
        cmd = [
            sys.executable,
            "/home/damonp/projects/chimera_proj/amazon-data/lib/dataset_builder.py",
            "process-huggingface-only",
        ]
        if use_slurm:
            cmd.append("--slurm")
        cmd.extend([
            "--category", category,
            "--work-dir", work_dir,
            "--windowing-strategy", windowing_strategy
        ])
        # Add optional arguments if provided
        for key, value in kwargs.items():
            if value is not None:
                arg_name = key.replace('_', '-')
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{arg_name}")
                else:
                    cmd.extend([f"--{arg_name}", str(value)])
        
        # Remove empty strings from command
        cmd = [arg for arg in cmd if arg]
        
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        
        if result.returncode == 0:
            if use_slurm:
                print(f"Completed processing for category: {category}", flush=True)
                print("----------------------------------------", flush=True)
            else:
                console.print(f"[green]Completed processing for category: {category}[/green]")
            return True
        else:
            if use_slurm:
                print(f"Error processing category {category}:", flush=True)
                print(f"STDOUT: {result.stdout}", flush=True)
                print(f"STDERR: {result.stderr}", flush=True)
            else:
                console.print(f"[red]Error processing category {category}:[/red]")
                console.print(f"STDOUT: {result.stdout}")
                console.print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        error_msg = f"Exception occurred while processing category {category}: {e}"
        if use_slurm:
            print(error_msg, flush=True)
        else:
            console.print(f"[red]{error_msg}[/red]")
        return False

@app.command()
def process_category(
    array_task_id: int = typer.Argument(..., help="SLURM array task ID (0-indexed)"),
    categories_file: str = typer.Option("/home/damonp/projects/chimera_proj/amazon-data/lib/all_categories.txt", "--categories", "-c", 
                                       help="Path to the text file containing category names"),
    work_dir: str = typer.Option("/pool001/damonp/amazon-data", "--work-dir", "-w",
                                help="Working directory for the dataset builder"),
    use_slurm: bool = typer.Option(True, "--slurm", help="Use SLURM-friendly output"),
    # Legacy options for backward compatibility (not used by default)
    windowing_strategy: str = typer.Option("calendar", "--windowing-strategy", 
                                          help="Windowing strategy: calendar or review_frequency (legacy option)"),
    calendar_window_interval: str = typer.Option("1d", "--calendar-window-interval",
                                                help="Time interval per calendar window (legacy option)"),
    review_window_size: int = typer.Option(10, "--review-window-size",
                                          help="Reviews per window (legacy option)"),
    include_all_reviews: bool = typer.Option(True, "--include-all-reviews",
                                            help="Include all reviews in time window (legacy option)"),
    min_reviews_per_asin: int = typer.Option(50, "--min-reviews-per-asin",
                                            help="Minimum reviews required per ASIN"),
    max_asins: Optional[int] = typer.Option(None, "--max-asins",
                                           help="Maximum ASINs to process (None = all ASINs)"),
    sub_dir: Optional[str] = typer.Option(None, "--sub-dir",
                                         help="Subdirectory name for organizing different configurations (legacy option)"),
    pull_huggingface: bool = typer.Option(False, "--pull-huggingface",
                                         help="Pull HuggingFace data from cloud instead of using local"),
    debug: bool = typer.Option(False, "--debug",
                              help="Enable debug mode with full tracebacks and stop on first error"),

    rolling_window_sizes: List[int] = typer.Option([3, 5, 10, 30], "--rolling-window-sizes",
                                                   help="Rolling window sizes for additional statistics"),
    upsample: bool = typer.Option(True, "--upsample",
                                 help="Create empty buckets for missing time periods (default: True)")
):
    """
    Process a single category based on SLURM array task ID.
    By default, runs both calendar windowing (1w intervals) and review frequency windowing (10 reviews).
    Loads categories from file and processes the category at the specified index.
    this accepts an index, and does calendar and review frequency windowing strategies
    for the category associated with that index in the categories list file
    """
    # Print at the very top to confirm entry
    print("Starting process_category", flush=True)
    # Load categories from file
    categories = load_categories(categories_file)
    
    if not categories:
        if use_slurm:
            print("No categories found in categories file")
        else:
            console.print("[yellow]No categories found in categories file[/yellow]")
        sys.exit(1)
    
    # Validate array task ID
    if array_task_id < 0 or array_task_id >= len(categories):
        error_msg = f"Array task ID {array_task_id} is out of range. Valid range: 0-{len(categories)-1}"
        if use_slurm:
            print(error_msg)
        else:
            console.print(f"[red]{error_msg}[/red]")
        sys.exit(1)
    
    # Get the category for this array task
    category = categories[array_task_id]
    
    if use_slurm:
        print(f"Processing category: {category} (Array task ID: {array_task_id})")
    else:
        console.print(f"[blue]Processing category: {category} (Array task ID: {array_task_id})[/blue]")
    
    # Process both windowing strategies by default
    success_count = 0
    total_strategies = 2
    
    # Strategy 1: Calendar windowing with 1mo intervals
    if use_slurm:
        print(f"Processing calendar windowing strategy for {category}")
    else:
        console.print(f"[blue]Processing calendar windowing strategy for {category}[/blue]")
    
    success1 = category_processing(
        category, work_dir, use_slurm, "calendar",
        calendar_window_interval="1mo",
        min_reviews_per_asin=100,
        sub_dir="calendar_1mo",
        debug=debug,
        upsample=True
    )
    
    if success1:
        success_count += 1
    
    # Strategy 2: Review frequency windowing with 10 reviews
    if use_slurm:
        print(f"Processing review frequency windowing strategy for {category}")
    else:
        console.print(f"[blue]Processing review frequency windowing strategy for {category}[/blue]")
    
    success2 = category_processing(
        category, work_dir, use_slurm, "review_frequency",
        review_window_size=10,
        min_reviews_per_asin=100,
        sub_dir="review_frequency_10",
        debug=debug
    )
    
    if success2:
        success_count += 1
    
    # Report results
    if success_count == total_strategies:
        if use_slurm:
            print(f"Successfully completed both strategies for category: {category}")
        else:
            console.print(f"[green]Successfully completed both strategies for category: {category}[/green]")
        sys.exit(0)
    elif success_count > 0:
        if use_slurm:
            print(f"Partially completed processing for category: {category} ({success_count}/{total_strategies} strategies)")
        else:
            console.print(f"[yellow]Partially completed processing for category: {category} ({success_count}/{total_strategies} strategies)[/yellow]")
        sys.exit(1)
    else:
        if use_slurm:
            print(f"Failed to process category: {category} (0/{total_strategies} strategies)")
        else:
            console.print(f"[red]Failed to process category: {category} (0/{total_strategies} strategies)[/red]")
        sys.exit(1)

@app.command()
def download(
    categories_file: str = typer.Option("/home/damonp/projects/chimera_proj/amazon-data/lib/all_categories.txt", "--categories", "-c", 
                                       help="Path to the text file containing category names"),
    work_dir: str = typer.Option("/pool001/damonp/amazon-data", "--work-dir", "-w",
                                help="Working directory for the dataset builder"),
    use_slurm: bool = typer.Option(True, "--slurm", help="Use SLURM-friendly output")
):
    """
    Download HuggingFace data for all categories listed in the categories file.
    """
    # Load categories from file
    categories = load_categories(categories_file)
    
    if not categories:
        if use_slurm:
            print("No categories found in categories file")
        else:
            console.print("[yellow]No categories found in categories file[/yellow]")
        return
    
    if use_slurm:
        print(f"Found {len(categories)} categories to download")
        print("Starting download process...")
        print("=" * 50)
    else:
        console.print(f"[blue]Found {len(categories)} categories to download[/blue]")
        console.print("[blue]Starting download process...[/blue]")
        console.print("=" * 50)
    
    # Process each category
    successful = 0
    failed = 0
    
    for category in categories:
        if category_download(category, work_dir, use_slurm):
            successful += 1
        else:
            failed += 1
    
    # Summary
    if use_slurm:
        print("=" * 50)
        print(f"Download process completed!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(categories)}")
    else:
        console.print("=" * 50)
        console.print(f"[green]Download process completed![/green]")
        console.print(f"Successful: {successful}")
        console.print(f"Failed: {failed}")
        console.print(f"Total: {len(categories)}")

@app.command()
def process(
    categories_file: str = typer.Option("all_categories.txt", "--categories", "-c", 
                                       help="Path to the text file containing category names"),
    work_dir: str = typer.Option("/home/damonp/projects/chimera_proj/amazon-data", "--work-dir", "-w",
                                help="Working directory for the dataset builder"),
    use_slurm: bool = typer.Option(False, "--slurm", help="Use SLURM-friendly output"),
    windowing_strategy: str = typer.Option("calendar", "--windowing-strategy", 
                                          help="Windowing strategy: calendar or review_frequency"),
    calendar_window_interval: str = typer.Option("1d", "--calendar-window-interval",
                                                help="Time interval per calendar window (e.g., 1d, 2M, 1w)"),
    review_window_size: int = typer.Option(10, "--review-window-size",
                                          help="Reviews per window (for review_frequency strategy)"),
    include_all_reviews: bool = typer.Option(True, "--include-all-reviews",
                                            help="Include all reviews in time window"),
    min_reviews_per_asin: int = typer.Option(10, "--min-reviews-per-asin",
                                            help="Minimum reviews required per ASIN"),
    max_asins: Optional[int] = typer.Option(None, "--max-asins",
                                           help="Maximum ASINs to process (None = all ASINs)"),
    sub_dir: Optional[str] = typer.Option(None, "--sub-dir",
                                         help="Subdirectory name for organizing different configurations"),
    pull_huggingface: bool = typer.Option(False, "--pull-huggingface",
                                         help="Pull HuggingFace data from cloud instead of using local"),
    debug: bool = typer.Option(False, "--debug",
                              help="Enable debug mode with full tracebacks and stop on first error"),

    rolling_window_sizes: List[int] = typer.Option([3, 5, 10, 30], "--rolling-window-sizes",
                                                   help="Rolling window sizes for additional statistics"),
    upsample: bool = typer.Option(False, "--upsample",
                                 help="Create empty buckets for missing time periods")
):
    """
    Process HuggingFace data for all categories listed in the categories file.
    """
    # Load categories from file
    categories = load_categories(categories_file)
    
    if not categories:
        if use_slurm:
            print("No categories found in categories file")
        else:
            console.print("[yellow]No categories found in categories file[/yellow]")
        return
    
    if use_slurm:
        print(f"Found {len(categories)} categories to process")
        print("Starting processing...")
        print("=" * 50)
    else:
        console.print(f"[blue]Found {len(categories)} categories to process[/blue]")
        console.print("[blue]Starting processing...[/blue]")
        console.print("=" * 50)
    
    # Process each category
    successful = 0
    failed = 0
    
    for category in categories:
        if category_processing(
            category, work_dir, use_slurm, windowing_strategy,
            calendar_window_interval=calendar_window_interval,
            review_window_size=review_window_size,
            include_all_reviews=include_all_reviews,
            min_reviews_per_asin=min_reviews_per_asin,
            max_asins=max_asins,
            sub_dir=sub_dir,
            pull_huggingface=pull_huggingface,
            debug=debug,
            rolling_window_sizes=rolling_window_sizes,
            upsample=upsample
        ):
            successful += 1
        else:
            failed += 1
    
    # Summary
    if use_slurm:
        print("=" * 50)
        print(f"Processing completed!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(categories)}")
    else:
        console.print("=" * 50)
        console.print(f"[green]Processing completed![/green]")
        console.print(f"Successful: {successful}")
        console.print(f"Failed: {failed}")
        console.print(f"Total: {len(categories)}")

if __name__ == "__main__":
    app() 