import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Set up logging for errors and console for pretty printing
logger = logging.getLogger(__name__)
console = Console()

def load_history(file_path: Path) -> dict:
    """Load price history from JSON file."""
    with open(file_path) as f:
        return json.load(f)

def parse_price_history(history_data: dict) -> pd.DataFrame:
    """
    Parses historical data from the 'csv' field of the Keepa product object.

    Args:
        history_data: The product data dictionary from Keepa API.

    Returns:
        A pandas DataFrame with a 'date' index and columns for each
        available price type (e.g., 'AMAZON', 'NEW').
    """
    product = history_data['products'][0]
    csv_data = product['csv']

    price_types = {
        'AMAZON': 0,
        'NEW': 1,
        'USED': 2
    }
    
    all_price_data = {}

    for name, index in price_types.items():
        if not (csv_data and len(csv_data) > index and csv_data[index] is not None):
            continue

        history_array = csv_data[index]
        
        type_data = []
        for i in range(0, len(history_array), 2):
            keepa_time = history_array[i]
            price_cents = history_array[i+1]
            
            if price_cents != -1:
                # Convert keepa time to unix timestamp in seconds
                unix_timestamp = (keepa_time + 21564000) * 60
                date = datetime.fromtimestamp(unix_timestamp)
                price_dollars = price_cents / 100.0
                type_data.append({'date': date, name: price_dollars})
        
        if type_data:
            df = pd.DataFrame(type_data).set_index('date')
            all_price_data[name] = df

    if not all_price_data:
        return pd.DataFrame()

    # Combine all dataframes, interpolating to create continuous series for plotting
    combined_df = pd.concat(all_price_data.values(), axis=1)
    combined_df = combined_df.resample('D').mean().interpolate(method='linear')
    
    return combined_df

def analyze_product(file_path: Path, figures_dir: Path) -> None:
    """Analyze and plot price history for a product."""
    history = load_history(file_path)
    df = parse_price_history(history)

    if df.empty:
        console.print(f"[yellow]Warning: No processable price data found in {file_path.name}[/yellow]")
        return

    console.print(Panel(f"Analysis for [bold cyan]{file_path.stem}[/bold cyan]", expand=False, border_style="green"))
    
    # --- Statistics Table ---
    stats_table = Table(title=f"Price Statistics ({df.index.min().date()} to {df.index.max().date()})")
    stats_table.add_column("Statistic", style="magenta")
    
    stats_data = df.describe()
    
    for col in stats_data.columns:
        stats_table.add_column(col, justify="right", style="cyan")

    for index, row in stats_data.iterrows():
        # Capitalize stat name and format values
        stat_name = index.capitalize()
        values = [f"${value:,.2f}" for value in row]
        stats_table.add_row(stat_name, *values)
        
    console.print(stats_table)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for column in df.columns:
        ax.plot(df.index, df[column], label=column)
        
    ax.set_title(f'Price History for {file_path.stem}', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(title='Price Type')
    ax.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()
    
    plot_path = figures_dir / f"{file_path.stem}_plot.png"
    plt.savefig(plot_path)
    plt.close()
    console.print(f"Generated plot: [green]{plot_path}[/green]\n")


def main():
    """Analyze all price history files in the data directory."""
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    console.print("[bold]Keepa Price History Analyzer[/bold]", style="underline2 blue")
    console.print(f"Looking for JSON files in: [cyan]{data_dir}[/cyan]")
    console.print(f"Saving figures to: [cyan]{figures_dir}[/cyan]\n")
    
    if not data_dir.exists():
        console.print(f"[bold red]Error: Data directory not found at {data_dir}[/bold red]")
        return
    
    json_files = list(data_dir.glob("*_history.json"))
    
    if not json_files:
        console.print("[bold red]Error: No JSON history files found![/bold red]")
        return
    
    console.print(f"Found {len(json_files)} files to analyze.")
    for file_path in json_files:
        try:
            analyze_product(file_path, figures_dir)
        except Exception:
            console.print_exception(show_locals=True)

if __name__ == "__main__":
    main() 