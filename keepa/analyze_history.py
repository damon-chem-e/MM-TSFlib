import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def load_history(file_path: Path) -> dict:
    """Load price history from JSON file."""
    logger.info(f"Loading history from {file_path}")
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
            logger.debug(f"No '{name}' price history found for this product.")
            continue

        history_array = csv_data[index]
        logger.info(f"Found {len(history_array)//2} data points for '{name}' price.")
        
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
    combined_df = combined_df.resample('D').mean().interpolate()
    
    return combined_df

def analyze_product(file_path: Path) -> None:
    """Analyze and plot price history for a product."""
    logger.info(f"Starting analysis of {file_path}")
    
    history = load_history(file_path)
    df = parse_price_history(history)

    if df.empty:
        logger.warning(f"No processable price data found in {file_path.name}")
        return

    print(f"\nAnalyzing {file_path.stem}:")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    for col in df.columns:
        print(f"\n'{col}' Price Statistics:")
        print(df[col].describe().apply(lambda x: f"${x:,.2f}"))

    # Plot price history
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
    
    # Save plot
    plot_path = file_path.parent / f"{file_path.stem}_plot.png"
    logger.info(f"Saving plot to {plot_path}")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def main():
    """Analyze all price history files in the data directory."""
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    logger.info(f"Looking for JSON files in {data_dir}")
    
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist!")
        return
    
    json_files = list(data_dir.glob("*_history.json"))
    logger.info(f"Found {len(json_files)} JSON files: {[f.name for f in json_files]}")
    
    if not json_files:
        logger.error("No JSON files found!")
        return
    
    for file_path in json_files:
        try:
            analyze_product(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)

if __name__ == "__main__":
    main() 