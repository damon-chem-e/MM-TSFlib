import typer
import json
from pathlib import Path
import requests
from typing import List
import time

app = typer.Typer()

def load_api_key(key_path: Path) -> str:
    """Load API key from file."""
    with open(key_path) as f:
        return f.read().strip()

def load_asins(asin_path: Path) -> List[str]:
    """Load ASINs from file."""
    with open(asin_path) as f:
        return [line.strip() for line in f if line.strip()]

def get_price_history(api_key: str, asin: str) -> dict:
    """Get price history for a single ASIN."""
    url = "https://api.keepa.com/product"
    params = {
        "key": api_key,
        "asin": asin,
        "domain": 1,  # US domain
        "history": 1,  # Include price history
        "stats": 1    # Include statistics
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

@app.command()
def main(
    api_key_path: Path = typer.Argument(..., help="Path to file containing Keepa API key"),
    asin_path: Path = typer.Argument(..., help="Path to file containing list of ASINs"),
    output_dir: Path = typer.Option("data", help="Directory to save results")
):
    """
    Pull price history for a list of ASINs from Keepa API.
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load API key and ASINs
    api_key = load_api_key(api_key_path)
    asins = load_asins(asin_path)
    
    # Process each ASIN
    for asin in asins:
        typer.echo(f"Processing ASIN: {asin}")
        try:
            # Get price history
            data = get_price_history(api_key, asin)
            
            # Save to file
            output_file = output_dir / f"{asin}_history.json"
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            
            typer.echo(f"Saved data to {output_file}")
            
            # Rate limiting (100 requests per minute)
            time.sleep(0.6)
            
        except Exception as e:
            typer.echo(f"Error processing {asin}: {e}", err=True)

if __name__ == "__main__":
    app() 