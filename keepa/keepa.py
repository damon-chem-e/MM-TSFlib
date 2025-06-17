import typer
import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
import pickle

app = typer.Typer()
console = Console()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("keepa_downloader")

@dataclass
class DownloadConfig:
    api_key: str
    domain: int = 1
    include_history: bool = True
    include_offers: bool = True
    include_buybox: bool = True
    include_rental: bool = True
    include_stats: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 0.6
    batch_size: int = 100
    timeout: int = 30
    save_raw_responses: bool = True
    save_processed_data: bool = True

class KeepaAPIError(Exception):
    pass

class RateLimitExceeded(KeepaAPIError):
    pass

class KeepaDownloader:
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'KeepaDownloader/1.0'
        })
        self.stats = {
            'total_asins': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'api_calls_made': 0,
            'total_tokens_consumed': 0,
            'start_time': None,
            'failed_asins': [],
            'rate_limited_count': 0
        }
    
    def _build_api_params(self, asins: List[str]) -> Dict[str, any]:
        params = {
            'key': self.config.api_key,
            'domain': self.config.domain,
            'asin': ','.join(asins),
        }
        
        if self.config.include_history:
            params['history'] = 1
        if self.config.include_offers:
            params['offers'] = 20
        if self.config.include_buybox:
            params['buybox'] = 1
        if self.config.include_rental:
            params['rental'] = 1
        if self.config.include_stats:
            params['stats'] = 1
            params['range'] = 365
        
        return params
    
    def _make_api_request(self, asins: List[str]) -> Dict:
        url = "https://api.keepa.com/product"
        params = self._build_api_params(asins)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.info(f"Making API request for {len(asins)} ASINs (attempt {attempt + 1}/{self.config.max_retries + 1})")
                
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.timeout
                )
                
                self.stats['api_calls_made'] += 1
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'error' in data:
                        error_msg = data['error'].get('message', 'Unknown API error')
                        logger.error(f"Keepa API error: {error_msg}")
                        raise KeepaAPIError(f"API error: {error_msg}")
                    
                    if 'tokensLeft' in data:
                        logger.info(f"Tokens remaining: {data['tokensLeft']}")
                    
                    if 'tokensConsumed' in data:
                        self.stats['total_tokens_consumed'] += data['tokensConsumed']
                        logger.info(f"Tokens consumed this request: {data['tokensConsumed']}")
                    
                    return data
                
                elif response.status_code == 429:
                    self.stats['rate_limited_count'] += 1
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    raise RateLimitExceeded("Rate limit exceeded")
                
                elif response.status_code == 401:
                    raise KeepaAPIError("Invalid API key")
                
                elif response.status_code == 402:
                    raise KeepaAPIError("Insufficient tokens")
                
                else:
                    response.raise_for_status()
                    
            except (requests.RequestException, RateLimitExceeded) as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed after {self.config.max_retries + 1} attempts")
                    raise KeepaAPIError(f"Request failed after retries: {e}")
        
        raise KeepaAPIError("Unexpected error in API request")
    
    def _save_data(self, data: Dict, output_dir: Path, asins: List[str], batch_num: int) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.save_raw_responses:
            raw_file = output_dir / "raw" / f"batch_{batch_num:04d}_{timestamp}.json"
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved raw data to {raw_file}")
        
        if self.config.save_processed_data and 'products' in data:
            products_dir = output_dir / "products"
            products_dir.mkdir(parents=True, exist_ok=True)
            
            for product in data['products']:
                if product and 'asin' in product:
                    asin = product['asin']
                    product_file = products_dir / f"{asin}_complete.json"
                    with open(product_file, 'w') as f:
                        json.dump(product, f, indent=2)
                    logger.debug(f"Saved product data for {asin}")
    
    def _validate_asins(self, asins: List[str]) -> List[str]:
        valid_asins = []
        for asin in asins:
            asin = asin.strip().upper()
            if len(asin) == 10 and asin.startswith('B'):
                valid_asins.append(asin)
            else:
                logger.warning(f"Invalid ASIN format: {asin}")
        
        return valid_asins
    
    def _create_batches(self, asins: List[str]) -> List[List[str]]:
        batches = []
        for i in range(0, len(asins), self.config.batch_size):
            batch = asins[i:i + self.config.batch_size]
            batches.append(batch)
        return batches
    
    def download_complete_dataset(
        self, 
        asins: List[str], 
        output_dir: Path,
        resume_from_batch: int = 0
    ) -> Dict:
        
        self.stats['start_time'] = datetime.now()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        valid_asins = self._validate_asins(asins)
        invalid_count = len(asins) - len(valid_asins)
        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid ASINs")
        
        self.stats['total_asins'] = len(valid_asins)
        
        if not valid_asins:
            logger.error("No valid ASINs provided")
            return self.stats
        
        batches = self._create_batches(valid_asins)
        logger.info(f"Created {len(batches)} batches of up to {self.config.batch_size} ASINs each")
        
        if resume_from_batch > 0:
            logger.info(f"Resuming from batch {resume_from_batch}")
            batches = batches[resume_from_batch:]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            download_task = progress.add_task(
                f"Downloading {len(valid_asins)} products", 
                total=len(batches)
            )
            
            for batch_idx, batch_asins in enumerate(batches):
                actual_batch_num = batch_idx + resume_from_batch
                
                try:
                    logger.info(f"Processing batch {actual_batch_num + 1}/{len(batches) + resume_from_batch}")
                    
                    response_data = self._make_api_request(batch_asins)
                    
                    self._save_data(response_data, output_dir, batch_asins, actual_batch_num)
                    
                    if 'products' in response_data:
                        successful_products = sum(1 for p in response_data['products'] if p is not None)
                        self.stats['successful_downloads'] += successful_products
                        self.stats['failed_downloads'] += len(batch_asins) - successful_products
                        
                        for i, product in enumerate(response_data['products']):
                            if product is None:
                                self.stats['failed_asins'].append(batch_asins[i])
                    
                    time.sleep(self.config.rate_limit_delay)
                    
                    progress.update(download_task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {actual_batch_num + 1}: {e}")
                    self.stats['failed_downloads'] += len(batch_asins)
                    self.stats['failed_asins'].extend(batch_asins)
                    
                    progress_file = output_dir / "download_progress.json"
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'last_completed_batch': actual_batch_num,
                            'total_batches': len(batches) + resume_from_batch,
                            'stats': self.stats,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
        
        self._save_final_report(output_dir)
        return self.stats
    
    def _save_final_report(self, output_dir: Path) -> None:
        end_time = datetime.now()
        duration = end_time - self.stats['start_time']
        
        report = {
            'download_summary': {
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'total_asins_requested': self.stats['total_asins'],
                'successful_downloads': self.stats['successful_downloads'],
                'failed_downloads': self.stats['failed_downloads'],
                'success_rate': self.stats['successful_downloads'] / max(self.stats['total_asins'], 1) * 100,
                'api_calls_made': self.stats['api_calls_made'],
                'total_tokens_consumed': self.stats['total_tokens_consumed'],
                'rate_limited_count': self.stats['rate_limited_count'],
                'failed_asins': self.stats['failed_asins']
            },
            'configuration': {
                'domain': self.config.domain,
                'batch_size': self.config.batch_size,
                'include_history': self.config.include_history,
                'include_offers': self.config.include_offers,
                'include_buybox': self.config.include_buybox,
                'include_rental': self.config.include_rental,
                'include_stats': self.config.include_stats,
                'max_retries': self.config.max_retries,
                'rate_limit_delay': self.config.rate_limit_delay
            }
        }
        
        report_file = output_dir / f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._display_summary_table(report['download_summary'])
        
        logger.info(f"Complete download report saved to {report_file}")
    
    def _display_summary_table(self, summary: Dict) -> None:
        table = Table(title="Download Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total ASINs Requested", str(summary['total_asins_requested']))
        table.add_row("Successful Downloads", str(summary['successful_downloads']))
        table.add_row("Failed Downloads", str(summary['failed_downloads']))
        table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
        table.add_row("API Calls Made", str(summary['api_calls_made']))
        table.add_row("Tokens Consumed", str(summary['total_tokens_consumed']))
        table.add_row("Duration", f"{summary['duration_seconds']:.1f} seconds")
        table.add_row("Rate Limited", str(summary['rate_limited_count']))
        
        console.print(table)


def load_api_key(key_path: Path) -> str:
    try:
        with open(key_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        raise typer.BadParameter(f"API key file not found: {key_path}")

def load_asins(asin_path: Path) -> List[str]:
    try:
        with open(asin_path) as f:
            asins = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if ',' in line:
                        asins.extend([asin.strip() for asin in line.split(',')])
                    else:
                        asins.append(line)
            return asins
    except FileNotFoundError:
        raise typer.BadParameter(f"ASIN file not found: {asin_path}")

@app.command()
def download(
    api_key_path: Path = typer.Argument(..., help="Path to file containing Keepa API key"),
    asin_path: Path = typer.Argument(..., help="Path to file containing list of ASINs"),
    output_dir: Path = typer.Option("keepa/data", help="Directory to save results"),
    domain: int = typer.Option(1, help="Amazon domain (1=US, 3=UK, 4=DE, etc.)"),
    batch_size: int = typer.Option(100, help="Number of ASINs per API request (max 100)"),
    max_retries: int = typer.Option(3, help="Maximum number of retries per request"),
    rate_limit_delay: float = typer.Option(0.6, help="Delay between requests in seconds"),
    resume_from_batch: int = typer.Option(0, help="Resume from specific batch number"),
    disable_offers: bool = typer.Option(False, help="Disable offer data collection"),
    disable_buybox: bool = typer.Option(False, help="Disable buybox data collection"),
    disable_rental: bool = typer.Option(False, help="Disable rental data collection"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging")
):
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(Panel.fit(
        "[bold blue]Keepa Comprehensive Data Downloader[/bold blue]\n"
        "Downloading complete dataset with maximum available data",
        border_style="blue"
    ))
    
    api_key = load_api_key(api_key_path)
    asins = load_asins(asin_path)
    
    console.print(f"Loaded [green]{len(asins)}[/green] ASINs from {asin_path}")
    
    if batch_size > 100:
        console.print("[yellow]Warning: Keepa API supports maximum 100 ASINs per request. Setting batch_size to 100.[/yellow]")
        batch_size = 100
    
    config = DownloadConfig(
        api_key=api_key,
        domain=domain,
        batch_size=batch_size,
        max_retries=max_retries,
        rate_limit_delay=rate_limit_delay,
        include_offers=not disable_offers,
        include_buybox=not disable_buybox,
        include_rental=not disable_rental
    )
    
    downloader = KeepaDownloader(config)
    
    try:
        stats = downloader.download_complete_dataset(
            asins=asins,
            output_dir=output_dir,
            resume_from_batch=resume_from_batch
        )
        
        if stats['failed_downloads'] == 0:
            console.print("[bold green]✓ All downloads completed successfully![/bold green]")
        else:
            console.print(f"[yellow]⚠ Download completed with {stats['failed_downloads']} failures[/yellow]")
            
    except KeyboardInterrupt:
        console.print("\n[red]Download interrupted by user[/red]")
        console.print("Use --resume-from-batch to continue from where you left off")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def validate_asins(
    asin_path: Path = typer.Argument(..., help="Path to file containing ASINs to validate")
):
    asins = load_asins(asin_path)
    
    valid_asins = []
    invalid_asins = []
    
    for asin in asins:
        asin = asin.strip().upper()
        if len(asin) == 10 and asin.startswith('B'):
            valid_asins.append(asin)
        else:
            invalid_asins.append(asin)
    
    console.print(f"[green]Valid ASINs: {len(valid_asins)}[/green]")
    console.print(f"[red]Invalid ASINs: {len(invalid_asins)}[/red]")
    
    if invalid_asins:
        console.print("\nInvalid ASINs found:")
        for asin in invalid_asins:
            console.print(f"  [red]{asin}[/red]")

if __name__ == "__main__":
    app()