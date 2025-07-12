import typer
import json
import requests
import time
import random
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
import sys
from contextlib import contextmanager

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class PlainTextProgress:
    """Simple progress tracking for non-rich output"""
    def __init__(self, total_batches: int, total_asins: int):
        self.total_batches = total_batches
        self.total_asins = total_asins
        self.completed_batches = 0
        self.completed_asins = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time
    
    def update(self, batch_asins: int, successful_asins: int = None):
        """Update progress and print status"""
        self.completed_batches += 1
        self.completed_asins += successful_asins if successful_asins is not None else batch_asins
        
        current_time = datetime.now()
        elapsed = current_time - self.start_time
        
        # Print update every 30 seconds or when batch completes
        if (current_time - self.last_update).total_seconds() >= 30 or self.completed_batches % 5 == 0:
            batch_progress = (self.completed_batches / self.total_batches) * 100
            asin_progress = (self.completed_asins / self.total_asins) * 100
            
            # Estimate remaining time
            if self.completed_batches > 0:
                avg_time_per_batch = elapsed.total_seconds() / self.completed_batches
                remaining_batches = self.total_batches - self.completed_batches
                eta_seconds = remaining_batches * avg_time_per_batch
                eta_str = f"ETA: {eta_seconds/60:.1f}min"
            else:
                eta_str = "ETA: calculating..."
            
            status_msg = (
                f"[{current_time.strftime('%H:%M:%S')}] "
                f"Batch {self.completed_batches}/{self.total_batches} "
                f"({batch_progress:.1f}%) | "
                f"ASINs: {self.completed_asins}/{self.total_asins} "
                f"({asin_progress:.1f}%) | "
                f"Elapsed: {elapsed.total_seconds()/60:.1f}min | "
                f"{eta_str}"
            )
            
            print(status_msg, flush=True)
            self.last_update = current_time
    
    def log_message(self, message: str):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}", flush=True)

app = typer.Typer()
console = Console()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("keepa_downloader")

@contextmanager
def output_mode(use_rich: bool = True):
    """Context manager for switching between rich and plain text output"""
    if use_rich:
        # Use rich output
        yield console
    else:
        # Use plain text output
        class PlainTextConsole:
            def print(self, *args, **kwargs):
                # Extract text from rich formatting
                text = " ".join(str(arg) for arg in args)
                # Remove rich formatting codes
                import re
                text = re.sub(r'\[[^\]]*\]', '', text)
                print(text, flush=True)
            
            def log(self, message: str):
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] {message}", flush=True)
        
        yield PlainTextConsole()

@dataclass
class DownloadConfig:
    api_key: str
    domain: int = 1
    include_history: bool = True
    include_offers: bool = True
    include_buybox: bool = True
    include_rental: bool = True
    include_stats: bool = True
    max_retries: int = 10  # Increased for better resilience
    retry_delay: float = 2.0
    rate_limit_delay: float = 0.6
    batch_size: int = 100
    timeout: int = 30
    save_raw_responses: bool = True
    save_processed_data: bool = True
    max_rate_limit_wait: int = 3600  # Maximum wait time for rate limits (1 hour)
    continue_on_batch_failure: bool = True  # Continue processing other batches if one fails
    tokens_per_minute: int = 20  # Rate limit tokens per minute for better backoff calculation

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
                logger.debug(f"Making API request for {len(asins)} ASINs (attempt {attempt + 1}/{self.config.max_retries + 1})")
                
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
                        logger.debug(f"Tokens remaining: {data['tokensLeft']}")
                    
                    if 'tokensConsumed' in data:
                        self.stats['total_tokens_consumed'] += data['tokensConsumed']
                        logger.debug(f"Tokens consumed this request: {data['tokensConsumed']}")
                    
                    return data
                
                elif response.status_code == 429:
                    self.stats['rate_limited_count'] += 1
                    retry_after = int(response.headers.get('Retry-After', 60))
                    
                    # Calculate better retry time based on tokens per minute
                    # If we're rate limited, we need to wait for tokens to replenish
                    tokens_needed = len(asins) * 2  # Rough estimate of tokens needed for this batch
                    estimated_wait = max(retry_after, (tokens_needed / self.config.tokens_per_minute) * 60)
                    
                    # Cap the retry wait time to prevent excessive delays
                    retry_after = min(int(estimated_wait), self.config.max_rate_limit_wait)
                    
                    logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds (estimated {tokens_needed} tokens needed)...")
                    console.print(f"[yellow]Rate limited - waiting {retry_after} seconds (need ~{tokens_needed} tokens)...[/yellow]")
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
                    # Exponential backoff with jitter to prevent thundering herd
                    base_delay = self.config.retry_delay * (2 ** attempt)
                    jitter = random.random() * 0.1 * base_delay
                    delay = base_delay + jitter
                    delay = min(delay, 300)  # Cap at 5 minutes
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    console.print(f"[dim]Retrying in {delay:.1f} seconds...[/dim]")
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
                json.dump(data, f, indent=2, cls=DateTimeEncoder)
            logger.debug(f"Saved raw data to {raw_file}")
        
        if self.config.save_processed_data and 'products' in data:
            products_dir = output_dir / "products"
            products_dir.mkdir(parents=True, exist_ok=True)
            
            for product in data['products']:
                if product and 'asin' in product:
                    asin = product['asin']
                    product_file = products_dir / f"{asin}_complete.json"
                    with open(product_file, 'w') as f:
                        json.dump(product, f, indent=2, cls=DateTimeEncoder)
                    logger.debug(f"Saved product data for {asin}")
    
    def _save_failed_batch(self, asins: List[str], batch_num: int, output_dir: Path, error: str) -> None:
        """Save failed batch ASINs for later retry"""
        failed_batches_dir = output_dir / "failed_batches"
        failed_batches_dir.mkdir(parents=True, exist_ok=True)
        
        failed_batch_file = failed_batches_dir / f"failed_batch_{batch_num:04d}.txt"
        with open(failed_batch_file, 'w') as f:
            f.write(f"# Failed batch {batch_num}\n")
            f.write(f"# Error: {error}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write("\n")
            for asin in asins:
                f.write(f"{asin}\n")
        
        logger.info(f"Saved failed batch {batch_num} to {failed_batch_file}")
    
    def _calculate_optimal_delay(self, tokens_consumed: int = None) -> float:
        """Calculate optimal delay between requests based on tokens per minute"""
        if tokens_consumed is None:
            # Default delay based on batch size and tokens per minute
            tokens_per_batch = self.config.batch_size * 2  # Rough estimate
            delay = (tokens_per_batch / self.config.tokens_per_minute) * 60
        else:
            # Calculate delay based on actual tokens consumed
            delay = (tokens_consumed / self.config.tokens_per_minute) * 60
        
        # Add some buffer to be safe
        delay = delay * 1.1
        
        # Ensure minimum delay
        delay = max(delay, self.config.rate_limit_delay)
        
        return delay
    
    def _validate_asins(self, asins: List[str], asin_source_path: Path = None) -> List[str]:
        valid_asins = []
        invalid_asins = []
        
        for asin in asins:
            asin = asin.strip().upper()
            if len(asin) == 10 and asin.startswith('B'):
                valid_asins.append(asin)
            else:
                invalid_asins.append(asin)
        
        # Save invalid ASINs to file if source path provided
        if invalid_asins and asin_source_path:
            invalid_file = asin_source_path.parent / f"{asin_source_path.stem}_invalid_asins.txt"
            with open(invalid_file, 'w') as f:
                for asin in invalid_asins:
                    f.write(f"{asin}\n")
            console.print(f"[yellow]Found {len(invalid_asins)} invalid ASINs - saved to {invalid_file}[/yellow]")
            
            # Show only first 5 invalid ASINs as examples
            if len(invalid_asins) > 5:
                examples = invalid_asins[:5]
                console.print(f"[dim]Examples: {', '.join(examples)}...[/dim]")
            else:
                console.print(f"[dim]Invalid ASINs: {', '.join(invalid_asins)}[/dim]")
        
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
        asin_source_path: Path = None,
        resume_from_batch: int = 0,
        use_rich_output: bool = True
    ) -> Dict:
        
        self.stats['start_time'] = datetime.now()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        valid_asins = self._validate_asins(asins, asin_source_path)
        self.stats['total_asins'] = len(valid_asins)
        
        if not valid_asins:
            logger.error("No valid ASINs provided")
            return self.stats
        
        batches = self._create_batches(valid_asins)
        
        with output_mode(use_rich_output) as output:
            output.print(f"[green]Created {len(batches)} batches of up to {self.config.batch_size} ASINs each[/green]")
            
            if resume_from_batch > 0:
                output.print(f"[blue]Resuming from batch {resume_from_batch}[/blue]")
                batches = batches[resume_from_batch:]
        
        # Initialize progress tracking
        if use_rich_output:
            # Rich progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=False
            ) as progress:
                download_task = progress.add_task(
                    f"Downloading {len(valid_asins)} products", 
                    total=len(batches)
                )
                
                for batch_idx, batch_asins in enumerate(batches):
                    actual_batch_num = batch_idx + resume_from_batch
                    
                    try:
                        # Update progress description with current batch info
                        progress.update(
                            download_task, 
                            description=f"Downloading {len(valid_asins)} products (batch {actual_batch_num + 1}/{len(batches) + resume_from_batch})"
                        )
                        
                        response_data = self._make_api_request(batch_asins)
                        
                        self._save_data(response_data, output_dir, batch_asins, actual_batch_num)
                        
                        if 'products' in response_data:
                            successful_products = sum(1 for p in response_data['products'] if p is not None)
                            self.stats['successful_downloads'] += successful_products
                            self.stats['failed_downloads'] += len(batch_asins) - successful_products
                            
                            for i, product in enumerate(response_data['products']):
                                if product is None:
                                    self.stats['failed_asins'].append(batch_asins[i])
                        
                        # Calculate optimal delay based on tokens consumed
                        tokens_consumed = response_data.get('tokensConsumed', None)
                        optimal_delay = self._calculate_optimal_delay(tokens_consumed)
                        
                        if tokens_consumed:
                            logger.debug(f"Using optimal delay of {optimal_delay:.1f}s based on {tokens_consumed} tokens consumed")
                        
                        time.sleep(optimal_delay)
                        
                        progress.update(download_task, advance=1)
                        
                        # Show periodic status updates every 10 batches
                        if (actual_batch_num + 1) % 10 == 0:
                            output.print(f"[dim]Completed {actual_batch_num + 1}/{len(batches) + resume_from_batch} batches[/dim]")
                        
                    except Exception as e:
                        logger.error(f"Failed to process batch {actual_batch_num + 1}: {e}")
                        output.print(f"[red]Failed to process batch {actual_batch_num + 1}: {e}[/red]")
                        self.stats['failed_downloads'] += len(batch_asins)
                        self.stats['failed_asins'].extend(batch_asins)
                        
                        # Save failed batch for later retry
                        self._save_failed_batch(batch_asins, actual_batch_num, output_dir, str(e))
                        
                        # Save progress after each batch failure
                        progress_file = output_dir / "download_progress.json"
                        with open(progress_file, 'w') as f:
                            json.dump({
                                'last_completed_batch': actual_batch_num,
                                'total_batches': len(batches) + resume_from_batch,
                                'stats': self.stats,
                                'timestamp': datetime.now().isoformat()
                            }, f, indent=2, cls=DateTimeEncoder)
                        
                        # Continue with next batch instead of stopping
                        if self.config.continue_on_batch_failure:
                            output.print(f"[yellow]Continuing with next batch...[/yellow]")
                            progress.update(download_task, advance=1)
                            continue
                        else:
                            raise e  # Re-raise if we don't want to continue
        else:
            # Plain text progress tracking
            progress = PlainTextProgress(len(batches), len(valid_asins))
            progress.log_message(f"Starting download of {len(valid_asins)} ASINs in {len(batches)} batches")
            
            for batch_idx, batch_asins in enumerate(batches):
                actual_batch_num = batch_idx + resume_from_batch
                
                try:
                    progress.log_message(f"Processing batch {actual_batch_num + 1}/{len(batches) + resume_from_batch} ({len(batch_asins)} ASINs)")
                    
                    response_data = self._make_api_request(batch_asins)
                    
                    self._save_data(response_data, output_dir, batch_asins, actual_batch_num)
                    
                    if 'products' in response_data:
                        successful_products = sum(1 for p in response_data['products'] if p is not None)
                        self.stats['successful_downloads'] += successful_products
                        self.stats['failed_downloads'] += len(batch_asins) - successful_products
                        
                        for i, product in enumerate(response_data['products']):
                            if product is None:
                                self.stats['failed_asins'].append(batch_asins[i])
                        
                        progress.log_message(f"Batch {actual_batch_num + 1}: {successful_products}/{len(batch_asins)} successful")
                    
                    # Calculate optimal delay based on tokens consumed
                    tokens_consumed = response_data.get('tokensConsumed', None)
                    optimal_delay = self._calculate_optimal_delay(tokens_consumed)
                    
                    if tokens_consumed:
                        progress.log_message(f"Tokens consumed: {tokens_consumed}, delay: {optimal_delay:.1f}s")
                    
                    time.sleep(optimal_delay)
                    
                    progress.update(len(batch_asins), successful_products if 'products' in response_data else len(batch_asins))
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {actual_batch_num + 1}: {e}")
                    progress.log_message(f"ERROR: Failed to process batch {actual_batch_num + 1}: {e}")
                    self.stats['failed_downloads'] += len(batch_asins)
                    self.stats['failed_asins'].extend(batch_asins)
                    
                    # Save failed batch for later retry
                    self._save_failed_batch(batch_asins, actual_batch_num, output_dir, str(e))
                    
                    # Save progress after each batch failure
                    progress_file = output_dir / "download_progress.json"
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'last_completed_batch': actual_batch_num,
                            'total_batches': len(batches) + resume_from_batch,
                            'stats': self.stats,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2, cls=DateTimeEncoder)
                    
                    # Continue with next batch instead of stopping
                    if self.config.continue_on_batch_failure:
                        progress.log_message("Continuing with next batch...")
                        progress.update(len(batch_asins), 0)
                        continue
                    else:
                        raise e  # Re-raise if we don't want to continue
        
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
                'rate_limit_delay': self.config.rate_limit_delay,
                'tokens_per_minute': self.config.tokens_per_minute
            }
        }
        
        report_file = output_dir / f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, cls=DateTimeEncoder)
        
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
    max_retries: int = typer.Option(10, help="Maximum number of retries per request"),
    rate_limit_delay: float = typer.Option(0.6, help="Delay between requests in seconds"),
    tokens_per_minute: int = typer.Option(20, help="Rate limit tokens per minute for better backoff calculation"),
    resume_from_batch: int = typer.Option(0, help="Resume from specific batch number"),
    disable_offers: bool = typer.Option(False, help="Disable offer data collection"),
    disable_buybox: bool = typer.Option(False, help="Disable buybox data collection"),
    disable_rental: bool = typer.Option(False, help="Disable rental data collection"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    plain_output: bool = typer.Option(False, "--plain", help="Use plain text output (useful for log files)")
):
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    with output_mode(not plain_output) as output:
        if not plain_output:
            output.print(Panel.fit(
                "[bold blue]Keepa Comprehensive Data Downloader[/bold blue]\n"
                "Downloading complete dataset with maximum available data",
                border_style="blue"
            ))
        else:
            output.log("Keepa Comprehensive Data Downloader - Starting download")
        
        api_key = load_api_key(api_key_path)
        asins = load_asins(asin_path)
        
        output.print(f"Loaded [green]{len(asins)}[/green] ASINs from {asin_path}")
        
        if batch_size > 100:
            output.print("[yellow]Warning: Keepa API supports maximum 100 ASINs per request. Setting batch_size to 100.[/yellow]")
            batch_size = 100
    
    config = DownloadConfig(
        api_key=api_key,
        domain=domain,
        batch_size=batch_size,
        max_retries=max_retries,
        rate_limit_delay=rate_limit_delay,
        tokens_per_minute=tokens_per_minute,
        include_offers=not disable_offers,
        include_buybox=not disable_buybox,
        include_rental=not disable_rental,
        continue_on_batch_failure=True  # Always continue on batch failure for resilience
    )
    
    downloader = KeepaDownloader(config)
    
    try:
        stats = downloader.download_complete_dataset(
            asins=asins,
            output_dir=output_dir,
            asin_source_path=asin_path,
            resume_from_batch=resume_from_batch,
            use_rich_output=not plain_output
        )
        
        with output_mode(not plain_output) as output:
            if stats['failed_downloads'] == 0:
                output.print("[bold green]✓ All downloads completed successfully![/bold green]")
            else:
                output.print(f"[yellow]⚠ Download completed with {stats['failed_downloads']} failures[/yellow]")
                
    except KeyboardInterrupt:
        with output_mode(not plain_output) as output:
            output.print("\n[red]Download interrupted by user[/red]")
            output.print("Use --resume-from-batch to continue from where you left off")
    except Exception as e:
        with output_mode(not plain_output) as output:
            output.print(f"[red]Download failed: {e}[/red]")
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
    
    # Save invalid ASINs to file
    if invalid_asins:
        invalid_file = asin_path.parent / f"{asin_path.stem}_invalid_asins.txt"
        with open(invalid_file, 'w') as f:
            for asin in invalid_asins:
                f.write(f"{asin}\n")
        console.print(f"[yellow]Invalid ASINs saved to: {invalid_file}[/yellow]")
        
        # Show only first 5 invalid ASINs as examples
        if len(invalid_asins) > 5:
            examples = invalid_asins[:5]
            console.print(f"[dim]Examples: {', '.join(examples)}...[/dim]")
        else:
            console.print(f"[dim]Invalid ASINs: {', '.join(invalid_asins)}[/dim]")

@app.command()
def retry_failed_batches(
    api_key_path: Path = typer.Argument(..., help="Path to file containing Keepa API key"),
    output_dir: Path = typer.Argument(..., help="Directory containing failed batches"),
    domain: int = typer.Option(1, help="Amazon domain (1=US, 3=UK, 4=DE, etc.)"),
    batch_size: int = typer.Option(100, help="Number of ASINs per API request (max 100)"),
    max_retries: int = typer.Option(10, help="Maximum number of retries per request"),
    rate_limit_delay: float = typer.Option(0.6, help="Delay between requests in seconds"),
    tokens_per_minute: int = typer.Option(20, help="Rate limit tokens per minute for better backoff calculation"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    plain_output: bool = typer.Option(False, "--plain", help="Use plain text output (useful for log files)")
):
    """Retry failed batches from a previous download"""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    with output_mode(not plain_output) as output:
        if not plain_output:
            output.print(Panel.fit(
                "[bold blue]Keepa Failed Batch Retry[/bold blue]\n"
                "Retrying failed batches from previous download",
                border_style="blue"
            ))
        else:
            output.log("Keepa Failed Batch Retry - Starting retry")
        
        api_key = load_api_key(api_key_path)
        failed_batches_dir = output_dir / "failed_batches"
        
        if not failed_batches_dir.exists():
            output.print(f"[red]No failed batches directory found at {failed_batches_dir}[/red]")
            raise typer.Exit(1)
        
        # Collect all failed batch ASINs
        all_failed_asins = []
        failed_batch_files = list(failed_batches_dir.glob("failed_batch_*.txt"))
        
        if not failed_batch_files:
            output.print(f"[yellow]No failed batch files found in {failed_batches_dir}[/yellow]")
            return
        
        for batch_file in failed_batch_files:
            with open(batch_file) as f:
                asins = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        asins.append(line)
                all_failed_asins.extend(asins)
        
        output.print(f"[yellow]Found {len(all_failed_asins)} ASINs in {len(failed_batch_files)} failed batches[/yellow]")
    
    config = DownloadConfig(
        api_key=api_key,
        domain=domain,
        batch_size=batch_size,
        max_retries=max_retries,
        rate_limit_delay=rate_limit_delay,
        tokens_per_minute=tokens_per_minute
    )
    
    downloader = KeepaDownloader(config)
    
    try:
        stats = downloader.download_complete_dataset(
            asins=all_failed_asins,
            output_dir=output_dir,
            resume_from_batch=0,
            use_rich_output=not plain_output
        )
        
        with output_mode(not plain_output) as output:
            if stats['failed_downloads'] == 0:
                output.print("[bold green]✓ All retry downloads completed successfully![/bold green]")
            else:
                output.print(f"[yellow]⚠ Retry completed with {stats['failed_downloads']} failures[/yellow]")
                
    except KeyboardInterrupt:
        with output_mode(not plain_output) as output:
            output.print("\n[red]Retry interrupted by user[/red]")
    except Exception as e:
        with output_mode(not plain_output) as output:
            output.print(f"[red]Retry failed: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()