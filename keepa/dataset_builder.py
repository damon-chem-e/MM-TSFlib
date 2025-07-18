#!/usr/bin/env python3

import subprocess
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging
from datasets import load_dataset
import sys
import os
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler

current_dir = Path(__file__).parent
if (current_dir / "keepa_test.py").exists():
    sys.path.append(str(current_dir))
elif (current_dir / "keepa" / "keepa_test.py").exists():
    sys.path.append(str(current_dir / "keepa"))
else:
    sys.path.append(str(current_dir))

# Import analyze_history only when needed for Keepa-related commands
_analyze_history_imported = False

def _import_analyze_history():
    """Import analyze_history module when needed for Keepa data processing."""
    global _analyze_history_imported
    if not _analyze_history_imported:
        try:
            from analyze_history import parse_price_history, load_history
            _analyze_history_imported = True
            return parse_price_history, load_history
        except ImportError:
            print("Error: Cannot import analyze_history. Make sure analyze_history.py is in the same directory.")
            sys.exit(1)
    else:
        # Return the already imported functions
        from analyze_history import parse_price_history, load_history
        return parse_price_history, load_history

# Set up logging - will be configured based on SLURM mode
logger = logging.getLogger(__name__)
console = Console()

def setup_logging(use_slurm: bool = False):
    """Configure logging based on SLURM mode."""
    if use_slurm:
        # Plain logging for SLURM
        logging.basicConfig(
            level="INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler()]
        )
    else:
        # Rich logging for interactive use
        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )

app = typer.Typer()

def flush_logger(slurm: bool = False):
    if slurm:
        sys.stdout.flush()

class SlurmOutput:
    """
    Context manager for SLURM-friendly or rich output.
    Use .print() for messages and .progress() for progress updates.
    """
    def __init__(self, use_slurm: bool):
        self.use_slurm = use_slurm
        self.console = console if not use_slurm else None
        self.progress_ctx = None
        self.progress_task = None
        self.progress_total = None
        self.progress_current = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_ctx:
            self.progress_ctx.__exit__(exc_type, exc_val, exc_tb)

    def print(self, *args, **kwargs):
        if self.use_slurm:
            # Strip rich formatting for SLURM mode
            text = " ".join(str(arg) for arg in args)
            # Remove rich formatting tags like [bold blue], [green], etc.
            import re
            text = re.sub(r'\[[^\]]*\]', '', text)
            print(text, flush=True, **kwargs)
        else:
            self.console.print(*args, **kwargs)

    def panel(self, *args, **kwargs):
        if self.use_slurm:
            # Just print the text content without rich formatting
            text = args[0] if args else ""
            if hasattr(text, 'renderable'):
                text = text.renderable
            # Strip rich formatting
            import re
            text = re.sub(r'\[[^\]]*\]', '', str(text))
            self.print(text)
        else:
            self.console.print(Panel(*args, **kwargs))

    def start_progress(self, description, total):
        self.progress_total = total
        self.progress_current = 0
        if self.use_slurm:
            self.print(f"{description} (0/{total})")
        else:
            self.progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )
            self.progress_ctx.__enter__()
            self.progress_task = self.progress_ctx.add_task(description, total=total)

    def update_progress(self, description=None):
        self.progress_current += 1
        if self.use_slurm:
            msg = f"{description} ({self.progress_current}/{self.progress_total})" if description else f"Progress: {self.progress_current}/{self.progress_total}"
            self.print(msg)
        else:
            if description:
                self.progress_ctx.update(self.progress_task, description=description)
            self.progress_ctx.advance(self.progress_task)

    def stop_progress(self):
        if self.progress_ctx:
            self.progress_ctx.__exit__(None, None, None)
            self.progress_ctx = None
            self.progress_task = None

class AmazonKeepaDataPipeline:
    
    def __init__(self, work_dir: Path = Path(".")):
        self.work_dir = work_dir
        
        # Check if we're already in keepa directory or if keepa is a subdirectory
        if (work_dir / "keepa_test.py").exists():
            self.keepa_dir = work_dir  # We're already in keepa directory
        elif (work_dir / "keepa" / "keepa_test.py").exists():
            self.keepa_dir = work_dir / "keepa"  # keepa is a subdirectory
        else:
            self.keepa_dir = work_dir  # Default to current directory
            
        self.data_dir = self.keepa_dir / "data"
        self.huggingface_dir = work_dir / "huggingface_data"
        self.processed_datasets_dir = work_dir / "processed_datasets"
        
        # Create directories as needed
        self.huggingface_dir.mkdir(parents=True, exist_ok=True)
        self.processed_datasets_dir.mkdir(parents=True, exist_ok=True)
        # Note: data_dir is only created when actually needed for Keepa operations
    
    def download_huggingface_data(self, category: str, min_reviews: int = 10, 
                                 max_asins: Optional[int] = None, sample_reviews: Optional[int] = None) -> Dict[str, Path]:
        """Download HuggingFace data, save locally, and extract ASINs in one operation."""
        logger.info(f"Downloading HuggingFace data for category: {category}")
        
        # Create category-specific directory
        category_dir = self.huggingface_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Set custom cache directory to avoid disk quota issues
            cache_dir = self.work_dir / ".hf_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Download reviews data with trust_remote_code=True to fix warning
            logger.info("Downloading reviews dataset...")
            logger.info(f"Using HuggingFace cache directory: {cache_dir}")
            reviews_dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_review_{category}",
                trust_remote_code=True,
                cache_dir=str(cache_dir)
            )
            reviews_df = reviews_dataset["full"].to_pandas()
            
            if sample_reviews:
                logger.info(f"Sampling {sample_reviews} reviews for testing")
                reviews_df = reviews_df.sample(n=min(sample_reviews, len(reviews_df)), random_state=42)
            
            reviews_df['datetime'] = pd.to_datetime(reviews_df['timestamp'], unit='ms')
            
            # Save reviews data
            reviews_path = category_dir / "reviews.parquet"
            reviews_df.to_parquet(reviews_path, index=False)
            logger.info(f"Saved {len(reviews_df)} reviews to {reviews_path}")
            
            # Extract ASINs from the downloaded data
            logger.info("Extracting ASINs from downloaded data...")
            asin_counts = reviews_df['parent_asin'].value_counts()
            popular_asins = asin_counts[asin_counts >= min_reviews]
            if max_asins is not None:
                popular_asins = popular_asins.head(max_asins)
            
            logger.info(f"Found {len(popular_asins)} ASINs with at least {min_reviews} reviews")
            logger.info(f"Top 5 ASINs: {list(popular_asins.head().index)}")
            
            # Save ASINs to file in the category directory
            asins_file = category_dir / f"{category}_asins.txt"
            with open(asins_file, 'w') as f:
                for asin in popular_asins.index:
                    f.write(f"{asin}\n")
            
            logger.info(f"Saved {len(popular_asins)} ASINs to {asins_file}")
            
            # Try to download metadata
            metadata_path = None
            try:
                logger.info("Downloading metadata dataset...")
                meta_dataset = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023",
                    f"raw_meta_{category}",
                    trust_remote_code=True,
                    cache_dir=str(cache_dir)
                )
                metadata_df = meta_dataset["full"].to_pandas()
                
                metadata_path = category_dir / "metadata.parquet"
                metadata_df.to_parquet(metadata_path, index=False)
                logger.info(f"Saved {len(metadata_df)} metadata records to {metadata_path}")
                
            except Exception as e:
                logger.warning(f"Could not download metadata: {e}")
                metadata_df = pd.DataFrame()
            
            # Save summary
            summary = {
                'category': category,
                'download_date': datetime.now().isoformat(),
                'reviews_count': len(reviews_df),
                'metadata_count': len(metadata_df) if not metadata_df.empty else 0,
                'date_range': {
                    'start': reviews_df['datetime'].min().isoformat(),
                    'end': reviews_df['datetime'].max().isoformat()
                },
                'unique_asins': reviews_df['parent_asin'].nunique(),
                'extracted_asins_count': len(popular_asins),
                'min_reviews_threshold': min_reviews,
                'max_asins_limit': max_asins if max_asins is not None else 'unlimited'
            }
            
            summary_path = category_dir / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return {
                'reviews': reviews_path,
                'metadata': metadata_path,
                'summary': summary_path,
                'asins': asins_file
            }
            
        except Exception as e:
            logger.error(f"Failed to download HuggingFace data: {e}")
            raise
    
    def load_local_huggingface_data(self, category: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load HuggingFace data from local storage."""
        logger.info(f"Loading local HuggingFace data for {category}")
        
        category_dir = self.huggingface_dir / category
        
        if not category_dir.exists():
            raise FileNotFoundError(f"HuggingFace data directory not found: {category_dir}")
        
        # Load reviews
        reviews_path = category_dir / "reviews.parquet"
        if not reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
        
        reviews_df = pd.read_parquet(reviews_path)
        reviews_df['datetime'] = pd.to_datetime(reviews_df['datetime'])
        
        # Load metadata if available
        metadata_df = pd.DataFrame()
        metadata_path = category_dir / "metadata.parquet"
        if metadata_path.exists():
            metadata_df = pd.read_parquet(metadata_path)
        
        logger.info(f"Loaded {len(reviews_df)} reviews and {len(metadata_df)} metadata records")
        return reviews_df, metadata_df
    
    def load_local_huggingface_data_polars(self, category: str) -> tuple:
        """Load HuggingFace data from local storage using Polars."""
        import polars as pl
        
        logger.info(f"Loading local HuggingFace data for {category} using Polars")
        
        category_dir = self.huggingface_dir / category
        
        if not category_dir.exists():
            raise FileNotFoundError(f"HuggingFace data directory not found: {category_dir}")
        
        # Load reviews
        reviews_path = category_dir / "reviews.parquet"
        if not reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
        
        reviews_pl = pl.read_parquet(reviews_path)
        # Ensure datetime column is properly formatted (it should already be datetime from parquet)
        if 'datetime' in reviews_pl.columns:
            # If datetime is stored as string, convert it; otherwise leave as is
            if reviews_pl['datetime'].dtype == pl.Utf8:
                reviews_pl = reviews_pl.with_columns([
                    pl.col('datetime').str.strptime(pl.Datetime, format=None, strict=False).alias('datetime')
                ])
        else:
            logger.warning("No datetime column found in reviews data")
        
        # Load metadata if available
        metadata_pl = pl.DataFrame()
        metadata_path = category_dir / "metadata.parquet"
        if metadata_path.exists():
            metadata_pl = pl.read_parquet(metadata_path)
        
        logger.info(f"Loaded {len(reviews_pl)} reviews and {len(metadata_pl)} metadata records")
        return reviews_pl, metadata_pl
    
    def process_huggingface_only(self, category: str, 
                                windowing_strategy: str = "calendar",
                                calendar_window_interval: str = "1d",
                                review_window_size: int = 10,
                                include_all_reviews: bool = True,
                                min_reviews_per_asin: int = 10,
                                max_asins: Optional[int] = None,
                                debug: bool = False,
                                rolling_window_sizes: list[int] = [3, 5, 10, 30],
                                upsample: bool = False,
                                asins_per_batch: int = 1000,
                                slurm: bool = False,
                                sub_dir: Optional[str] = None,
                                out=None) -> None:
        """
        Process HuggingFace review data without Keepa price data using high-performance Polars.
        Processes ASINs in batches, saving each batch to disk to minimize memory usage.
        After all batches are processed, use finalize_batches to concatenate and clean up.
        """
        import polars as pl
        import gc
        userlog = (lambda msg: out.print(msg) if out else logger.info(msg))
        warnlog = (lambda msg: out.print(f"[yellow]{msg}[/yellow]") if out else logger.warning(msg))
        errorlog = (lambda msg: out.print(f"[red]{msg}[/red]") if out else logger.error(msg))
        userlog(f"Processing HuggingFace data for {category} using {windowing_strategy} strategy (Polars, batched, disk)")
        # Load data directly in Polars
        reviews_pl, metadata_pl = self.load_local_huggingface_data_polars(category)
        if reviews_pl.is_empty() or metadata_pl.is_empty():
            warnlog(f"No data found for category {category}")
            return
        
        # Filter ASINs with minimum reviews - use parent_asin for consistency with other methods
        asin_counts = reviews_pl.group_by('parent_asin').len()
        valid_asins_df = asin_counts.filter(pl.col('len') >= min_reviews_per_asin)
        valid_asins = valid_asins_df.select('parent_asin').to_series().to_list()
        
        # Limit to max_asins if specified (for debugging/testing)
        if max_asins is not None:
            valid_asins = valid_asins[:max_asins]
            userlog(f"Limited to first {max_asins} ASINs for debugging/testing")
        userlog(f"Processing {len(valid_asins)} ASINs with at least {min_reviews_per_asin} reviews in batches of {asins_per_batch}")
        if not valid_asins:
            warnlog("No ASINs meet the minimum review requirement")
            return
        # Prepare output batch directory
        output_dir = self.processed_datasets_dir / "huggingface_only"
        if sub_dir:
            output_dir = output_dir / sub_dir
        batch_dir = output_dir / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        n_batches = (len(valid_asins) + asins_per_batch - 1) // asins_per_batch
        for batch_idx in range(n_batches):
            batch_asins = valid_asins[batch_idx*asins_per_batch : (batch_idx+1)*asins_per_batch]
            userlog(f"  [Batch {batch_idx+1}/{n_batches}] Processing {len(batch_asins)} ASINs...")
            batch_reviews = reviews_pl.filter(pl.col('parent_asin').is_in(batch_asins))
            batch_metadata = metadata_pl.filter(pl.col('parent_asin').is_in(batch_asins)) if not metadata_pl.is_empty() else metadata_pl
            reviews_pl_batch = batch_reviews
            metadata_pl_batch = batch_metadata
            if not metadata_pl_batch.is_empty():
                if 'parent_asin' in metadata_pl_batch.columns:
                    reviews_pl_batch = reviews_pl_batch.join(metadata_pl_batch, on='parent_asin', how='left')
                else:
                    warnlog("Metadata does not have parent_asin column, skipping metadata join")
            else:
                warnlog("No metadata available, processing without metadata")
            column_renames = {}
            if 'title_right' in reviews_pl_batch.columns:
                column_renames['title_right'] = 'product_name'
            if 'images_right' in reviews_pl_batch.columns:
                column_renames['images_right'] = 'product_images'
            if 'images' in reviews_pl_batch.columns:
                column_renames['images'] = 'review_images'
            if 'videos' in reviews_pl_batch.columns:
                column_renames['videos'] = 'product_videos'
            if 'title' in reviews_pl_batch.columns:
                column_renames['title'] = 'review_title'
            if column_renames:
                reviews_pl_batch = reviews_pl_batch.rename(column_renames)
            reviews_pl_batch = reviews_pl_batch.with_columns([
                pl.lit(1).alias('count')
            ])
            review_text_cols = []
            for col in ['review_title', 'rating', 'helpful_vote', 'text', 'verified_purchase']:
                if col in reviews_pl_batch.columns:
                    review_text_cols.append(col)
            if review_text_cols:
                reviews_pl_batch = reviews_pl_batch.with_columns([
                    pl.concat_str([
                        pl.lit(r"<\ begin review \> title: "),
                        pl.col('review_title').fill_null(''),
                        pl.lit(" || rating: "),
                        pl.col('rating').cast(pl.Utf8),
                        pl.lit(" || helpful votes: "),
                        pl.col('helpful_vote').fill_null(0).cast(pl.Utf8),
                        pl.lit(" || content: "),
                        pl.col('text').fill_null(''),
                        pl.lit(" || verified purchase: "),
                        pl.col('verified_purchase').fill_null(False).cast(pl.Utf8),
                        pl.lit(r" <\ end review \> ")
                    ]).alias('formatted_review_text')
                ])
            if windowing_strategy == "calendar":
                if upsample:
                    userlog(f"    Creating upsampled time series per ASIN with {calendar_window_interval} windows (batch {batch_idx+1})")
                    reviews_pl_batch = reviews_pl_batch.with_columns([
                        pl.col('datetime').dt.truncate(calendar_window_interval).alias('window_start')
                    ])
                    reviews_pl_batch = reviews_pl_batch.sort(["parent_asin", "datetime"])
                    reviews_pl_batch = reviews_pl_batch.upsample(
                        time_column="window_start",
                        every=calendar_window_interval,
                        group_by="parent_asin",
                        maintain_order=True
                    )
                    fill_expressions = []
                    metadata_columns = ["product_name", "main_category", "categories", "brand", "store", "asin", "parent_asin",
                                      "price", "description", "features", "average_rating", "product_images", "product_videos"]
                    for col in metadata_columns:
                        if col in reviews_pl_batch.columns:
                            fill_expressions.append(pl.col(col).fill_null(strategy="forward"))
                    count_columns = ["helpful_vote", "count"]
                    for col in count_columns:
                        if col in reviews_pl_batch.columns:
                            fill_expressions.append(pl.col(col).fill_null(0))
                    if fill_expressions:
                        reviews_pl_batch = reviews_pl_batch.with_columns(fill_expressions)
                    group_cols = ["parent_asin", "window_start"]
                else:
                    reviews_pl_batch = reviews_pl_batch.with_columns([
                        pl.col('datetime').dt.truncate(calendar_window_interval).alias('window_start')
                    ])
                    group_cols = ['parent_asin', 'window_start']
            elif windowing_strategy == "review_frequency":
                reviews_pl_batch = reviews_pl_batch.sort(['parent_asin', 'datetime'])
                reviews_pl_batch = reviews_pl_batch.with_columns([
                    pl.int_range(pl.len()).over('parent_asin').alias('row_nr')
                ])
                reviews_pl_batch = reviews_pl_batch.with_columns([
                    (pl.col('row_nr') // review_window_size).alias('window_id')
                ])
                group_cols = ['parent_asin', 'window_id']
            else:
                errorlog(f"Unsupported windowing strategy: {windowing_strategy}")
                raise ValueError(f"Unsupported windowing strategy: {windowing_strategy}")
            agg_exprs = [
                pl.col('rating').mean().alias('avg_rating_window'),
                pl.col('rating').sum().alias('sum_rating_window'),  # Add sum of ratings for correct rolling averages
                pl.col('count').sum().alias('review_count_window'),
            ]
            if 'helpful_vote' in reviews_pl_batch.columns and 'rating' in reviews_pl_batch.columns:
                agg_exprs.extend([
                    pl.when(pl.col('helpful_vote').sum() > 0)
                    .then((pl.col('helpful_vote') * pl.col('rating')).sum() / pl.col('helpful_vote').sum())
                    .otherwise(None)
                    .alias('helpful_vote_weighted_avg_rating'),
                    (pl.col('helpful_vote') * pl.col('rating')).sum().alias('helpful_vote_weighted_sum_rating'),  # For correct rolling weighted averages
                    pl.col('helpful_vote').sum().alias('helpful_vote_sum')  # For correct rolling weighted averages
                ])
            else:
                agg_exprs.extend([
                    pl.lit(None).alias('helpful_vote_weighted_avg_rating'),
                    pl.lit(0).alias('helpful_vote_weighted_sum_rating'),  # For correct rolling weighted averages
                    pl.lit(0).alias('helpful_vote_sum')  # For correct rolling weighted averages
                ])
            if 'verified_purchase' in reviews_pl_batch.columns and 'rating' in reviews_pl_batch.columns:
                agg_exprs.extend([
                    pl.when(pl.col('verified_purchase').sum() > 0)
                    .then((pl.col('verified_purchase').cast(pl.Float64) * pl.col('rating')).sum() / pl.col('verified_purchase').sum())
                    .otherwise(None)
                    .alias('verified_purchase_weighted_avg_rating'),
                    (pl.col('verified_purchase').cast(pl.Float64) * pl.col('rating')).sum().alias('verified_purchase_weighted_sum_rating'),  # For correct rolling weighted averages
                    pl.col('verified_purchase').sum().alias('verified_purchase_sum')  # For correct rolling weighted averages
                ])
            else:
                agg_exprs.extend([
                    pl.lit(None).alias('verified_purchase_weighted_avg_rating'),
                    pl.lit(0).alias('verified_purchase_weighted_sum_rating'),  # For correct rolling weighted averages
                    pl.lit(0).alias('verified_purchase_sum')  # For correct rolling weighted averages
                ])
            if 'product_name' in reviews_pl_batch.columns:
                agg_exprs.append(pl.col('product_name').first().alias('product_name'))
            else:
                agg_exprs.append(pl.lit('').alias('product_name'))
            if 'main_category' in reviews_pl_batch.columns:
                agg_exprs.append(pl.col('main_category').first().alias('main_category'))
            else:
                agg_exprs.append(pl.lit(category).alias('main_category'))
            if 'categories' in reviews_pl_batch.columns:
                agg_exprs.append(pl.col('categories').first().alias('categories'))
            else:
                agg_exprs.append(pl.lit('').alias('categories'))
            if 'brand' in reviews_pl_batch.columns:
                agg_exprs.append(pl.col('brand').first().alias('brand'))
            else:
                agg_exprs.append(pl.lit('').alias('brand'))
            if 'store' in reviews_pl_batch.columns:
                agg_exprs.append(pl.col('store').first().alias('store'))
            else:
                agg_exprs.append(pl.lit('').alias('store'))
            # Add author field for Books and Kindle_Store categories
            if category in ["Books", "Kindle_Store"]:
                if 'author' in reviews_pl_batch.columns:
                    agg_exprs.append(pl.col('author').first().alias('author'))
                else:
                    agg_exprs.append(pl.lit('').alias('author'))
            if 'price' in reviews_pl_batch.columns:
                agg_exprs.append(pl.col('price').first().alias('price'))
            else:
                agg_exprs.append(pl.lit('').alias('price'))
            if 'description' in reviews_pl_batch.columns:
                agg_exprs.append(pl.col('description').first().alias('description'))
            else:
                agg_exprs.append(pl.lit('').alias('description'))
            if 'features' in reviews_pl_batch.columns:
                agg_exprs.append(pl.col('features').first().alias('features'))
            else:
                agg_exprs.append(pl.lit('').alias('features'))
            if 'average_rating' in reviews_pl_batch.columns:
                agg_exprs.append(pl.col('average_rating').first().alias('avg_rating_metadata'))
            else:
                agg_exprs.append(pl.lit(None).alias('avg_rating_metadata'))
            if 'formatted_review_text' in reviews_pl_batch.columns:
                agg_exprs.append(
                    pl.col('formatted_review_text').drop_nulls().str.join('').alias('aggregated_reviews')
                )
            else:
                agg_exprs.append(pl.lit('').alias('aggregated_reviews'))
            if windowing_strategy == "calendar":
                agg_exprs.append(pl.col('window_start').first().alias('timestamp'))
            else:
                agg_exprs.append(pl.col('datetime').first().alias('timestamp'))
            aggregated = reviews_pl_batch.group_by(group_cols).agg(agg_exprs)
            if upsample:
                aggregated = aggregated.with_columns([
                    pl.col('review_count_window').fill_null(0)
                ])
            aggregated = aggregated.with_columns([
                pl.col('parent_asin').alias('asin')
            ])
            def list_to_str(col_name):
                return (
                    pl.when(pl.col(col_name).is_not_null())
                    .then(
                        pl.col(col_name)
                        .list.eval(pl.element().cast(str).fill_null(''))
                        .list.join(', ')
                    )
                    .otherwise(pl.lit(''))
                    .alias(f"{col_name}_str")
                )
            aggregated = aggregated.with_columns([
                list_to_str('categories'),
                list_to_str('features'),
                list_to_str('description'),
            ])
            # Build combined text with conditional author field for Books and Kindle_Store categories
            combined_text_parts = [
                pl.lit("product name: "),
                pl.col('product_name').fill_null(''),
                pl.lit("\nmain category: "),
                pl.col('main_category').fill_null(''),
                pl.lit("\nproduct categories: "),
                pl.col('categories_str').fill_null(''),
                pl.lit("\n2023 price: "),
                pl.col('price').fill_null(''),
                pl.lit("\nbrand: "),
                pl.col('brand').fill_null('')
            ]
            
            # Add author field for Books and Kindle_Store categories (after brand, before seller)
            if category in ["Books", "Kindle_Store"]:
                combined_text_parts.extend([
                    pl.lit("\nauthor: "),
                    pl.col('author').fill_null('')
                ])
            
            combined_text_parts.extend([
                pl.lit("\nseller: "),
                pl.col('store').fill_null(''),
                pl.lit("\nproduct description: "),
                pl.col('description_str').fill_null(''),
                pl.lit("\nreviews: "),
                pl.col('aggregated_reviews').fill_null(''),
                pl.lit("\nproduct features: "),
                pl.col('features_str').fill_null('')
            ])
            
            aggregated = aggregated.with_columns([
                pl.concat_str(combined_text_parts).alias('combined_text')
            ])
            aggregated = aggregated.with_columns([
                pl.col('timestamp').dt.date().alias('date'),
                pl.col('timestamp').dt.year().alias('year'),
                pl.col('timestamp').dt.month().alias('month')
            ])
            
            # Drop redundant columns that are already included in combined_text
            columns_to_drop = [
                'product_name', 'main_category', 'categories', 'brand', 'store', 'price', 'description', 'features',
                'avg_rating_metadata', 'aggregated_reviews', 'categories_str', 'features_str', 'description_str'
            ]
            existing_cols_to_drop = [col for col in columns_to_drop if col in aggregated.columns]
            if existing_cols_to_drop:
                aggregated = aggregated.drop(existing_cols_to_drop)

            userlog(f"    [Batch {batch_idx+1}/{n_batches}] Adding rolling window statistics (Polars)...")
            try:
                aggregated = self._add_rolling_window_statistics(aggregated, rolling_window_sizes)
            except Exception as e:
                if debug:
                    errorlog(f"Error adding rolling window statistics in batch {batch_idx+1}: {e}")
                    import traceback
                    errorlog(f"Full traceback:\n{traceback.format_exc()}")
                    raise
                else:
                    warnlog(f"Error adding rolling window statistics in batch {batch_idx+1}: {e}")
                    userlog("Continuing without rolling window statistics for this batch")
            
            batch_file = batch_dir / f"{category}_batch{batch_idx+1}.parquet"
            aggregated.write_parquet(batch_file)
            userlog(f"  [Batch {batch_idx+1}] Saved to {batch_file}")
            del aggregated, reviews_pl_batch, batch_reviews, batch_metadata, metadata_pl_batch
            gc.collect()
        userlog(f"All batches processed and saved to {batch_dir}. Use finalize_batches to concatenate and clean up.")

    def finalize_batches(self, category: str, sub_dir: Optional[str] = None, out=None) -> None:
        """
        Concatenate all batch parquet files for a category/sub_dir, save as final merged parquet, and clean up batch files only if successful.
        """
        import polars as pl
        userlog = (lambda msg: out.print(msg) if out else logger.info(msg))
        warnlog = (lambda msg: out.print(f"[yellow]{msg}[/yellow]") if out else logger.warning(msg))
        errorlog = (lambda msg: out.print(f"[red]{msg}[/red]") if out else logger.error(msg))
        output_dir = self.processed_datasets_dir / "huggingface_only"
        if sub_dir:
            output_dir = output_dir / sub_dir
        batch_dir = output_dir / "batches"
        batch_files = sorted(batch_dir.glob(f"{category}_batch*.parquet"))
        if not batch_files:
            errorlog(f"No batch files found in {batch_dir}")
            return
        userlog(f"Concatenating {len(batch_files)} batch files for {category}")
        
        # Use Polars to read and concatenate batch files
        dfs = []
        for f in batch_files:
            userlog(f"  Loading {f.name}")
            dfs.append(pl.read_parquet(f))
        merged = pl.concat(dfs, how="vertical")
        
        final_file = output_dir / f"{category}.parquet"
        try:
            merged.write_parquet(final_file)
            userlog(f"Saved merged dataset to {final_file}")
            for f in batch_files:
                f.unlink()
            batch_dir.rmdir()
            userlog(f"Cleaned up batch files in {batch_dir}")
        except Exception as e:
            errorlog(f"Failed to save merged dataset: {e}")
            errorlog("Batch files NOT deleted. Please check disk space and try again.")

    def combine_datasets(self, category: str, time_window_days: int = 30, 
                        include_all_reviews: bool = True, 
                        windowing_strategy: str = "calendar",
                        calendar_window_interval: str = "1d",
                        review_window_size: int = 10,
                        debug: bool = False) -> pd.DataFrame:
        """
        Combine Keepa price data with Amazon review data using different windowing strategies.
        
        Args:
            category: Amazon product category
            time_window_days: Days around price observation (for price_observation strategy)
            include_all_reviews: Include all reviews in time window
            windowing_strategy: One of "price_observation", "calendar", or "review_frequency"
            calendar_window_days: Days per calendar window (for calendar strategy)
            review_window_size: Reviews per window (for review_frequency strategy)
        """
        logger.info(f"Combining datasets using {windowing_strategy} strategy")
        
        # Load HuggingFace data
        reviews_df, metadata_df = self.load_local_huggingface_data(category)
        
        # Load Keepa data from local storage
        # Ensure data_dir exists for Keepa operations
        self.data_dir.mkdir(parents=True, exist_ok=True)
        json_files = list(self.data_dir.glob("*_complete.json"))
        if not json_files:
            raise RuntimeError("No Keepa data files found. Run keepa.py download first.")
        
        combined_records = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing ASINs", total=len(json_files))
            
            for json_file in json_files:
                asin = json_file.stem.replace("_complete", "")
                progress.update(task, description=f"Processing {asin}")
                
                try:
                    # Load Keepa data
                    with open(json_file, 'r') as f:
                        keepa_data = json.load(f)
                    
                    # Parse price history
                    parse_price_history, _ = _import_analyze_history()
                    price_df = parse_price_history({"products": [keepa_data]})
                    
                    if price_df.empty:
                        logger.warning(f"No price data for {asin}")
                        progress.advance(task)
                        continue
                    
                    # Get product metadata
                    product_meta = metadata_df[metadata_df['parent_asin'] == asin]
                    if not product_meta.empty:
                        meta_row = product_meta.iloc[0]
                        product_title = meta_row.get('title', '')
                        product_category = meta_row.get('main_category', category)
                        product_brand = meta_row.get('brand', '')
                        product_price = meta_row.get('price', '')
                        avg_rating = meta_row.get('average_rating', None)
                        rating_count = meta_row.get('rating_number', None)
                    else:
                        product_title = f"Product_{asin}"
                        product_category = category
                        product_brand = ""
                        product_price = ""
                        avg_rating = None
                        rating_count = None
                    
                    # Get reviews for this ASIN
                    asin_reviews = reviews_df[
                        (reviews_df['asin'] == asin) | (reviews_df['parent_asin'] == asin)
                    ].copy().sort_values('datetime')
                    
                    if windowing_strategy == "price_observation":
                        combined_records.extend(self._process_price_observation_windows(
                            asin, price_df, asin_reviews, product_title, product_category, 
                            product_brand, product_price, avg_rating, rating_count,
                            time_window_days, include_all_reviews
                        ))
                    elif windowing_strategy == "calendar":
                        combined_records.extend(self._process_calendar_windows(
                            asin, price_df, asin_reviews, product_title, product_category,
                            product_brand, product_price, avg_rating, rating_count,
                            calendar_window_interval, include_all_reviews
                        ))
                    elif windowing_strategy == "review_frequency":
                        combined_records.extend(self._process_review_frequency_windows(
                            asin, price_df, asin_reviews, product_title, product_category,
                            product_brand, product_price, avg_rating, rating_count,
                            review_window_size, include_all_reviews
                        ))
                    else:
                        raise ValueError(f"Unknown windowing strategy: {windowing_strategy}")
                        
                except Exception as e:
                    logger.error(f"Error processing {asin}: {e}")
                    continue
                
                progress.advance(task)
        
        combined_df = pd.DataFrame(combined_records)
        logger.info(f"Combined dataset created: {len(combined_df)} records, {combined_df['asin'].nunique()} unique ASINs")
        

        
        return combined_df
    
    def _process_price_observation_windows(self, asin: str, price_df: pd.DataFrame, 
                                         asin_reviews: pd.DataFrame, product_title: str,
                                         product_category: str, product_brand: str,
                                         product_price: str, avg_rating: float, rating_count: int,
                                         time_window_days: int, include_all_reviews: bool) -> List[Dict]:
        """Process windows centered on price observations."""
        records = []
        
        for _, price_row in price_df.iterrows():
            timestamp = price_row.name
            
            # Define time window around price observation
            start_date = timestamp - pd.Timedelta(days=time_window_days)
            end_date = timestamp + pd.Timedelta(days=time_window_days)
            
            # Get reviews in time window
            window_reviews = asin_reviews[
                (asin_reviews['datetime'] >= start_date) & 
                (asin_reviews['datetime'] <= end_date)
            ]
            
            record = self._create_record_from_window(
                asin, timestamp, price_row, window_reviews, product_title, product_category,
                product_brand, product_price, avg_rating, rating_count, time_window_days,
                include_all_reviews, "price_observation"
            )
            records.append(record)
        
        return records
    
    def _process_calendar_windows(self, asin: str, price_df: pd.DataFrame,
                                asin_reviews: pd.DataFrame, product_title: str,
                                product_category: str, product_brand: str,
                                product_price: str, avg_rating: float, rating_count: int,
                                calendar_window_days: int, include_all_reviews: bool) -> List[Dict]:
        """Process fixed calendar windows."""
        records = []
        
        # Skip ASINs with no reviews entirely
        if asin_reviews.empty:
            return records
        
        # Determine date range from reviews
        start_date = asin_reviews['datetime'].min().normalize()
        end_date = asin_reviews['datetime'].max().normalize()
        
        # Create calendar windows
        current_date = start_date
        while current_date <= end_date:
            window_start = current_date
            window_end = current_date + pd.Timedelta(days=calendar_window_days)
            window_midpoint = window_start + pd.Timedelta(days=calendar_window_days/2)
            
            # Get reviews in this calendar window
            window_reviews = asin_reviews[
                (asin_reviews['datetime'] >= window_start) & 
                (asin_reviews['datetime'] < window_end)
            ]
            
            # Find the most recent price before or during this window
            available_prices = price_df[price_df.index <= window_midpoint]
            
            # Create record even if no price data is available
            if not available_prices.empty:
                # Use the most recent price
                latest_price_idx = available_prices.index[-1]
                price_row = price_df.loc[latest_price_idx]
            else:
                # Create empty price row with None values
                price_row = pd.Series({
                    'AMAZON': None, 'NEW': None, 'USED': None, 'LISTPRICE': None,
                    'SALES_RANK': None, 'RATING': None, 'COUNT_REVIEWS': None,
                    'COUNT_NEW': None, 'COUNT_USED': None
                })
            
            record = self._create_record_from_window(
                asin, window_midpoint, price_row, window_reviews, product_title, product_category,
                product_brand, product_price, avg_rating, rating_count, calendar_window_days,
                include_all_reviews, "calendar"
            )
            records.append(record)
            
            current_date = window_end
        
        return records
    
    def _process_review_frequency_windows(self, asin: str, price_df: pd.DataFrame,
                                        asin_reviews: pd.DataFrame, product_title: str,
                                        product_category: str, product_brand: str,
                                        product_price: str, avg_rating: float, rating_count: int,
                                        review_window_size: int, include_all_reviews: bool) -> List[Dict]:
        """Process windows based on review frequency."""
        records = []
        
        if asin_reviews.empty:
            return records
        
        # Create review-based windows
        for i in range(0, len(asin_reviews), review_window_size):
            window_reviews = asin_reviews.iloc[i:i+review_window_size]
            
            if not window_reviews.empty:
                # Use the midpoint of the review window as timestamp
                window_midpoint = window_reviews['datetime'].iloc[len(window_reviews)//2]
                
                # Find the most recent price before or during this window
                available_prices = price_df[price_df.index <= window_midpoint]
                
                # Create record even if no price data is available
                if not available_prices.empty:
                    # Use the most recent price
                    latest_price_idx = available_prices.index[-1]
                    price_row = price_df.loc[latest_price_idx]
                else:
                    # Create empty price row with None values
                    price_row = pd.Series({
                        'AMAZON': None, 'NEW': None, 'USED': None, 'LISTPRICE': None,
                        'SALES_RANK': None, 'RATING': None, 'COUNT_REVIEWS': None,
                        'COUNT_NEW': None, 'COUNT_USED': None
                    })
                
                record = self._create_record_from_window(
                    asin, window_midpoint, price_row, window_reviews, product_title, product_category,
                    product_brand, product_price, avg_rating, rating_count, review_window_size,
                    include_all_reviews, "review_frequency"
                )
                records.append(record)
        
        return records
    
    def _create_record_from_window(self, asin: str, timestamp: pd.Timestamp, 
                                 price_row: pd.Series, window_reviews: pd.DataFrame,
                                 product_title: str, product_category: str, product_brand: str,
                                 product_price: str, avg_rating: float, rating_count: int,
                                 window_size: int, include_all_reviews: bool, 
                                 windowing_strategy: str) -> Dict:
        """Create a record from a window of reviews and price data."""
        
        if len(window_reviews) > 0:
            review_count_window = len(window_reviews)
            avg_rating_window = window_reviews['rating'].mean()
            sum_rating_window = window_reviews['rating'].sum()  # Add sum of ratings for correct rolling averages
            helpful_votes = window_reviews['helpful_vote'].sum()
            verified_ratio = window_reviews['verified_purchase'].mean()
            
            # Add weighted sums for correct rolling weighted averages
            helpful_vote_weighted_sum_rating = (window_reviews['helpful_vote'] * window_reviews['rating']).sum()
            verified_purchase_weighted_sum_rating = (window_reviews['verified_purchase'] * window_reviews['rating']).sum()
            verified_purchase_sum = window_reviews['verified_purchase'].sum()
            
            # Handle review text based on include_all_reviews flag
            if include_all_reviews and len(window_reviews) > 1:
                # Combine all reviews with delimiters
                review_texts = []
                for _, review in window_reviews.iterrows():
                    text = review['text'].fillna('')
                    if text.strip():
                        review_texts.append(text)
                
                if review_texts:
                    combined_reviews = " |*| end review |*| ".join(review_texts)
                    sample_review = f"|*| start new review |*| {combined_reviews} |*| end review |*|"
                else:
                    sample_review = ""
            else:
                # Use single sample review (original behavior)
                sample_review = window_reviews['text'].fillna('').iloc[0] if len(window_reviews) > 0 else ''
        else:
            review_count_window = 0
            avg_rating_window = None
            sum_rating_window = 0  # Add sum of ratings for correct rolling averages
            helpful_votes = 0
            verified_ratio = 0.0
            sample_review = ''
            
            # Add weighted sums for correct rolling weighted averages
            helpful_vote_weighted_sum_rating = 0
            verified_purchase_weighted_sum_rating = 0
            verified_purchase_sum = 0
        
        record = {
            'asin': asin,
            'timestamp': timestamp,
            'date': timestamp.date(),
            'year': timestamp.year,
            'month': timestamp.month,
            
            'title': product_title,
            'category': product_category, 
            'brand': product_brand,
            'price_metadata': product_price,
            'avg_rating_metadata': avg_rating,
            'rating_count_metadata': rating_count,
            
            'amazon_price': price_row.get('AMAZON') if price_row.get('AMAZON') is not None else None,
            'new_price': price_row.get('NEW') if price_row.get('NEW') is not None else None, 
            'used_price': price_row.get('USED') if price_row.get('USED') is not None else None,
            'list_price': price_row.get('LISTPRICE') if price_row.get('LISTPRICE') is not None else None,
            'sales_rank': price_row.get('SALES_RANK') if price_row.get('SALES_RANK') is not None else None,
            'rating_keepa': price_row.get('RATING') if price_row.get('RATING') is not None else None,
            'review_count_keepa': price_row.get('COUNT_REVIEWS') if price_row.get('COUNT_REVIEWS') is not None else None,
            'new_offers_count': price_row.get('COUNT_NEW') if price_row.get('COUNT_NEW') is not None else None,
            'used_offers_count': price_row.get('COUNT_USED') if price_row.get('COUNT_USED') is not None else None,
            
            'review_count_window': review_count_window,
            'avg_rating_window': avg_rating_window,
            'sum_rating_window': sum_rating_window,  # Add sum of ratings for correct rolling averages
            'helpful_votes_sum': helpful_votes,
            'verified_purchases_ratio': verified_ratio,
            
            # Add weighted sums for correct rolling weighted averages
            'helpful_vote_weighted_sum_rating': helpful_vote_weighted_sum_rating,
            'helpful_vote_sum': helpful_votes,  # This is the same as helpful_votes_sum but with consistent naming
            'verified_purchase_weighted_sum_rating': verified_purchase_weighted_sum_rating,
            'verified_purchase_sum': verified_purchase_sum,
            'review_text': sample_review[:5000] if include_all_reviews else sample_review[:500],
            'window_size': window_size,
            'windowing_strategy': windowing_strategy,
            'include_all_reviews': include_all_reviews
        }
        
        return record
    
    def save_organized_data(self, df: pd.DataFrame, category: str, 
                           config_metadata: Optional[Dict] = None, sub_dir: Optional[str] = None) -> Dict[str, Path]:
        """
        Save organized dataset in multiple formats with configuration metadata.
        
        Args:
            df: DataFrame to save
            category: Product category
            config_metadata: Optional configuration parameters used to create the dataset
            sub_dir: Optional subdirectory for organizing different configurations
        """
        logger.info("Saving organized dataset")
        
        if df.empty:
            logger.warning("No data to save")
            return {}
        
        # Determine output directory and base name based on data source
        if config_metadata and config_metadata.get('data_source') == 'huggingface_only':
            # Use new directory structure for HuggingFace-only data
            output_dir = self.processed_datasets_dir / "huggingface_only"
            if sub_dir:
                output_dir = output_dir / sub_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = category
        elif config_metadata and config_metadata.get('data_source') == 'keepa_huggingface_combined':
            # Use processed_datasets for combined data as well
            output_dir = self.processed_datasets_dir / "keepa_huggingface_combined"
            if sub_dir:
                output_dir = output_dir / sub_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"amazon_keepa_{category}_{datetime.now().strftime('%Y%m%d')}"
        else:
            # Default fallback - use processed_datasets
            output_dir = self.processed_datasets_dir / "default"
            if sub_dir:
                output_dir = output_dir / sub_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"amazon_keepa_{category}_{datetime.now().strftime('%Y%m%d')}"
        
        output_paths = {}
        
        # Save Parquet
        parquet_path = output_dir / f"{base_name}.parquet"
        df.to_parquet(parquet_path, index=False)
        output_paths['parquet'] = parquet_path
        logger.info(f"Saved Parquet: {parquet_path}")
        
        # For HuggingFace-only datasets, save metadata separately
        if config_metadata and config_metadata.get('data_source') == 'huggingface_only':
            metadata_path = self._save_product_metadata(df, category, output_dir, base_name)
            if metadata_path:
                output_paths['metadata'] = metadata_path
        
        # Create summary with configuration metadata
        summary = {
            'created_at': datetime.now().isoformat(),
            'category': category,
            'total_records': len(df),
            'unique_asins': df['asin'].nunique(),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'columns': list(df.columns),
            'review_data_coverage': {
                'avg_reviews_per_timepoint': df['review_count_window'].mean(),
                'timepoints_with_reviews': f"{(df['review_count_window'] > 0).mean():.1%}"
            }
        }
        
        # Add metadata file information for HuggingFace-only datasets
        if config_metadata and config_metadata.get('data_source') == 'huggingface_only':
            summary['metadata_file'] = f"{base_name}_metadata.parquet"
            summary['note'] = "Product metadata (title, category, brand, etc.) has been moved to a separate metadata file to reduce redundancy"
        
        # Add price data coverage only if price columns exist
        price_columns = ['amazon_price', 'new_price', 'used_price']
        existing_price_columns = [col for col in price_columns if col in df.columns]
        if existing_price_columns:
            summary['price_data_coverage'] = {
                col: f"{df[col].notna().mean():.1%}" 
                for col in existing_price_columns
            }
        else:
            summary['price_data_coverage'] = "No price data available"
        
        # Add configuration metadata if provided
        if config_metadata:
            summary['processing_config'] = config_metadata
        
        summary_path = output_dir / f"{base_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        output_paths['summary'] = summary_path
        logger.info(f"Saved summary: {summary_path}")
        
        return output_paths

    def _save_product_metadata(self, df: pd.DataFrame, category: str, output_dir: Path, base_name: str) -> Optional[Path]:
        """
        Save product metadata separately for HuggingFace-only datasets.
        
        Args:
            df: DataFrame containing the processed data
            category: Product category
            output_dir: Directory to save metadata
            base_name: Base name for the file
            
        Returns:
            Path to the saved metadata file, or None if no metadata to save
        """
        # Load original metadata to get complete product information
        try:
            _, metadata_df = self.load_local_huggingface_data(category)
            if metadata_df.empty:
                logger.warning("No metadata available to save separately")
                return None
            
            # Get unique ASINs from the processed data
            unique_asins = df['asin'].unique()
            
            # Filter metadata to only include ASINs in the processed data
            # Use parent_asin for joining since that's what we used in processing
            if 'parent_asin' in metadata_df.columns:
                product_metadata = metadata_df[metadata_df['parent_asin'].isin(unique_asins)].copy()
                product_metadata = product_metadata.drop_duplicates(subset=['parent_asin'])
                
                # Drop existing 'asin' column if it exists to avoid duplicates
                if 'asin' in product_metadata.columns:
                    product_metadata = product_metadata.drop(columns=['asin'])
                
                # Rename parent_asin to asin for consistency
                product_metadata = product_metadata.rename(columns={'parent_asin': 'asin'})
                
                # Select relevant columns
                metadata_columns = ['asin', 'title', 'main_category', 'brand', 'price', 'average_rating', 'rating_number']
                available_columns = [col for col in metadata_columns if col in product_metadata.columns]
                
                if available_columns:
                    product_metadata = product_metadata[available_columns]
                    
                    # Rename columns to match the expected format
                    column_mapping = {
                        'main_category': 'category',
                        'price': 'price_metadata',
                        'average_rating': 'avg_rating_metadata',
                        'rating_number': 'rating_count_metadata'
                    }
                    product_metadata = product_metadata.rename(columns=column_mapping)
                    
                    # Save metadata
                    metadata_path = output_dir / f"{base_name}_metadata.parquet"
                    product_metadata.to_parquet(metadata_path, index=False)
                    logger.info(f"Saved product metadata: {metadata_path} ({len(product_metadata)} products)")
                    return metadata_path
                else:
                    logger.warning("No relevant metadata columns found")
                    return None
            else:
                logger.warning("No parent_asin column in metadata")
                return None
                
        except Exception as e:
            logger.warning(f"Could not save product metadata: {e}")
            return None

    def get_available_categories(self) -> List[str]:
        """Get available Amazon product categories from the all_categories.txt file."""
        logger.info("Loading available Amazon product categories from all_categories.txt")
        
        # Look for the categories file in the keepa directory
        categories_file = self.keepa_dir / "all_categories.txt"
        
        if not categories_file.exists():
            logger.error(f"Categories file not found: {categories_file}")
            raise FileNotFoundError(f"Categories file not found: {categories_file}")
        
        try:
            with open(categories_file, 'r') as f:
                categories = [line.strip() for line in f if line.strip()]
            
            # Sort alphabetically
            categories.sort()
            
            logger.info(f"Loaded {len(categories)} categories from {categories_file}")
            return categories
            
        except Exception as e:
            logger.error(f"Failed to read categories file: {e}")
            raise
    
    def get_category_info(self, category: str) -> Dict[str, any]:
        """Get information about a specific category."""
        logger.info(f"Getting information for category: {category}")
        
        info = {
            'category': category,
            'local_data_exists': False,
            'reviews_count': 0,
            'metadata_count': 0,
            'date_range': None,
            'unique_asins': 0
        }
        
        # Check if local data exists
        category_dir = self.huggingface_dir / category
        if category_dir.exists():
            info['local_data_exists'] = True
            
            # Load summary if available
            summary_path = category_dir / "summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    info.update(summary)
                except Exception as e:
                    logger.warning(f"Could not load summary for {category}: {e}")
            
            # Try to get more detailed info from actual data files
            reviews_path = category_dir / "reviews.parquet"
            if reviews_path.exists():
                try:
                    reviews_df = pd.read_parquet(reviews_path)
                    info['reviews_count'] = len(reviews_df)
                    if 'datetime' in reviews_df.columns:
                        info['date_range'] = {
                            'start': reviews_df['datetime'].min().isoformat(),
                            'end': reviews_df['datetime'].max().isoformat()
                        }
                    if 'parent_asin' in reviews_df.columns:
                        info['unique_asins'] = reviews_df['parent_asin'].nunique()
                except Exception as e:
                    logger.warning(f"Could not read reviews data for {category}: {e}")
            
            metadata_path = category_dir / "metadata.parquet"
            if metadata_path.exists():
                try:
                    metadata_df = pd.read_parquet(metadata_path)
                    info['metadata_count'] = len(metadata_df)
                except Exception as e:
                    logger.warning(f"Could not read metadata for {category}: {e}")
        
        return info

    def _drop_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove metadata columns that are redundant (in config)."""
        # Remove configuration metadata columns
        config_cols = ['window_size', 'windowing_strategy', 'include_all_reviews']
        for col in config_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Remove redundant product metadata columns that are identical across all observations for a given ASIN
        # These should be stored in metadata files instead of repeated in every row
        redundant_metadata_cols = [
            'title', 'category', 'brand', 'price_metadata', 
            'avg_rating_metadata', 'rating_count_metadata'
        ]
        for col in redundant_metadata_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.info(f"Removed redundant metadata column: {col}")
        
        return df

    def _add_rolling_window_statistics(self, df_pl, window_sizes: list[int] = [3, 5, 10, 30]):
        """
        Add rolling statistics across a number of windows (not days) using native Polars operations.
        
        This method uses native Polars vectorized operations for optimal performance.
        
        Args:
            df_pl: Polars DataFrame
            window_sizes: List of window sizes (number of observations, not days)
            
        Returns:
            Polars DataFrame with additional rolling statistics columns
        """
        import polars as pl
        
        logger.info(f"Adding rolling window statistics using native Polars operations (windows: {window_sizes})")
        
        # Sort by ASIN and timestamp
        df_pl = df_pl.sort(['asin', 'timestamp'])
        
        # Create expressions for rolling statistics
        rolling_expressions = []
        
        for w in window_sizes:
            # Correct rolling average: sum of ratings / sum of review counts (not average of averages)
            rolling_expressions.append(
                (pl.col('sum_rating_window').fill_null(0).rolling_sum(window_size=w, min_samples=1) / 
                 pl.col('review_count_window').fill_null(0).rolling_sum(window_size=w, min_samples=1))
                .over('asin').alias(f'rolling_avg_rating_{w}w')
            )
            
            # Rolling review count (include 0s from empty windows)
            rolling_expressions.append(
                pl.col('review_count_window').fill_null(0).rolling_sum(window_size=w, min_samples=1)
                .over('asin').alias(f'rolling_review_count_{w}w')
            )
            
            # Add rolling statistics for weighted averages if they exist (using correct weighted sum approach)
            if 'helpful_vote_weighted_sum_rating' in df_pl.columns and 'helpful_vote_sum' in df_pl.columns:
                rolling_expressions.append(
                    (pl.col('helpful_vote_weighted_sum_rating').fill_null(0).rolling_sum(window_size=w, min_samples=1) / 
                     pl.col('helpful_vote_sum').fill_null(0).rolling_sum(window_size=w, min_samples=1))
                    .over('asin').alias(f'rolling_helpful_vote_weighted_avg_{w}w')
                )
            
            if 'verified_purchase_weighted_sum_rating' in df_pl.columns and 'verified_purchase_sum' in df_pl.columns:
                rolling_expressions.append(
                    (pl.col('verified_purchase_weighted_sum_rating').fill_null(0).rolling_sum(window_size=w, min_samples=1) / 
                     pl.col('verified_purchase_sum').fill_null(0).rolling_sum(window_size=w, min_samples=1))
                    .over('asin').alias(f'rolling_verified_purchase_weighted_avg_{w}w')
                )
        
        # Add all rolling statistics columns at once using vectorized operations
        df_pl = df_pl.with_columns(rolling_expressions)
        
        logger.info(f"Successfully added {len(rolling_expressions)} rolling window statistics columns")
        
        return df_pl

    def load_dataset_with_metadata(self, category: str, work_dir: Path = Path("."), sub_dir: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load a processed HuggingFace-only dataset along with its metadata file.
        
        Args:
            category: Product category
            work_dir: Working directory
            sub_dir: Optional subdirectory where the dataset is stored
            
        Returns:
            Tuple of (main_dataset, metadata_dataset)
        """
        # Determine file paths
        processed_dir = work_dir / "processed_datasets" / "huggingface_only"
        if sub_dir:
            processed_dir = processed_dir / sub_dir
        main_file = processed_dir / f"{category}.parquet"
        metadata_file = processed_dir / f"{category}_metadata.parquet"
        
        if not main_file.exists():
            raise FileNotFoundError(f"Main dataset file not found: {main_file}")
        
        # Load main dataset
        main_df = pd.read_parquet(main_file)
        
        # Load metadata if available
        metadata_df = pd.DataFrame()
        if metadata_file.exists():
            metadata_df = pd.read_parquet(metadata_file)
            logger.info(f"Loaded metadata for {len(metadata_df)} products")
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
        
        return main_df, metadata_df
    
    def load_dataset_with_metadata_polars(self, category: str, work_dir: Path = Path("."), sub_dir: Optional[str] = None) -> tuple:
        """
        Load a processed HuggingFace-only dataset along with its metadata file using Polars.
        
        Args:
            category: Product category
            work_dir: Working directory
            sub_dir: Optional subdirectory where the dataset is stored
            
        Returns:
            Tuple of (main_dataset, metadata_dataset) as Polars DataFrames
        """
        import polars as pl
        
        # Determine file paths
        processed_dir = work_dir / "processed_datasets" / "huggingface_only"
        if sub_dir:
            processed_dir = processed_dir / sub_dir
        main_file = processed_dir / f"{category}.parquet"
        metadata_file = processed_dir / f"{category}_metadata.parquet"
        
        if not main_file.exists():
            raise FileNotFoundError(f"Main dataset file not found: {main_file}")
        
        # Load main dataset
        main_df = pl.read_parquet(main_file)
        
        # Load metadata if available
        metadata_df = pl.DataFrame()
        if metadata_file.exists():
            metadata_df = pl.read_parquet(metadata_file)
            logger.info(f"Loaded metadata for {len(metadata_df)} products")
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
        
        return main_df, metadata_df


@app.command()
def download_huggingface(
    category: str = typer.Option("All_Beauty", help="Amazon product category"),
    min_reviews: int = typer.Option(10, help="Minimum reviews per ASIN"),
    max_asins: Optional[int] = typer.Option(None, help="Maximum ASINs to extract (None = all ASINs)"),
    sample_reviews: Optional[int] = typer.Option(None, help="Sample size for testing"),
    work_dir: Path = typer.Option(".", help="Working directory"),
    slurm: bool = typer.Option(False, "--slurm", help="Flush log output for SLURM/plain mode")
):
    """Download HuggingFace data, save locally, and extract ASINs in one operation."""
    setup_logging(slurm)
    max_asins_display = "All ASINs" if max_asins is None else str(max_asins)
    with SlurmOutput(slurm) as out:
        out.panel(
            f"[bold blue]Downloading HuggingFace Data & Extracting ASINs[/bold blue]\n"
            f"Category: {category}\n"
            f"Min Reviews: {min_reviews}\n"
            f"Max ASINs: {max_asins_display}",
            border_style="blue"
        )
        pipeline = AmazonKeepaDataPipeline(work_dir)
        try:
            output_paths = pipeline.download_huggingface_data(category, min_reviews, max_asins, sample_reviews)
            out.print("\n[bold green]Download and extraction completed![/bold green]")
            out.print("Files saved:")
            for file_type, path in output_paths.items():
                if path:
                    out.print(f"  {file_type}: {path}")
            out.print(f"\n[bold]Next Steps:[/bold]")
            out.print(f"  Use ASIN file with: python keepa.py download <api_key_file> {output_paths['asins']}")
            out.print(f"  Then merge datasets: python dataset_builder.py merge-datasets --category {category}")
        except Exception as e:
            out.print(f"[bold red]Download failed: {e}[/bold red]")
            raise typer.Exit(1)

@app.command()
def merge_datasets(
    category: str = typer.Option("All_Beauty", help="Amazon product category"),
    windowing_strategy: str = typer.Option("price_observation", help="Windowing strategy: price_observation, calendar, or review_frequency"),
    time_window_days: int = typer.Option(30, help="Days around price observation (for price_observation strategy)"),
    calendar_window_interval: str = typer.Option("1d", help="Time interval per calendar window (e.g., 1d, 2M, 1w) for calendar strategy"),
    review_window_size: int = typer.Option(10, help="Reviews per window (for review_frequency strategy)"),
    include_all_reviews: bool = typer.Option(True, help="Include all reviews in time window"),
    work_dir: Path = typer.Option(".", help="Working directory"),
    pull_huggingface: bool = typer.Option(False, help="Pull HuggingFace data from cloud instead of using local"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with full tracebacks and stop on first error"),
    slurm: bool = typer.Option(False, "--slurm", help="Flush log output for SLURM/plain mode")
):
    """Merge Keepa price data with Amazon review data using different windowing strategies."""
    setup_logging(slurm)
    valid_strategies = ["price_observation", "calendar", "review_frequency"]
    with SlurmOutput(slurm) as out:
        if windowing_strategy not in valid_strategies:
            out.print(f"[bold red]Invalid windowing strategy: {windowing_strategy}[/bold red]")
            out.print(f"Valid options: {', '.join(valid_strategies)}")
            raise typer.Exit(1)
        if windowing_strategy == "price_observation":
            strategy_desc = f"Price observation windows ({time_window_days} days around each price change)"
        elif windowing_strategy == "calendar":
            strategy_desc = f"Calendar windows ({calendar_window_interval} intervals)"
        else:
            strategy_desc = f"Review frequency windows (every {review_window_size} reviews)"
        out.panel(
            f"[bold blue]Merging Datasets[/bold blue]\n"
            f"Category: {category}\n"
            f"Strategy: {strategy_desc}\n"
            f"Include All Reviews: {include_all_reviews}",
            border_style="blue"
        )
        pipeline = AmazonKeepaDataPipeline(work_dir)
        try:
            # Ensure data_dir exists for Keepa operations
            pipeline.data_dir.mkdir(parents=True, exist_ok=True)
            keepa_files = list(pipeline.data_dir.glob("*_complete.json"))
            if not keepa_files:
                out.print("[bold red]No Keepa data found! Run keepa.py download first.[/bold red]")
                raise typer.Exit(1)
            out.print(f"Found {len(keepa_files)} Keepa data files")
            if pull_huggingface:
                out.print("Pulling HuggingFace data from cloud...")
                pipeline.download_huggingface_data(category, 10, None)
            else:
                category_dir = pipeline.huggingface_dir / category
                if not category_dir.exists():
                    out.print("[bold red]Local HuggingFace data not found! Use --pull-huggingface to download.[/bold red]")
                    raise typer.Exit(1)
            if windowing_strategy == "price_observation":
                combined_df = pipeline.combine_datasets(
                    category, time_window_days, include_all_reviews, 
                    windowing_strategy, calendar_window_interval, review_window_size, debug
                )
            elif windowing_strategy == "calendar":
                combined_df = pipeline.combine_datasets(
                    category, time_window_days, include_all_reviews,
                    windowing_strategy, calendar_window_interval, review_window_size, debug
                )
            else:
                combined_df = pipeline.combine_datasets(
                    category, time_window_days, include_all_reviews,
                    windowing_strategy, calendar_window_interval, review_window_size, debug
                )
            config_metadata = {
                'data_source': 'keepa_huggingface_combined',
                'windowing_strategy': windowing_strategy,
                'time_window_days': time_window_days if windowing_strategy == "price_observation" else None,
                'calendar_window_interval': calendar_window_interval if windowing_strategy == "calendar" else None,
                'review_window_size': review_window_size if windowing_strategy == "review_frequency" else None,
                'include_all_reviews': include_all_reviews
            }
            output_paths = pipeline.save_organized_data(combined_df, category, config_metadata)
            out.print("\n[bold green]Merge completed![/bold green]")
            out.print("Output files:")
            for file_type, path in output_paths.items():
                out.print(f"  {file_type}: {path}")
        except Exception as e:
            if debug:
                out.print(f"[bold red]Merge failed: {e}[/bold red]")
                import traceback
                out.print(f"[bold red]Full traceback:[/bold red]\n{traceback.format_exc()}")
                raise typer.Exit(1)
            else:
                out.print(f"[bold red]Merge failed: {e}[/bold red]")
                raise typer.Exit(1)

@app.command()
def full_pipeline(
    category: str = typer.Option("All_Beauty", help="Amazon product category"),
    api_key_file: str = typer.Option("keepa_api_key.txt", help="Keepa API key file"),
    min_reviews: int = typer.Option(10, help="Minimum reviews per ASIN"),
    max_asins: Optional[int] = typer.Option(None, help="Maximum ASINs to process (None = all ASINs)"),
    windowing_strategy: str = typer.Option("price_observation", help="Windowing strategy: price_observation, calendar, or review_frequency"),
    time_window_days: int = typer.Option(30, help="Days around price observation (for price_observation strategy)"),
    calendar_window_days: int = typer.Option(1, help="Days per calendar window (for calendar strategy)"),
    review_window_size: int = typer.Option(10, help="Reviews per window (for review_frequency strategy)"),
    include_all_reviews: bool = typer.Option(True, help="Include all reviews in time window"),
    sample_reviews: Optional[int] = typer.Option(None, help="Sample size for testing"),
    work_dir: Path = typer.Option(".", help="Working directory"),
    slurm: bool = typer.Option(False, "--slurm", help="Flush log output for SLURM/plain mode")
):
    """Run the complete pipeline: extract ASINs, download data, and merge."""
    setup_logging(slurm)
    with SlurmOutput(slurm) as out:
        if windowing_strategy == "price_observation":
            strategy_desc = f"Price observation windows ({time_window_days} days around each price change)"
        elif windowing_strategy == "calendar":
            strategy_desc = f"Calendar windows ({calendar_window_days} day intervals)"
        else:
            strategy_desc = f"Review frequency windows (every {review_window_size} reviews)"
        max_asins_display = "All ASINs" if max_asins is None else str(max_asins)
        out.panel(
            f"[bold blue]Full Pipeline[/bold blue]\n"
            f"Category: {category}\n"
            f"Max ASINs: {max_asins_display}\n"
            f"Strategy: {strategy_desc}",
            border_style="blue"
        )
        pipeline = AmazonKeepaDataPipeline(work_dir)
        try:
            out.print("\n[bold cyan]Step 1: Downloading HuggingFace data and extracting ASINs[/bold cyan]")
            output_paths = pipeline.download_huggingface_data(category, min_reviews, max_asins, sample_reviews)
            asins_file = output_paths['asins']
            out.print("\n[bold cyan]Step 2: Keepa Download Required[/bold cyan]")
            out.print(f"Please run: python keepa.py download {api_key_file} {asins_file}")
            out.print("\n[bold cyan]Step 3: Merge Datasets[/bold cyan]")
            if windowing_strategy == "price_observation":
                merge_cmd = f"python dataset_builder.py merge-datasets --category {category} --windowing-strategy {windowing_strategy} --time-window-days {time_window_days}"
            elif windowing_strategy == "calendar":
                merge_cmd = f"python dataset_builder.py merge-datasets --category {category} --windowing-strategy {windowing_strategy} --calendar-window-days {calendar_window_days}"
            else:
                merge_cmd = f"python dataset_builder.py merge-datasets --category {category} --windowing-strategy {windowing_strategy} --review-window-size {review_window_size}"
            if include_all_reviews:
                merge_cmd += " --include-all-reviews"
            out.print(f"Then run: {merge_cmd}")
        except Exception as e:
            out.print(f"[bold red]Pipeline failed: {e}[/bold red]")
            raise typer.Exit(1)

@app.command()
def list_categories(
    work_dir: Path = typer.Option(".", help="Working directory"),
    show_info: bool = typer.Option(False, "--info", help="Show detailed information for each category"),
    slurm: bool = typer.Option(False, "--slurm", help="Flush log output for SLURM/plain mode")
):
    """List available Amazon product categories from all_categories.txt."""
    setup_logging(slurm)
    with SlurmOutput(slurm) as out:
        out.panel(
            "[bold blue]Loading Amazon Product Categories[/bold blue]",
            border_style="blue"
        )
        pipeline = AmazonKeepaDataPipeline(work_dir)
        try:
            categories = pipeline.get_available_categories()
            if not categories:
                out.print("[bold red]No categories found![/bold red]")
                raise typer.Exit(1)
            out.print(f"\n[bold green]Found {len(categories)} categories:[/bold green]")
            if show_info:
                for category in categories:
                    info = pipeline.get_category_info(category)
                    status = "[green]✓[/green]" if info['local_data_exists'] else "[red]✗[/red]"
                    out.print(f"\n{status} [bold]{category}[/bold]")
                    if info['local_data_exists']:
                        out.print(f"  Reviews: {info['reviews_count']:,}")
                        out.print(f"  Metadata: {info['metadata_count']:,}")
                        out.print(f"  Unique ASINs: {info['unique_asins']:,}")
                        if info['date_range']:
                            out.print(f"  Date Range: {info['date_range']['start'][:10]} to {info['date_range']['end'][:10]}")
                    else:
                        out.print("  [yellow]No local data[/yellow]")
            else:
                for i, category in enumerate(categories, 1):
                    out.print(f"  {i:2d}. {category}")
            out.print(f"\n[bold]Usage:[/bold]")
            out.print(f"  python dataset_builder.py download-huggingface --category <category_name>")
            out.print(f"  python dataset_builder.py merge-datasets --category <category_name>")
        except Exception as e:
            out.print(f"[bold red]Failed to load categories: {e}[/bold red]")
            raise typer.Exit(1)

@app.command()
def process_huggingface_only(
    category: str = typer.Option("All_Beauty", help="Amazon product category"),
    windowing_strategy: str = typer.Option("calendar", help="Windowing strategy: calendar or review_frequency"),
    calendar_window_interval: str = typer.Option("1mo", help="Time interval per calendar window (e.g., 1d, 2M, 1w) for calendar strategy"),
    review_window_size: int = typer.Option(10, help="Reviews per window (for review_frequency strategy)"),
    include_all_reviews: bool = typer.Option(True, help="Include all reviews in time window"),
    min_reviews_per_asin: int = typer.Option(10, help="Minimum reviews required per ASIN"),
    max_asins: Optional[int] = typer.Option(None, help="Maximum ASINs to process (None = all ASINs)"),
    work_dir: Path = typer.Option(".", help="Working directory"),
    sub_dir: Optional[str] = typer.Option(None, help="Subdirectory name for organizing different configurations"),
    pull_huggingface: bool = typer.Option(False, help="Pull HuggingFace data from cloud instead of using local"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with full tracebacks and stop on first error"),
    rolling_window_sizes: list[int] = typer.Option([3, 5, 10, 30], help="Rolling window sizes for additional statistics"),
    upsample: bool = typer.Option(False, "--upsample", help="Create empty buckets for missing time periods to maintain continuity for rolling statistics. Empty windows have review_count=0 (not counted as reviews). Example: if an ASIN has reviews in months 1-3 and 6-12, this creates empty observations for months 4-5."),
    asins_per_batch: int = typer.Option(100_000, help="Number of ASINs to process in each batch for memory efficiency (default: 100,000)"),
    slurm: bool = typer.Option(False, "--slurm", help="Flush log output for SLURM/plain mode")
):
    """
    Process HuggingFace review data without Keepa price data using high-performance batch processing.
    After processing, run finalize_batches to merge and clean up.
    """
    setup_logging(slurm)
    valid_strategies = ["calendar", "review_frequency"]
    with SlurmOutput(slurm) as out:
        if windowing_strategy not in valid_strategies:
            out.print(f"[bold red]Invalid windowing strategy: {windowing_strategy}[/bold red]")
            out.print(f"Valid options: {', '.join(valid_strategies)}")
            raise typer.Exit(1)
        if windowing_strategy == "calendar":
            strategy_desc = f"Calendar windows ({calendar_window_interval} intervals)"
        else:
            strategy_desc = f"Review frequency windows (every {review_window_size} reviews)"
        sub_dir_display = f"Subdirectory: {sub_dir}" if sub_dir else "Subdirectory: None (default location)"
        out.panel(
            f"[bold blue]Processing HuggingFace Data Only (High-Performance Batch Processing)[/bold blue]\n"
            f"Category: {category}\n"
            f"Strategy: {strategy_desc}\n"
            f"Include All Reviews: {include_all_reviews}\n"
            f"Min Reviews per ASIN: {min_reviews_per_asin}\n"
            f"Max ASINs: {max_asins if max_asins is not None else 'All'}\n"
            f"{sub_dir_display}",
            border_style="blue"
        )
        pipeline = AmazonKeepaDataPipeline(work_dir)
        try:
            if pull_huggingface:
                out.print("Pulling HuggingFace data from cloud...")
                pipeline.download_huggingface_data(category, min_reviews_per_asin, max_asins)
            else:
                category_dir = pipeline.huggingface_dir / category
                if not category_dir.exists():
                    out.print("[bold red]Local HuggingFace data not found! Use --pull-huggingface to download.[/bold red]")
                    raise typer.Exit(1)
            out.print("[bold cyan]Processing with high-performance batch processing, saving each batch to disk...[/bold cyan]")
            pipeline.process_huggingface_only(
                category, windowing_strategy, calendar_window_interval, 
                review_window_size, include_all_reviews, min_reviews_per_asin, max_asins, debug, rolling_window_sizes, upsample, asins_per_batch, slurm, sub_dir, out
            )
            out.print("[bold green]All batches processed and saved. Now merging...[/bold green]")
            pipeline.finalize_batches(category, sub_dir, out)
            out.print("[bold green]Processing and merge completed![/bold green]")
        except Exception as e:
            if debug:
                out.print(f"[bold red]Processing failed: {e}[/bold red]")
                import traceback
                out.print(f"[bold red]Full traceback:[/bold red]\n{traceback.format_exc()}")
                raise typer.Exit(1)
            else:
                out.print(f"[bold red]Processing failed: {e}[/bold red]")
                raise typer.Exit(1)

@app.command()
def list_configurations(
    category: str = typer.Option(None, help="Filter by specific category"),
    work_dir: Path = typer.Option(".", help="Working directory"),
    slurm: bool = typer.Option(False, "--slurm", help="Flush log output for SLURM/plain mode")
):
    """List available processed dataset configurations (subdirectories)."""
    setup_logging(slurm)
    with SlurmOutput(slurm) as out:
        out.panel(
            "[bold blue]Available Processed Dataset Configurations[/bold blue]",
            border_style="blue"
        )
        pipeline = AmazonKeepaDataPipeline(work_dir)
        try:
            processed_dir = pipeline.processed_datasets_dir / "huggingface_only"
            if not processed_dir.exists():
                out.print("[yellow]No processed datasets found.[/yellow]")
                return
            
            # Get all subdirectories
            subdirs = [d for d in processed_dir.iterdir() if d.is_dir()]
            if not subdirs:
                out.print("[yellow]No configuration subdirectories found.[/yellow]")
                return
            
            out.print(f"\n[bold green]Found {len(subdirs)} configuration(s):[/bold green]")
            
            for subdir in sorted(subdirs):
                subdir_name = subdir.name
                out.print(f"\n[bold cyan]{subdir_name}[/bold cyan]")
                
                # Look for summary files to get configuration info
                summary_files = list(subdir.glob("*_summary.json"))
                if summary_files:
                    for summary_file in summary_files:
                        try:
                            with open(summary_file, 'r') as f:
                                summary = json.load(f)
                            
                            cat_name = summary.get('category', 'Unknown')
                            if category and cat_name != category:
                                continue
                                
                            out.print(f"  Category: {cat_name}")
                            out.print(f"  Records: {summary.get('total_records', 0):,}")
                            out.print(f"  ASINs: {summary.get('unique_asins', 0):,}")
                            
                            config = summary.get('processing_config', {})
                            if config:
                                out.print(f"  Strategy: {config.get('windowing_strategy', 'Unknown')}")
                                if config.get('windowing_strategy') == 'calendar':
                                    out.print(f"  Interval: {config.get('calendar_window_interval', 'Unknown')}")
                                elif config.get('windowing_strategy') == 'review_frequency':
                                    out.print(f"  Window Size: {config.get('review_window_size', 'Unknown')}")
                                out.print(f"  Min Reviews: {config.get('min_reviews_per_asin', 'Unknown')}")
                                out.print(f"  Upsample: {config.get('upsample', False)}")
                            
                            if summary.get('date_range'):
                                date_range = summary['date_range']
                                out.print(f"  Date Range: {date_range['start'][:10]} to {date_range['end'][:10]}")
                                
                        except Exception as e:
                            out.print(f"  [red]Error reading summary: {e}[/red]")
                else:
                    # Just list the files in the subdirectory
                    files = list(subdir.glob("*.parquet"))
                    if files:
                        out.print(f"  Files: {len(files)} parquet files")
                        for file in files:
                            out.print(f"    {file.name}")
                    else:
                        out.print("  [yellow]No parquet files found[/yellow]")
            
            out.print(f"\n[bold]Usage:[/bold]")
            out.print(f"  Load dataset: pipeline.load_dataset_with_metadata('{category or 'CategoryName'}', sub_dir='{subdirs[0].name if subdirs else 'SubDirName'}')")
            
        except Exception as e:
            out.print(f"[bold red]Failed to list configurations: {e}[/bold red]")
            raise typer.Exit(1)

@app.command()
def category_info(
    category: str = typer.Argument(..., help="Category name to get information for"),
    work_dir: Path = typer.Option(".", help="Working directory"),
    slurm: bool = typer.Option(False, "--slurm", help="Flush log output for SLURM/plain mode")
):
    """Get detailed information about a specific category."""
    setup_logging(slurm)
    with SlurmOutput(slurm) as out:
        out.panel(
            f"[bold blue]Category Information[/bold blue]\n"
            f"Category: {category}",
            border_style="blue"
        )
        pipeline = AmazonKeepaDataPipeline(work_dir)
        try:
            info = pipeline.get_category_info(category)
            out.print(f"\n[bold]Category:[/bold] {info['category']}")
            out.print(f"[bold]Local Data:[/bold] {'✓ Available' if info['local_data_exists'] else '✗ Not Available'}")
            if info['local_data_exists']:
                out.print(f"[bold]Reviews:[/bold] {info['reviews_count']:,}")
                out.print(f"[bold]Metadata Records:[/bold] {info['metadata_count']:,}")
                out.print(f"[bold]Unique ASINs:[/bold] {info['unique_asins']:,}")
                if info['date_range']:
                    out.print(f"[bold]Date Range:[/bold] {info['date_range']['start'][:10]} to {info['date_range']['end'][:10]}")
                if info.get('download_date'):
                    out.print(f"[bold]Downloaded:[/bold] {info['download_date'][:10]}")
            out.print(f"\n[bold]Next Steps:[/bold]")
            if not info['local_data_exists']:
                out.print(f"  1. Download data and extract ASINs: python dataset_builder.py download-huggingface --category {category}")
            out.print(f"  2. Process HuggingFace only: python dataset_builder.py process-huggingface-only --category {category}")
            out.print(f"  3. Download Keepa data: python keepa.py download <api_key> <asins_file>")
            out.print(f"  4. Merge datasets: python dataset_builder.py merge-datasets --category {category}")
        except Exception as e:
            out.print(f"[bold red]Failed to get category info: {e}[/bold red]")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()