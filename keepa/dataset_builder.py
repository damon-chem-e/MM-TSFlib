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

try:
    from analyze_history import parse_price_history, load_history
except ImportError:
    print("Error: Cannot import analyze_history. Make sure analyze_history.py is in the same directory.")
    sys.exit(1)

# Set up rich logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer()

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
        self.output_dir = work_dir / "combined_dataset"
        self.huggingface_dir = work_dir / "huggingface_data"
        self.processed_datasets_dir = work_dir / "processed_datasets"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.huggingface_dir.mkdir(parents=True, exist_ok=True)
        self.processed_datasets_dir.mkdir(parents=True, exist_ok=True)
    

    
    def download_huggingface_data(self, category: str, min_reviews: int = 10, 
                                 max_asins: Optional[int] = None, sample_reviews: Optional[int] = None) -> Dict[str, Path]:
        """Download HuggingFace data, save locally, and extract ASINs in one operation."""
        logger.info(f"Downloading HuggingFace data for category: {category}")
        
        # Create category-specific directory
        category_dir = self.huggingface_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download reviews data with trust_remote_code=True to fix warning
            logger.info("Downloading reviews dataset...")
            reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", trust_remote_code=True)
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
                meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", trust_remote_code=True)
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
    
    def process_huggingface_only(self, category: str, 
                                windowing_strategy: str = "calendar",
                                calendar_window_interval: str = "1d",
                                review_window_size: int = 10,
                                include_all_reviews: bool = True,
                                min_reviews_per_asin: int = 10,
                                max_asins: Optional[int] = None,
                                debug: bool = False,
                                rolling_window_sizes: list[int] = [3, 5, 10, 30],
                                upsample: bool = False) -> pd.DataFrame:
        """
        Process HuggingFace review data without Keepa price data using pandas groupby approach.
        
        This method processes all ASINs together using a single groupby operation for consistency
        with the Polars method and better performance than individual ASIN processing.
        
        Note: Images and videos are ignored for now but could be included later
        when expanding to vision as another modality.
        """
        logger.info(f"Processing HuggingFace data for {category} using {windowing_strategy} strategy (pandas)")
        
        # Load data
        reviews_df, metadata_df = self.load_local_huggingface_data(category)
        
        if reviews_df.empty or metadata_df.empty:
            logger.warning(f"No data found for category {category}")
            return pd.DataFrame()
        
        # Filter ASINs with minimum reviews - use parent_asin for consistency
        asin_counts = reviews_df['parent_asin'].value_counts()
        valid_asins = asin_counts[asin_counts >= min_reviews_per_asin].index.tolist()
        
        # Limit to max_asins if specified (for debugging/testing)
        if max_asins is not None:
            valid_asins = valid_asins[:max_asins]
            logger.info(f"Limited to first {max_asins} ASINs for debugging/testing")
        
        reviews_df = reviews_df[reviews_df['parent_asin'].isin(valid_asins)]
        
        logger.info(f"Processing {len(valid_asins)} ASINs with at least {min_reviews_per_asin} reviews")
        
        if reviews_df.empty:
            logger.warning("No ASINs meet the minimum review requirement")
            return pd.DataFrame()
        
        # Merge with metadata - use parent_asin for joining
        if not metadata_df.empty and 'parent_asin' in metadata_df.columns:
            reviews_df = reviews_df.merge(metadata_df, on='parent_asin', how='left')
        else:
            logger.warning("No metadata available or missing parent_asin column, processing without metadata")
        
        # Rename columns 
        # title -> review_title, title_right -> product_name
        column_renames = {}
        if 'title_right' in reviews_df.columns:
            column_renames['title_right'] = 'product_name'
        if 'images_right' in reviews_df.columns:
            column_renames['images_right'] = 'product_images'
        if 'images' in reviews_df.columns:
            column_renames['images'] = 'review_images'
        if 'videos' in reviews_df.columns:
            column_renames['videos'] = 'product_videos'
        if 'title' in reviews_df.columns:
            column_renames['title'] = 'review_title'
        
        if column_renames:
            reviews_df = reviews_df.rename(columns=column_renames)
        
        # Create individual review text in specified format before aggregation
        if all(col in reviews_df.columns for col in ['review_title', 'rating', 'helpful_vote', 'text', 'verified_purchase']):
            reviews_df['formatted_review_text'] = (
                r"<\ begin review \> title: " +
                reviews_df['review_title'].fillna('').astype(str) +
                " || rating: " +
                reviews_df['rating'].astype(str) +
                " || helpful votes: " +
                reviews_df['helpful_vote'].fillna(0).astype(str) +
                " || content: " +
                reviews_df['text'].fillna('').astype(str) +
                " || verified purchase: " +
                reviews_df['verified_purchase'].fillna(False).astype(str) +
                r" <\ end review \> "
            )
        
        # Create window labels based on strategy
        # Convert polars interval to pandas frequency string
        def polars_to_pandas_freq(interval: str) -> str:
            """Convert polars interval to pandas frequency string."""
            interval = interval.lower()
            if interval.endswith('d'):
                return interval.upper()  # '1d' -> '1D'
            elif interval.endswith('m'):
                return interval.replace('m', 'min')  # '1m' -> '1min' (minutes)
            elif interval.endswith('mo'):
                return interval.replace('mo', 'M')  # '1mo' -> '1M' (months)
            elif interval.endswith('w'):
                return interval.upper()  # '1w' -> '1W'
            elif interval.endswith('h'):
                return interval.upper()  # '1h' -> '1H'
            elif interval.endswith('s'):
                return interval.upper()  # '1s' -> '1S'
            elif interval.endswith('y'):
                return interval.replace('y', 'Y')  # '1y' -> '1Y'
            elif interval.endswith('q'):
                return interval.replace('q', 'Q')  # '1q' -> '1Q'
            else:
                return interval.upper()  # fallback
        
        pandas_freq = polars_to_pandas_freq(calendar_window_interval)
        
        if windowing_strategy == "calendar":
            if upsample:
                # For upsampling, we need to create a full time range and join
                logger.info(f"Creating upsampled time series with {calendar_window_interval} windows")
                # Get the full date range for all ASINs
                min_date = reviews_df['datetime'].min()
                max_date = reviews_df['datetime'].max()
                
                # Create full time range
                time_range = pd.date_range(start=min_date, end=max_date, freq=pandas_freq)
                
                # Create skeleton DataFrame with all ASINs and time periods
                skeleton_data = []
                for asin in reviews_df['parent_asin'].unique():
                    for timestamp in time_range:
                        skeleton_data.append({
                            'parent_asin': asin,
                            'window_start': timestamp
                        })
                skeleton_df = pd.DataFrame(skeleton_data)
                
                # Add window_start to original data
                reviews_df['window_start'] = reviews_df['datetime'].dt.floor(pandas_freq)
                
                # Join skeleton with original data to create full time series
                reviews_df = skeleton_df.merge(
                    reviews_df, 
                    on=['parent_asin', 'window_start'], 
                    how='left'
                )
                
                logger.info(f"Upsampling created {len(reviews_df)} records (including empty windows)")
                group_cols = ['parent_asin', 'window_start']
            else:
                # Only process existing data buckets (no upsampling)
                reviews_df['window_start'] = reviews_df['datetime'].dt.floor(pandas_freq)
                group_cols = ['parent_asin', 'window_start']
        elif windowing_strategy == "review_frequency":
            # Create review frequency windows
            reviews_df = reviews_df.sort_values(['parent_asin', 'datetime'])
            reviews_df['window_id'] = reviews_df.groupby('parent_asin').cumcount() // review_window_size
            group_cols = ['parent_asin', 'window_id']
        else:
            raise ValueError(f"Unsupported windowing strategy: {windowing_strategy}")
        
        # Define aggregations
        agg_dict = {
            'rating': 'mean',
            'parent_asin': 'count',  # This will count reviews per window
        }
        
        # Add weighted averages for helpful_vote and verified_purchase
        def helpful_vote_weighted_avg(group):
            if 'helpful_vote' in group.columns and 'rating' in group.columns:
                helpful_votes = group['helpful_vote'].fillna(0)
                ratings = group['rating']
                if helpful_votes.sum() > 0:
                    return (helpful_votes * ratings).sum() / helpful_votes.sum()
            return None
            
        def verified_purchase_weighted_avg(group):
            if 'verified_purchase' in group.columns and 'rating' in group.columns:
                verified = group['verified_purchase'].fillna(False).astype(int)
                ratings = group['rating']
                if verified.sum() > 0:
                    return (verified * ratings).sum() / verified.sum()
            return None
        
        # Add metadata columns to aggregation - use correct column names
        # Check if columns exist before adding them to avoid errors
        metadata_cols = ['product_name', 'main_category', 'categories', 'brand', 'store', 'price', 
                        'description', 'features', 'average_rating', 'window_rating_count']
        
        for col in metadata_cols:
            if col in reviews_df.columns:
                agg_dict[col] = 'first'
        
        # Handle formatted review text aggregation
        if 'formatted_review_text' in reviews_df.columns:
            agg_dict['formatted_review_text'] = lambda x: ''.join(x.dropna().astype(str))
        
        # Perform groupby aggregation
        grouped = reviews_df.groupby(group_cols)
        aggregated = grouped.agg(agg_dict)
        
        # Add weighted averages using custom functions
        aggregated['helpful_vote_weighted_avg_rating'] = grouped.apply(helpful_vote_weighted_avg)
        aggregated['verified_purchase_weighted_avg_rating'] = grouped.apply(verified_purchase_weighted_avg)
        
        # Rename columns
        aggregated = aggregated.rename(columns={
            'rating': 'avg_rating_window',
            'parent_asin': 'review_count_window',
            'formatted_review_text': 'aggregated_reviews'
        })
        
        # Handle list columns by converting to strings
        list_columns = ['categories', 'features', 'description']
        for col in list_columns:
            if col in aggregated.columns:
                # Convert list columns to string by joining with ', '
                aggregated[f'{col}_str'] = aggregated[col].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x) if x is not None else ''
                )
            else:
                aggregated[f'{col}_str'] = ''
        
        # Create the final combined text column in the specified format
        aggregated['combined_text'] = (
            "product name: " + aggregated.get('product_name', '').fillna('').astype(str) +
            "\nproduct categories: " + aggregated.get('categories_str', '').fillna('').astype(str) +
            "\n2023 price: " + aggregated.get('price', '').fillna('').astype(str) +
            "\nbrand: " + aggregated.get('brand', '').fillna('').astype(str) +
            "\nseller: " + aggregated.get('store', '').fillna('').astype(str) +
            "\nproduct description: " + aggregated.get('description_str', '').fillna('').astype(str) +
            "\nreviews: " + aggregated.get('aggregated_reviews', '').fillna('').astype(str) +
            "\nproduct features: " + aggregated.get('features_str', '').fillna('').astype(str)
        )
        
        # Add date columns
        if windowing_strategy == "calendar":
            aggregated['timestamp'] = aggregated.index.get_level_values('window_start')
            aggregated['date'] = aggregated['timestamp'].dt.date
            aggregated['year'] = aggregated['timestamp'].dt.year
            aggregated['month'] = aggregated['timestamp'].dt.month
        else:
            # For review frequency, get the first datetime in each window
            first_dates = grouped['datetime'].first()
            aggregated['timestamp'] = first_dates
            aggregated['date'] = aggregated['timestamp'].dt.date
            aggregated['year'] = aggregated['timestamp'].dt.year
            aggregated['month'] = aggregated['timestamp'].dt.month
        
        # Add avg_review_count_per_interval for calendar windows
        if windowing_strategy == "calendar":
            aggregated['avg_review_count_per_interval'] = aggregated['review_count_window']
        
        # Reset index and add ASIN column
        aggregated = aggregated.reset_index()
        aggregated['asin'] = aggregated['parent_asin']
        aggregated = aggregated.drop(columns=['parent_asin'])
        
        # Add rolling statistics first (we need review_count_window for calculations)
        # Note: We use both types of rolling statistics for different purposes:
        # 1. Time-based rolling (7d, 30d, 90d) - for calendar windows only
        # 2. Window-count-based rolling (3w, 5w, 10w, 30w) - for both calendar and review frequency windows
        if windowing_strategy == "calendar" and not aggregated.empty:
            try:
                aggregated = self._add_rolling_statistics(aggregated)
            except Exception as e:
                if debug:
                    logger.error(f"Error adding rolling statistics: {e}")
                    import traceback
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    raise  # Re-raise the exception to stop processing
                else:
                    logger.warning(f"Error adding rolling statistics: {e}")
                    logger.info("Continuing without rolling statistics")
        
        aggregated = self._add_rolling_window_statistics(aggregated, rolling_window_sizes)
        
        # Select only the columns we want to keep 
        # Keep base columns and all rolling statistics
        base_columns = ['asin', 'timestamp', 'date', 'year', 'month', 'combined_text', 'review_count_window', 'avg_review_count_per_interval',
                       'avg_rating_window', 'helpful_vote_weighted_avg_rating', 'verified_purchase_weighted_avg_rating']
        
        # Add all rolling columns
        rolling_columns = [col for col in aggregated.columns if col.startswith('rolling_')]
        columns_to_keep = base_columns + rolling_columns
        
        # Keep only the columns we want
        available_columns = [col for col in columns_to_keep if col in aggregated.columns]
        aggregated = aggregated[available_columns]
        logger.info(f"Generated {len(aggregated)} records from {len(valid_asins)} ASINs (pandas)")
        return aggregated
    
    def process_huggingface_only_polars(self, category: str, 
                                       windowing_strategy: str = "calendar",
                                       calendar_window_interval: str = "1d",
                                       review_window_size: int = 10,
                                       include_all_reviews: bool = True,
                                       min_reviews_per_asin: int = 10,
                                       max_asins: Optional[int] = None,
                                       debug: bool = False,
                                       rolling_window_sizes: list[int] = [3, 5, 10, 30],
                                       upsample: bool = False) -> pd.DataFrame:
        """
        Polars-based implementation of process_huggingface_only for better performance.
        
        This method uses Polars for faster data processing, especially beneficial
        for large datasets with millions of reviews.
        
        Note: Images and videos are ignored for now but could be included later
        when expanding to vision as another modality.
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Polars is required for this method. Install with: pip install polars")
        
        logger.info(f"Processing HuggingFace data for {category} using {windowing_strategy} strategy (Polars)")
        
        # Load data
        reviews_df, metadata_df = self.load_local_huggingface_data(category)
        
        if reviews_df.empty or metadata_df.empty:
            logger.warning(f"No data found for category {category}")
            return pd.DataFrame()
        
        # Convert to Polars DataFrames
        reviews_pl = pl.from_pandas(reviews_df)
        metadata_pl = pl.from_pandas(metadata_df)
        
        # Filter ASINs with minimum reviews - use parent_asin for consistency with other methods
        asin_counts = reviews_pl.group_by('parent_asin').len()
        valid_asins_df = asin_counts.filter(pl.col('len') >= min_reviews_per_asin)
        valid_asins = valid_asins_df.select('parent_asin').to_series().to_list()
        
        # Limit to max_asins if specified (for debugging/testing)
        if max_asins is not None:
            valid_asins = valid_asins[:max_asins]
            logger.info(f"Limited to first {max_asins} ASINs for debugging/testing")
        
        # Filter to only the valid ASINs
        valid_asins_df = valid_asins_df.filter(pl.col('parent_asin').is_in(valid_asins))
        reviews_pl = reviews_pl.join(valid_asins_df.select('parent_asin'), on='parent_asin', how='inner')
        
        logger.info(f"Processing {len(valid_asins)} ASINs with at least {min_reviews_per_asin} reviews")
        
        if reviews_pl.is_empty():
            logger.warning("No ASINs meet the minimum review requirement")
            return pd.DataFrame()
        
        # Merge with metadata - use parent_asin for joining
        if not metadata_pl.is_empty():
            # Ensure metadata has the correct column names for joining
            if 'parent_asin' in metadata_pl.columns:
                reviews_pl = reviews_pl.join(metadata_pl, on='parent_asin', how='left')
            else:
                logger.warning("Metadata does not have parent_asin column, skipping metadata join")
        else:
            logger.warning("No metadata available, processing without metadata")
        
        # Rename columns as specified by user
        # title -> review_title, title_right -> product_name
        column_renames = {}
        if 'title_right' in reviews_pl.columns:
            column_renames['title_right'] = 'product_name'
        if 'images_right' in reviews_pl.columns:
            column_renames['images_right'] = 'product_images'
        if 'images' in reviews_pl.columns:
            column_renames['images'] = 'review_images'
        if 'videos' in reviews_pl.columns:
            column_renames['videos'] = 'product_videos'
        if 'title' in reviews_pl.columns:
            column_renames['title'] = 'review_title'
        
        if column_renames:
            reviews_pl = reviews_pl.rename(column_renames)
        
        # Create individual review text in specified format before aggregation
        review_text_cols = []
        for col in ['review_title', 'rating', 'helpful_vote', 'text', 'verified_purchase']:
            if col in reviews_pl.columns:
                review_text_cols.append(col)
        
        if review_text_cols:
            # Create formatted review text for each review
            reviews_pl = reviews_pl.with_columns([
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
        
        # Create window labels based on strategy
        if windowing_strategy == "calendar":
            if upsample:
                # For upsampling, we need to create a full time range and join
                # Get the full date range for all ASINs
                min_date = reviews_pl.select(pl.col('datetime').min()).item()
                max_date = reviews_pl.select(pl.col('datetime').max()).item()
                
                # Create full time range
                time_range = pl.datetime_range(
                    start=min_date,
                    end=max_date,
                    interval=calendar_window_interval
                )
                
                # Create skeleton DataFrame with all ASINs and time periods
                skeleton = pl.DataFrame({
                    'window_start': time_range
                }).join(
                    reviews_pl.select('parent_asin').unique(),
                    how='cross'
                )
                
                # Join with original data
                reviews_pl = skeleton.join(
                    reviews_pl.with_columns([
                        pl.col('datetime').dt.truncate(calendar_window_interval).alias('window_start')
                    ]),
                    on=['parent_asin', 'window_start'],
                    how='left'
                )
            else:
                # Only process existing data buckets (no upsampling)
                reviews_pl = reviews_pl.with_columns([
                    pl.col('datetime').dt.truncate(calendar_window_interval).alias('window_start')
                ])
            
            group_cols = ['parent_asin', 'window_start']
        elif windowing_strategy == "review_frequency":
            # Create review frequency windows
            reviews_pl = reviews_pl.sort(['parent_asin', 'datetime'])
            reviews_pl = reviews_pl.with_columns([
                pl.int_range(pl.len()).over('parent_asin').alias('row_nr')
            ])
            reviews_pl = reviews_pl.with_columns([
                (pl.col('row_nr') // review_window_size).alias('window_id')
            ])
            group_cols = ['parent_asin', 'window_id']
        else:
            raise ValueError(f"Unsupported windowing strategy: {windowing_strategy}")
        
        # Define aggregations
        agg_exprs = [
            pl.col('rating').mean().alias('avg_rating_window'),
            pl.col('parent_asin').len().alias('review_count_window'),
        ]
        
        # Add weighted averages for helpful_vote and verified_purchase
        if 'helpful_vote' in reviews_pl.columns and 'rating' in reviews_pl.columns:
            # Helpful vote weighted average rating (only considering reviews with helpful votes > 0)
            agg_exprs.append(
                pl.when(pl.col('helpful_vote').sum() > 0)
                .then((pl.col('helpful_vote') * pl.col('rating')).sum() / pl.col('helpful_vote').sum())
                .otherwise(None)
                .alias('helpful_vote_weighted_avg_rating')
            )
        else:
            agg_exprs.append(pl.lit(None).alias('helpful_vote_weighted_avg_rating'))
            
        if 'verified_purchase' in reviews_pl.columns and 'rating' in reviews_pl.columns:
            # Verified purchase weighted average rating (only considering verified purchases)
            agg_exprs.append(
                pl.when(pl.col('verified_purchase').sum() > 0)
                .then((pl.col('verified_purchase').cast(pl.Float64) * pl.col('rating')).sum() / pl.col('verified_purchase').sum())
                .otherwise(None)
                .alias('verified_purchase_weighted_avg_rating')
            )
        else:
            agg_exprs.append(pl.lit(None).alias('verified_purchase_weighted_avg_rating'))
        
        # Add metadata columns - use correct column names from metadata
        # Check if columns exist before adding them to avoid errors
        if 'product_name' in reviews_pl.columns:
            agg_exprs.append(pl.col('product_name').first().alias('product_name'))
        else:
            agg_exprs.append(pl.lit('').alias('product_name'))
            
        if 'main_category' in reviews_pl.columns:
            agg_exprs.append(pl.col('main_category').first().alias('main_category'))
        else:
            agg_exprs.append(pl.lit(category).alias('main_category'))
            
        if 'categories' in reviews_pl.columns:
            agg_exprs.append(pl.col('categories').first().alias('categories'))
        else:
            agg_exprs.append(pl.lit('').alias('categories'))
            
        if 'brand' in reviews_pl.columns:
            agg_exprs.append(pl.col('brand').first().alias('brand'))
        else:
            agg_exprs.append(pl.lit('').alias('brand'))
            
        if 'store' in reviews_pl.columns:
            agg_exprs.append(pl.col('store').first().alias('store'))
        else:
            agg_exprs.append(pl.lit('').alias('store'))
            
        if 'price' in reviews_pl.columns:
            agg_exprs.append(pl.col('price').first().alias('price'))
        else:
            agg_exprs.append(pl.lit('').alias('price'))
            
        if 'description' in reviews_pl.columns:
            agg_exprs.append(pl.col('description').first().alias('description'))
        else:
            agg_exprs.append(pl.lit('').alias('description'))
            
        if 'features' in reviews_pl.columns:
            agg_exprs.append(pl.col('features').first().alias('features'))
        else:
            agg_exprs.append(pl.lit('').alias('features'))
            
        if 'average_rating' in reviews_pl.columns:
            agg_exprs.append(pl.col('average_rating').first().alias('avg_rating_metadata'))
        else:
            agg_exprs.append(pl.lit(None).alias('avg_rating_metadata'))

        
        # Handle review text aggregation - combine all formatted review texts
        if 'formatted_review_text' in reviews_pl.columns:
            agg_exprs.append(
                pl.col('formatted_review_text').drop_nulls().str.join('').alias('aggregated_reviews')
            )
        else:
            agg_exprs.append(pl.lit('').alias('aggregated_reviews'))
        
        # Add timestamp column - ensure we get a single value, not a list
        if windowing_strategy == "calendar":
            agg_exprs.append(pl.col('window_start').first().alias('timestamp'))
        else:
            agg_exprs.append(pl.col('datetime').first().alias('timestamp'))
        
        # Perform groupby aggregation
        aggregated = reviews_pl.group_by(group_cols).agg(agg_exprs)
        
        # Add ASIN column (rename parent_asin to asin for consistency)
        aggregated = aggregated.with_columns([
            pl.col('parent_asin').alias('asin')
        ])
        
        # Create the final combined text column in the specified format

        # Convert any list[str] columns to string by joining with ', ' before concatenation
        aggregated = aggregated.with_columns([
            pl.col('categories').list.join(', ').alias('categories_str'),
            pl.col('features').list.join(', ').alias('features_str'),
            pl.col('description').list.join(', ').alias('description_str'),
        ])

        # Now build the combined_text using the stringified columns
        aggregated = aggregated.with_columns([
            pl.concat_str([
                pl.lit("product name: "),
                pl.col('product_name').fill_null(''),
                pl.lit("\nproduct categories: "),
                pl.col('categories_str').fill_null(''),
                pl.lit("\n2023 price: "),
                pl.col('price').fill_null(''),
                pl.lit("\nbrand: "),
                pl.col('brand').fill_null(''),
                pl.lit("\nseller: "),
                pl.col('store').fill_null(''),
                pl.lit("\nproduct description: "),
                pl.col('description_str').fill_null(''),
                pl.lit("\nreviews: "),
                pl.col('aggregated_reviews').fill_null(''),
                pl.lit("\nproduct features: "),
                pl.col('features_str').fill_null('')
            ]).alias('combined_text')
        ])
        
        # Add date columns
        aggregated = aggregated.with_columns([
            pl.col('timestamp').dt.date().alias('date'),
            pl.col('timestamp').dt.year().alias('year'),
            pl.col('timestamp').dt.month().alias('month')
        ])
        
        # Add avg_review_count_per_interval for calendar windows
        if windowing_strategy == "calendar":
            aggregated = aggregated.with_columns([
                pl.col('review_count_window').alias('avg_review_count_per_interval')
            ])
        
        # Convert back to pandas for consistency
        result_df = aggregated.to_pandas()
        
        # Add rolling statistics first (we need review_count_window for calculations)
        # Note: We use both types of rolling statistics for different purposes:
        # 1. Time-based rolling (7d, 30d, 90d) - for calendar windows only
        # 2. Window-count-based rolling (3w, 5w, 10w, 30w) - for both calendar and review frequency windows
        if windowing_strategy == "calendar" and not result_df.empty:
            try:
                result_df = self._add_rolling_statistics_polars(result_df, calendar_window_interval)
            except Exception as e:
                if debug:
                    logger.error(f"Error adding rolling statistics: {e}")
                    import traceback
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    raise  # Re-raise the exception to stop processing
                else:
                    logger.warning(f"Error adding rolling statistics: {e}")
                    logger.info("Continuing without rolling statistics")
        
        result_df = self._add_rolling_window_statistics(result_df, rolling_window_sizes)
        
        # Select only the columns we want to keep
        # Keep base columns and all rolling statistics
        base_columns = ['asin', 'timestamp', 'date', 'year', 'month', 'combined_text', 'review_count_window', 'avg_review_count_per_interval',
                       'avg_rating_window', 'helpful_vote_weighted_avg_rating', 'verified_purchase_weighted_avg_rating']
        
        # Add all rolling columns
        rolling_columns = [col for col in result_df.columns if col.startswith('rolling_')]
        columns_to_keep = base_columns + rolling_columns
        
        # Keep only the columns we want
        available_columns = [col for col in columns_to_keep if col in result_df.columns]
        result_df = result_df[available_columns]
        logger.info(f"Generated {len(result_df)} records from {len(valid_asins)} ASINs (Polars)")
        return result_df
    
    def _process_calendar_windows_huggingface_only(self, asin: str, asin_reviews: pd.DataFrame,
                                                  product_title: str, product_category: str,
                                                  product_brand: str, product_price: str,
                                                  avg_rating: float, rating_count: int,
                                                  calendar_window_days: int, include_all_reviews: bool, upsample: bool = False) -> pd.DataFrame:
        """Process calendar windows for HuggingFace-only data using groupby aggregations."""
        if asin_reviews.empty:
            return pd.DataFrame()
        
        try:
            # Use pandas resample for standard day frequencies
            asin_reviews_copy = asin_reviews.copy()
            asin_reviews_copy = asin_reviews_copy.set_index('datetime')
            
            # Resample to calendar windows
            if upsample:
                # Create full time range and resample (includes empty buckets)
                resampled = asin_reviews_copy.resample(f'{calendar_window_days}D')
            else:
                # Only process existing data buckets (no upsampling)
                # Group by truncated datetime to avoid creating empty buckets
                asin_reviews_copy['window_start'] = asin_reviews_copy.index.floor(f'{calendar_window_days}D')
                grouped = asin_reviews_copy.groupby('window_start')
                resampled = grouped
            
            # Define aggregations
            agg_dict = {
                'rating': 'mean',
                'asin': 'count',
            }
            
            if 'helpful_votes' in asin_reviews_copy.columns:
                agg_dict['helpful_votes'] = 'sum'
            if 'verified_purchase' in asin_reviews_copy.columns:
                agg_dict['verified_purchase'] = 'mean'
            
            # Perform aggregations
            aggregated = resampled.agg(agg_dict)
            
            # Rename columns
            aggregated = aggregated.rename(columns={
                'rating': 'avg_rating_window',
                'asin': 'review_count_window',
                'helpful_votes': 'helpful_votes_sum',
                'verified_purchase': 'verified_purchases_ratio'
            })
            
            # Handle review text
            if include_all_reviews:
                text_agg = resampled['text'].apply(lambda x: ' ||| '.join(x.dropna().astype(str)) if not x.empty else '')
                aggregated['review_text'] = text_agg.str[:5000]
            else:
                text_agg = resampled['text'].first()
                aggregated['review_text'] = text_agg.str[:500] if text_agg is not None else ''
            
            # Add metadata columns
            aggregated['asin'] = asin
            aggregated['title'] = product_title
            aggregated['category'] = product_category
            aggregated['brand'] = product_brand
            aggregated['price_metadata'] = product_price
            aggregated['avg_rating_metadata'] = avg_rating
            aggregated['rating_count_metadata'] = rating_count
            aggregated['window_size'] = calendar_window_days
            aggregated['windowing_strategy'] = 'calendar'
            aggregated['include_all_reviews'] = include_all_reviews
            
            # Reset index to make datetime a column
            if upsample:
                aggregated = aggregated.reset_index().rename(columns={'datetime': 'timestamp'})
            else:
                aggregated = aggregated.reset_index().rename(columns={'window_start': 'timestamp'})
            
            # Add date columns from timestamp
            aggregated['date'] = aggregated['timestamp'].dt.date
            aggregated['year'] = aggregated['timestamp'].dt.year
            aggregated['month'] = aggregated['timestamp'].dt.month
            
            # Fill NaN values
            aggregated = aggregated.fillna({
                'avg_rating_window': pd.NA,
                'helpful_votes_sum': 0,
                'verified_purchases_ratio': 0.0,
                'review_text': ''
            })
            
            return aggregated
            
        except ValueError:
            # Fallback for custom day counts that pandas doesn't support directly
            # Create date bins and use groupby
            start_date = asin_reviews['datetime'].min()
            end_date = asin_reviews['datetime'].max()
            
            # Create date range with the specified frequency
            date_range = pd.date_range(start=start_date, end=end_date, freq=f'{calendar_window_days}D')
            
            # Create bins for grouping
            asin_reviews_copy = asin_reviews.copy()
            asin_reviews_copy['window_start'] = pd.cut(
                asin_reviews_copy['datetime'], 
                bins=date_range, 
                labels=date_range[:-1],
                include_lowest=True
            )
            
            # Group by window and aggregate
            grouped = asin_reviews_copy.groupby('window_start')
            
            # Define aggregations
            agg_dict = {
                'rating': 'mean',
                'asin': 'count',
            }
            
            if 'helpful_votes' in asin_reviews_copy.columns:
                agg_dict['helpful_votes'] = 'sum'
            if 'verified_purchase' in asin_reviews_copy.columns:
                agg_dict['verified_purchase'] = 'mean'
            
            # Perform aggregations
            aggregated = grouped.agg(agg_dict)
            
            # Rename columns
            aggregated = aggregated.rename(columns={
                'rating': 'avg_rating_window',
                'asin': 'review_count_window',
                'helpful_votes': 'helpful_votes_sum',
                'verified_purchase': 'verified_purchases_ratio'
            })
            
            # Handle review text
            if include_all_reviews:
                text_agg = grouped['text'].apply(lambda x: ' ||| '.join(x.dropna().astype(str)) if not x.empty else '')
                aggregated['review_text'] = text_agg.str[:5000]
            else:
                text_agg = grouped['text'].first()
                aggregated['review_text'] = text_agg.str[:500] if text_agg is not None else ''
            
            # Add metadata columns
            aggregated['asin'] = asin
            aggregated['title'] = product_title
            aggregated['category'] = product_category
            aggregated['brand'] = product_brand
            aggregated['price_metadata'] = product_price
            aggregated['avg_rating_metadata'] = avg_rating
            aggregated['rating_count_metadata'] = rating_count
            aggregated['window_size'] = calendar_window_days
            aggregated['windowing_strategy'] = 'calendar'
            aggregated['include_all_reviews'] = include_all_reviews
            
            # Add date columns - fix for CategoricalIndex
            aggregated['date'] = aggregated.index.astype('datetime64[ns]').date
            aggregated['year'] = aggregated.index.astype('datetime64[ns]').year
            aggregated['month'] = aggregated.index.astype('datetime64[ns]').month
            
            # Reset index
            aggregated = aggregated.reset_index().rename(columns={'window_start': 'timestamp'})
            
            # Fill NaN values
            aggregated = aggregated.fillna({
                'avg_rating_window': pd.NA,
                'helpful_votes_sum': 0,
                'verified_purchases_ratio': 0.0,
                'review_text': ''
            })
            
            return aggregated
    
    def _process_review_frequency_windows_huggingface_only(self, asin: str, asin_reviews: pd.DataFrame,
                                                          product_title: str, product_category: str,
                                                          product_brand: str, product_price: str,
                                                          avg_rating: float, rating_count: int,
                                                          review_window_size: int, include_all_reviews: bool) -> pd.DataFrame:
        """Process review frequency windows for HuggingFace-only data using groupby aggregations."""
        if asin_reviews.empty:
            return pd.DataFrame()
        
        # Create window labels for grouping
        asin_reviews_copy = asin_reviews.copy()
        asin_reviews_copy['window_id'] = (asin_reviews_copy.index // review_window_size).astype(int)
        
        # Group by window and aggregate
        grouped = asin_reviews_copy.groupby('window_id')
        
        # Define aggregations
        agg_dict = {
            'rating': 'mean',  # Average rating per window
            'asin': 'count',   # Review count per window
            'datetime': 'first'  # Timestamp of first review in window
        }
        
        # Add optional columns if they exist
        if 'helpful_votes' in asin_reviews_copy.columns:
            agg_dict['helpful_votes'] = 'sum'
        if 'verified_purchase' in asin_reviews_copy.columns:
            agg_dict['verified_purchase'] = 'mean'
        
        # Perform aggregations
        aggregated = grouped.agg(agg_dict)
        
        # Rename columns to match expected output
        aggregated = aggregated.rename(columns={
            'rating': 'avg_rating_window',
            'asin': 'review_count_window',
            'datetime': 'timestamp',
            'helpful_votes': 'helpful_votes_sum',
            'verified_purchase': 'verified_purchases_ratio'
        })
        
        # Handle review text aggregation
        if include_all_reviews:
            # For all reviews, concatenate all text in each window
            text_agg = grouped['text'].apply(lambda x: ' ||| '.join(x.dropna().astype(str)) if not x.empty else '')
            aggregated['review_text'] = text_agg.str[:5000]  # Limit to 5000 chars
        else:
            # For single review, take first review text in each window
            text_agg = grouped['text'].first()
            aggregated['review_text'] = text_agg.str[:500] if text_agg is not None else ''  # Limit to 500 chars
        
        # Add metadata columns
        aggregated['asin'] = asin
        aggregated['title'] = product_title
        aggregated['category'] = product_category
        aggregated['brand'] = product_brand
        aggregated['price_metadata'] = product_price
        aggregated['avg_rating_metadata'] = avg_rating
        aggregated['rating_count_metadata'] = rating_count
        aggregated['window_size'] = review_window_size
        aggregated['windowing_strategy'] = 'review_frequency'
        aggregated['include_all_reviews'] = include_all_reviews
        
        # Add date columns
        aggregated['date'] = aggregated['timestamp'].dt.date
        aggregated['year'] = aggregated['timestamp'].dt.year
        aggregated['month'] = aggregated['timestamp'].dt.month
        
        # Reset index to remove window_id
        aggregated = aggregated.reset_index(drop=True)
        
        # Fill NaN values
        aggregated = aggregated.fillna({
            'avg_rating_window': pd.NA,
            'helpful_votes_sum': 0,
            'verified_purchases_ratio': 0.0,
            'review_text': ''
        })
        
        return aggregated

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
        
        # Add rolling statistics for calendar windows
        if windowing_strategy == "calendar" and not combined_df.empty:
            try:
                combined_df = self._add_rolling_statistics(combined_df)
            except Exception as e:
                if debug:
                    logger.error(f"Error adding rolling statistics: {e}")
                    import traceback
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    raise  # Re-raise the exception to stop processing
                else:
                    logger.warning(f"Error adding rolling statistics: {e}")
                    logger.info("Continuing without rolling statistics")
        
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
            helpful_votes = window_reviews['helpful_vote'].sum()
            verified_ratio = window_reviews['verified_purchase'].mean()
            
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
            helpful_votes = 0
            verified_ratio = 0.0
            sample_review = ''
        
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
            'helpful_votes_sum': helpful_votes,
            'verified_purchases_ratio': verified_ratio,
            'review_text': sample_review[:5000] if include_all_reviews else sample_review[:500],
            'window_size': window_size,
            'windowing_strategy': windowing_strategy,
            'include_all_reviews': include_all_reviews
        }
        
        return record
    
    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics for calendar windows (legacy method for backward compatibility)."""
        logger.info("Adding rolling statistics for calendar windows")
        
        # Sort by ASIN and timestamp
        df = df.sort_values(['asin', 'timestamp']).copy()
        
        # Calculate rolling statistics for each ASIN
        rolling_stats = []
        
        for asin in df['asin'].unique():
            asin_data = df[df['asin'] == asin].copy()
            
            # Rolling window sizes (in days)
            windows = [7, 30, 90]  # 1 week, 1 month, 3 months
            
            for window_days in windows:
                # Calculate rolling averages for ratings
                asin_data[f'rolling_avg_rating_{window_days}d'] = asin_data['avg_rating_window'].rolling(
                    window=window_days, min_periods=1
                ).mean()
                
                # Calculate rolling review counts (total reviews over rolling window)
                if 'review_count_window' in asin_data.columns:
                    asin_data[f'rolling_review_count_{window_days}d'] = asin_data['review_count_window'].rolling(
                        window=window_days, min_periods=1
                    ).sum()
                    
                    # Calculate rolling average reviews per day (total reviews / window days)
                    asin_data[f'rolling_avg_reviews_per_day_{window_days}d'] = (
                        asin_data['review_count_window'].rolling(window=window_days, min_periods=1).sum() / window_days
                    )
                
                # Calculate rolling helpful votes (only if column exists)
                if 'helpful_votes_sum' in asin_data.columns:
                    asin_data[f'rolling_helpful_votes_{window_days}d'] = asin_data['helpful_votes_sum'].rolling(
                        window=window_days, min_periods=1
                    ).sum()
                
                # Calculate rolling verified purchase ratio (only if column exists)
                if 'verified_purchases_ratio' in asin_data.columns:
                    # This is a weighted average based on review counts
                    asin_data[f'rolling_verified_ratio_{window_days}d'] = (
                        (asin_data['verified_purchases_ratio'] * asin_data['review_count_window']).rolling(
                            window=window_days, min_periods=1
                        ).sum() / asin_data['review_count_window'].rolling(
                            window=window_days, min_periods=1
                        ).sum()
                    ).fillna(0)
            
            rolling_stats.append(asin_data)
        
        # Combine all ASINs back together
        result_df = pd.concat(rolling_stats, ignore_index=True)
        
        logger.info(f"Added rolling statistics with windows: 7, 30, 90 days")
        return result_df

    def _add_rolling_statistics_polars(self, df: pd.DataFrame, calendar_window_interval: str) -> pd.DataFrame:
        """Add rolling statistics for calendar windows in Polars implementation."""
        logger.info("Adding rolling statistics for calendar windows (Polars)")
        
        # Sort by ASIN and timestamp
        df = df.sort_values(['asin', 'timestamp']).copy()
        
        # Calculate rolling statistics for each ASIN
        rolling_stats = []
        
        for asin in df['asin'].unique():
            asin_data = df[df['asin'] == asin].copy()
            
            # Rolling window sizes (in days)
            windows = [7, 30, 90]  # 1 week, 1 month, 3 months
            
            for window_days in windows:
                # Calculate rolling averages for ratings
                asin_data[f'rolling_avg_rating_{window_days}d'] = asin_data['avg_rating_window'].rolling(
                    window=window_days, min_periods=1
                ).mean()
                
                # Calculate rolling review counts (total reviews over rolling window)
                asin_data[f'rolling_review_count_{window_days}d'] = asin_data['review_count_window'].rolling(
                    window=window_days, min_periods=1
                ).sum()
                
                # Calculate rolling average reviews per day (total reviews / window days)
                asin_data[f'rolling_avg_reviews_per_day_{window_days}d'] = (
                    asin_data['review_count_window'].rolling(window=window_days, min_periods=1).sum() / window_days
                )
                
                # Calculate rolling weighted averages for helpful votes and verified purchases
                if 'helpful_vote_weighted_avg_rating' in asin_data.columns:
                    asin_data[f'rolling_helpful_vote_weighted_avg_{window_days}d'] = asin_data['helpful_vote_weighted_avg_rating'].rolling(
                        window=window_days, min_periods=1
                    ).mean()
                
                if 'verified_purchase_weighted_avg_rating' in asin_data.columns:
                    asin_data[f'rolling_verified_purchase_weighted_avg_{window_days}d'] = asin_data['verified_purchase_weighted_avg_rating'].rolling(
                        window=window_days, min_periods=1
                    ).mean()
            
            rolling_stats.append(asin_data)
        
        # Combine all ASINs back together
        result_df = pd.concat(rolling_stats, ignore_index=True)
        
        logger.info(f"Added rolling statistics with windows: 7, 30, 90 days")
        return result_df
    
    def save_organized_data(self, df: pd.DataFrame, category: str, 
                           config_metadata: Optional[Dict] = None) -> Dict[str, Path]:
        """
        Save organized dataset in multiple formats with configuration metadata.
        
        Args:
            df: DataFrame to save
            category: Product category
            config_metadata: Optional configuration parameters used to create the dataset
        """
        logger.info("Saving organized dataset")
        
        if df.empty:
            logger.warning("No data to save")
            return {}
        
        # Determine output directory and base name based on data source
        if config_metadata and config_metadata.get('data_source') == 'huggingface_only':
            # Use new directory structure for HuggingFace-only data
            output_dir = self.processed_datasets_dir / "huggingface_only"
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = category
        elif config_metadata and config_metadata.get('data_source') == 'keepa_huggingface_combined':
            output_dir = self.output_dir
            base_name = f"amazon_keepa_{category}_{datetime.now().strftime('%Y%m%d')}"
        else:
            output_dir = self.output_dir
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

    def _add_rolling_window_statistics(self, df: pd.DataFrame, window_sizes: list[int] = [3, 5, 10, 30]) -> pd.DataFrame:
        """Add rolling statistics across a number of windows (not days)."""
        df = df.sort_values(['asin', 'timestamp']).copy()
        for asin in df['asin'].unique():
            asin_mask = df['asin'] == asin
            for w in window_sizes:
                # Basic rolling statistics
                df.loc[asin_mask, f'rolling_avg_rating_{w}w'] = (
                    df.loc[asin_mask, 'avg_rating_window'].rolling(window=w, min_periods=1).mean()
                )
                
                # Rolling review count (total reviews over w windows)
                if 'review_count_window' in df.columns:
                    df.loc[asin_mask, f'rolling_review_count_{w}w'] = (
                        df.loc[asin_mask, 'review_count_window'].rolling(window=w, min_periods=1).sum()
                    )
                
                # Add rolling statistics for weighted averages if they exist
                if 'helpful_vote_weighted_avg_rating' in df.columns:
                    df.loc[asin_mask, f'rolling_helpful_vote_weighted_avg_{w}w'] = (
                        df.loc[asin_mask, 'helpful_vote_weighted_avg_rating'].rolling(window=w, min_periods=1).mean()
                    )
                
                if 'verified_purchase_weighted_avg_rating' in df.columns:
                    df.loc[asin_mask, f'rolling_verified_purchase_weighted_avg_{w}w'] = (
                        df.loc[asin_mask, 'verified_purchase_weighted_avg_rating'].rolling(window=w, min_periods=1).mean()
                    )
        return df

    def load_dataset_with_metadata(self, category: str, work_dir: Path = Path(".")) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load a processed HuggingFace-only dataset along with its metadata file.
        
        Args:
            category: Product category
            work_dir: Working directory
            
        Returns:
            Tuple of (main_dataset, metadata_dataset)
        """
        # Determine file paths
        processed_dir = work_dir / "processed_datasets" / "huggingface_only"
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


@app.command()
def download_huggingface(
    category: str = typer.Option("All_Beauty", help="Amazon product category"),
    min_reviews: int = typer.Option(10, help="Minimum reviews per ASIN"),
    max_asins: Optional[int] = typer.Option(None, help="Maximum ASINs to extract (None = all ASINs)"),
    sample_reviews: Optional[int] = typer.Option(None, help="Sample size for testing"),
    work_dir: Path = typer.Option(".", help="Working directory")
):
    """Download HuggingFace data, save locally, and extract ASINs in one operation."""
    max_asins_display = "All ASINs" if max_asins is None else str(max_asins)
    console.print(Panel.fit(
        f"[bold blue]Downloading HuggingFace Data & Extracting ASINs[/bold blue]\n"
        f"Category: {category}\n"
        f"Min Reviews: {min_reviews}\n"
        f"Max ASINs: {max_asins_display}",
        border_style="blue"
    ))
    
    pipeline = AmazonKeepaDataPipeline(work_dir)
    
    try:
        output_paths = pipeline.download_huggingface_data(category, min_reviews, max_asins, sample_reviews)
        
        console.print("\n[bold green]Download and extraction completed![/bold green]")
        console.print("Files saved:")
        for file_type, path in output_paths.items():
            if path:
                console.print(f"  {file_type}: {path}")
        
        console.print(f"\n[bold]Next Steps:[/bold]")
        console.print(f"  Use ASIN file with: python keepa.py download <api_key_file> {output_paths['asins']}")
        console.print(f"  Then merge datasets: python dataset_builder.py merge-datasets --category {category}")
                
    except Exception as e:
        console.print(f"[bold red]Download failed: {e}[/bold red]")
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
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with full tracebacks and stop on first error")
):
    """Merge Keepa price data with Amazon review data using different windowing strategies."""
    
    # Validate windowing strategy
    valid_strategies = ["price_observation", "calendar", "review_frequency"]
    if windowing_strategy not in valid_strategies:
        console.print(f"[bold red]Invalid windowing strategy: {windowing_strategy}[/bold red]")
        console.print(f"Valid options: {', '.join(valid_strategies)}")
        raise typer.Exit(1)
    
    # Create strategy description
    if windowing_strategy == "price_observation":
        strategy_desc = f"Price observation windows ({time_window_days} days around each price change)"
    elif windowing_strategy == "calendar":
        strategy_desc = f"Calendar windows ({calendar_window_interval} intervals)"
    else:  # review_frequency
        strategy_desc = f"Review frequency windows (every {review_window_size} reviews)"
    
    console.print(Panel.fit(
        f"[bold blue]Merging Datasets[/bold blue]\n"
        f"Category: {category}\n"
        f"Strategy: {strategy_desc}\n"
        f"Include All Reviews: {include_all_reviews}",
        border_style="blue"
    ))
    
    pipeline = AmazonKeepaDataPipeline(work_dir)
    
    try:
        # Check if Keepa data exists
        keepa_files = list(pipeline.data_dir.glob("*_complete.json"))
        if not keepa_files:
            console.print("[bold red]No Keepa data found! Run keepa.py download first.[/bold red]")
            raise typer.Exit(1)
        
        console.print(f"Found {len(keepa_files)} Keepa data files")
        
        # Handle HuggingFace data
        if pull_huggingface:
            console.print("Pulling HuggingFace data from cloud...")
            pipeline.download_huggingface_data(category, 10, None)  # Default values for min_reviews and max_asins
        else:
            # Check if local HuggingFace data exists
            category_dir = pipeline.huggingface_dir / category
            if not category_dir.exists():
                console.print("[bold red]Local HuggingFace data not found! Use --pull-huggingface to download.[/bold red]")
                raise typer.Exit(1)
        
        # Merge datasets with appropriate parameters
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
        else:  # review_frequency
            combined_df = pipeline.combine_datasets(
                category, time_window_days, include_all_reviews,
                windowing_strategy, calendar_window_interval, review_window_size, debug
            )
        
        # Prepare configuration metadata
        config_metadata = {
            'data_source': 'keepa_huggingface_combined',
            'windowing_strategy': windowing_strategy,
            'time_window_days': time_window_days if windowing_strategy == "price_observation" else None,
            'calendar_window_interval': calendar_window_interval if windowing_strategy == "calendar" else None,
            'review_window_size': review_window_size if windowing_strategy == "review_frequency" else None,
            'include_all_reviews': include_all_reviews
        }
        
        # Save results with configuration metadata
        output_paths = pipeline.save_organized_data(combined_df, category, config_metadata)
        
        console.print("\n[bold green]Merge completed![/bold green]")
        console.print("Output files:")
        for file_type, path in output_paths.items():
            console.print(f"  {file_type}: {path}")
            
    except Exception as e:
        if debug:
            console.print(f"[bold red]Merge failed: {e}[/bold red]")
            import traceback
            console.print(f"[bold red]Full traceback:[/bold red]\n{traceback.format_exc()}")
            raise typer.Exit(1)
        else:
            console.print(f"[bold red]Merge failed: {e}[/bold red]")
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
    work_dir: Path = typer.Option(".", help="Working directory")
):
    """Run the complete pipeline: extract ASINs, download data, and merge."""
    
    # Create strategy description
    if windowing_strategy == "price_observation":
        strategy_desc = f"Price observation windows ({time_window_days} days around each price change)"
    elif windowing_strategy == "calendar":
        strategy_desc = f"Calendar windows ({calendar_window_days} day intervals)"
    else:  # review_frequency
        strategy_desc = f"Review frequency windows (every {review_window_size} reviews)"
    
    max_asins_display = "All ASINs" if max_asins is None else str(max_asins)
    console.print(Panel.fit(
        f"[bold blue]Full Pipeline[/bold blue]\n"
        f"Category: {category}\n"
        f"Max ASINs: {max_asins_display}\n"
        f"Strategy: {strategy_desc}",
        border_style="blue"
    ))
    
    pipeline = AmazonKeepaDataPipeline(work_dir)
    
    try:
        # Step 1: Download HuggingFace data and extract ASINs
        console.print("\n[bold cyan]Step 1: Downloading HuggingFace data and extracting ASINs[/bold cyan]")
        output_paths = pipeline.download_huggingface_data(category, min_reviews, max_asins, sample_reviews)
        asins_file = output_paths['asins']
        
        # Step 2: Note about Keepa download and merge
        console.print("\n[bold cyan]Step 2: Keepa Download Required[/bold cyan]")
        console.print(f"Please run: python keepa.py download {api_key_file} {asins_file}")
        
        # Step 3: Note about merge with windowing strategy
        console.print("\n[bold cyan]Step 3: Merge Datasets[/bold cyan]")
        if windowing_strategy == "price_observation":
            merge_cmd = f"python dataset_builder.py merge-datasets --category {category} --windowing-strategy {windowing_strategy} --time-window-days {time_window_days}"
        elif windowing_strategy == "calendar":
            merge_cmd = f"python dataset_builder.py merge-datasets --category {category} --windowing-strategy {windowing_strategy} --calendar-window-days {calendar_window_days}"
        else:  # review_frequency
            merge_cmd = f"python dataset_builder.py merge-datasets --category {category} --windowing-strategy {windowing_strategy} --review-window-size {review_window_size}"
        
        if include_all_reviews:
            merge_cmd += " --include-all-reviews"
        
        console.print(f"Then run: {merge_cmd}")
        
    except Exception as e:
        console.print(f"[bold red]Pipeline failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def list_categories(
    work_dir: Path = typer.Option(".", help="Working directory"),
    show_info: bool = typer.Option(False, "--info", help="Show detailed information for each category")
):
    """List available Amazon product categories from all_categories.txt."""
    console.print(Panel.fit(
        "[bold blue]Loading Amazon Product Categories[/bold blue]",
        border_style="blue"
    ))
    
    pipeline = AmazonKeepaDataPipeline(work_dir)
    
    try:
        categories = pipeline.get_available_categories()
        
        if not categories:
            console.print("[bold red]No categories found![/bold red]")
            raise typer.Exit(1)
        
        console.print(f"\n[bold green]Found {len(categories)} categories:[/bold green]")
        
        if show_info:
            # Show detailed information
            for category in categories:
                info = pipeline.get_category_info(category)
                
                status = "[green]✓[/green]" if info['local_data_exists'] else "[red]✗[/red]"
                console.print(f"\n{status} [bold]{category}[/bold]")
                
                if info['local_data_exists']:
                    console.print(f"  Reviews: {info['reviews_count']:,}")
                    console.print(f"  Metadata: {info['metadata_count']:,}")
                    console.print(f"  Unique ASINs: {info['unique_asins']:,}")
                    if info['date_range']:
                        console.print(f"  Date Range: {info['date_range']['start'][:10]} to {info['date_range']['end'][:10]}")
                else:
                    console.print("  [yellow]No local data[/yellow]")
        else:
            # Show simple list
            for i, category in enumerate(categories, 1):
                console.print(f"  {i:2d}. {category}")
        
        console.print(f"\n[bold]Usage:[/bold]")
        console.print(f"  python dataset_builder.py download-huggingface --category <category_name>")
        console.print(f"  python dataset_builder.py merge-datasets --category <category_name>")
        
    except Exception as e:
        console.print(f"[bold red]Failed to load categories: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def process_huggingface_only(
    category: str = typer.Option("All_Beauty", help="Amazon product category"),
    windowing_strategy: str = typer.Option("calendar", help="Windowing strategy: calendar or review_frequency"),
    calendar_window_interval: str = typer.Option("1d", help="Time interval per calendar window (e.g., 1d, 2M, 1w) for calendar strategy"),
    review_window_size: int = typer.Option(10, help="Reviews per window (for review_frequency strategy)"),
    include_all_reviews: bool = typer.Option(True, help="Include all reviews in time window"),
    min_reviews_per_asin: int = typer.Option(10, help="Minimum reviews required per ASIN"),
    max_asins: Optional[int] = typer.Option(None, help="Maximum ASINs to process (None = all ASINs)"),
    work_dir: Path = typer.Option(".", help="Working directory"),
    pull_huggingface: bool = typer.Option(False, help="Pull HuggingFace data from cloud instead of using local"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with full tracebacks and stop on first error"),
    use_polars: bool = typer.Option(False, "--polars", help="Use Polars for faster processing (requires polars package)"),
    rolling_window_sizes: list[int] = typer.Option([3, 5, 10, 30], help="Rolling window sizes for additional statistics"),
    upsample: bool = typer.Option(False, "--upsample", help="Create empty buckets for missing time periods (increases row count)")
):
    """Process HuggingFace review data without Keepa price data.
    
    Performance options:
    - --optimized: Use single groupby operation over all ASINs (much faster for large datasets)
    - --polars: Use Polars for processing (requires polars package, fastest option)
    """
    
    # Validate windowing strategy
    valid_strategies = ["calendar", "review_frequency"]
    if windowing_strategy not in valid_strategies:
        console.print(f"[bold red]Invalid windowing strategy: {windowing_strategy}[/bold red]")
        console.print(f"Valid options: {', '.join(valid_strategies)}")
        raise typer.Exit(1)
    
    # Create strategy description
    if windowing_strategy == "calendar":
        strategy_desc = f"Calendar windows ({calendar_window_interval} intervals)"
    else:  # review_frequency
        strategy_desc = f"Review frequency windows (every {review_window_size} reviews)"
    
    console.print(Panel.fit(
        f"[bold blue]Processing HuggingFace Data Only[/bold blue]\n"
        f"Category: {category}\n"
        f"Strategy: {strategy_desc}\n"
        f"Include All Reviews: {include_all_reviews}\n"
        f"Min Reviews per ASIN: {min_reviews_per_asin}\n"
        f"Max ASINs: {max_asins if max_asins is not None else 'All'}",
        border_style="blue"
    ))
    
    pipeline = AmazonKeepaDataPipeline(work_dir)
    
    try:
        # Handle HuggingFace data
        if pull_huggingface:
            console.print("Pulling HuggingFace data from cloud...")
            pipeline.download_huggingface_data(category, min_reviews_per_asin, max_asins)
        else:
            # Check if local HuggingFace data exists
            category_dir = pipeline.huggingface_dir / category
            if not category_dir.exists():
                console.print("[bold red]Local HuggingFace data not found! Use --pull-huggingface to download.[/bold red]")
                raise typer.Exit(1)
        
        # Process HuggingFace data using selected method
        if use_polars:
            console.print("[bold cyan]Using Polars for processing...[/bold cyan]")
            combined_df = pipeline.process_huggingface_only_polars(
                category, windowing_strategy, calendar_window_interval, 
                review_window_size, include_all_reviews, min_reviews_per_asin, max_asins, debug, rolling_window_sizes, upsample
            )
        else:
            console.print("[bold cyan]Using pandas processing...[/bold cyan]")
            combined_df = pipeline.process_huggingface_only(
                category, windowing_strategy, calendar_window_interval, 
                review_window_size, include_all_reviews, min_reviews_per_asin, max_asins, debug, rolling_window_sizes, upsample
            )
        
        # Prepare configuration metadata
        config_metadata = {
            'data_source': 'huggingface_only',
            'windowing_strategy': windowing_strategy,
            'calendar_window_interval': calendar_window_interval if windowing_strategy == "calendar" else None,
            'review_window_size': review_window_size if windowing_strategy == "review_frequency" else None,
            'include_all_reviews': include_all_reviews,
            'min_reviews_per_asin': min_reviews_per_asin,
            'max_asins': max_asins,
            'rolling_window_sizes': rolling_window_sizes,
            'upsample': upsample
        }
        
        # Save results using the organized data method with configuration metadata
        try:
            output_paths = pipeline.save_organized_data(combined_df, category, config_metadata)
            
            console.print("\n[bold green]Processing completed![/bold green]")
            console.print("Output files:")
            for file_type, path in output_paths.items():
                console.print(f"  {file_type}: {path}")
            
            console.print(f"\n[bold]Dataset Summary:[/bold]")
            console.print(f"  Total Records: {len(combined_df):,}")
            console.print(f"  Unique ASINs: {combined_df['asin'].nunique():,}")
            console.print(f"  Date Range: {combined_df['timestamp'].min().date()} to {combined_df['timestamp'].max().date()}")
            console.print(f"  Average Reviews per Timepoint: {combined_df['review_count_window'].mean():.1f}")
            
            # Add note about metadata file
            if 'metadata' in output_paths:
                console.print(f"\n[bold yellow]Note:[/bold yellow] Product metadata has been saved to a separate file to reduce dataset size.")
                console.print(f"  Use the metadata file to join product information (title, category, brand, etc.) when needed.")
                
        except Exception as e:
            if debug:
                console.print(f"[bold red]Error saving data: {e}[/bold red]")
                import traceback
                console.print(f"[bold red]Full traceback:[/bold red]\n{traceback.format_exc()}")
                raise typer.Exit(1)
            else:
                console.print(f"[bold red]Error saving data: {e}[/bold red]")
                raise typer.Exit(1)
            
    except Exception as e:
        if debug:
            console.print(f"[bold red]Processing failed: {e}[/bold red]")
            import traceback
            console.print(f"[bold red]Full traceback:[/bold red]\n{traceback.format_exc()}")
            raise typer.Exit(1)
        else:
            console.print(f"[bold red]Processing failed: {e}[/bold red]")
            raise typer.Exit(1)

@app.command()
def category_info(
    category: str = typer.Argument(..., help="Category name to get information for"),
    work_dir: Path = typer.Option(".", help="Working directory")
):
    """Get detailed information about a specific category."""
    console.print(Panel.fit(
        f"[bold blue]Category Information[/bold blue]\n"
        f"Category: {category}",
        border_style="blue"
    ))
    
    pipeline = AmazonKeepaDataPipeline(work_dir)
    
    try:
        info = pipeline.get_category_info(category)
        
        console.print(f"\n[bold]Category:[/bold] {info['category']}")
        console.print(f"[bold]Local Data:[/bold] {'✓ Available' if info['local_data_exists'] else '✗ Not Available'}")
        
        if info['local_data_exists']:
            console.print(f"[bold]Reviews:[/bold] {info['reviews_count']:,}")
            console.print(f"[bold]Metadata Records:[/bold] {info['metadata_count']:,}")
            console.print(f"[bold]Unique ASINs:[/bold] {info['unique_asins']:,}")
            
            if info['date_range']:
                console.print(f"[bold]Date Range:[/bold] {info['date_range']['start'][:10]} to {info['date_range']['end'][:10]}")
            
            if info['download_date']:
                console.print(f"[bold]Downloaded:[/bold] {info['download_date'][:10]}")
        
        console.print(f"\n[bold]Next Steps:[/bold]")
        if not info['local_data_exists']:
            console.print(f"  1. Download data and extract ASINs: python dataset_builder.py download-huggingface --category {category}")
        console.print(f"  2. Process HuggingFace only: python dataset_builder.py process-huggingface-only --category {category}")
        console.print(f"  3. Download Keepa data: python keepa.py download <api_key> <asins_file>")
        console.print(f"  4. Merge datasets: python dataset_builder.py merge-datasets --category {category}")
        
    except Exception as e:
        console.print(f"[bold red]Failed to get category info: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()