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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def step1_extract_asins(self, category: str = "All_Beauty", min_reviews: int = 10, 
                           max_asins: int = 1000, sample_reviews: Optional[int] = None) -> Path:
        logger.info(f"Step 1: Extracting ASINs from {category} category")
        
        logger.info("Loading Amazon Reviews dataset...")
        try:
            dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}")
            df = dataset["full"].to_pandas()
            
            if sample_reviews:
                logger.info(f"Sampling {sample_reviews} reviews for testing")
                df = df.sample(n=min(sample_reviews, len(df)), random_state=42)
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        logger.info("Counting reviews per ASIN...")
        asin_counts = df['parent_asin'].value_counts()
        
        popular_asins = asin_counts[asin_counts >= min_reviews].head(max_asins)
        
        logger.info(f"Found {len(popular_asins)} ASINs with at least {min_reviews} reviews")
        logger.info(f"Top 5 ASINs: {list(popular_asins.head().index)}")
        
        asins_file = self.work_dir / f"{category}_asins.txt"
        with open(asins_file, 'w') as f:
            for asin in popular_asins.index:
                f.write(f"{asin}\n")
        
        logger.info(f"Saved {len(popular_asins)} ASINs to {asins_file}")
        return asins_file
    
    def step2_fetch_keepa_data(self, asins_file: Path, api_key_file: Path) -> None:
        logger.info("Step 2: Fetching Keepa price data using existing script")
        
        if not asins_file.exists():
            raise FileNotFoundError(f"ASINs file not found: {asins_file}")
        if not api_key_file.exists():
            raise FileNotFoundError(f"API key file not found: {api_key_file}")
        
        keepa_script = self.keepa_dir / "keepa_test.py"
        if not keepa_script.exists():
            logger.error(f"Keepa script not found at: {keepa_script}")
            logger.error(f"Looking in keepa_dir: {self.keepa_dir}")
            logger.error(f"Files in keepa_dir: {list(self.keepa_dir.glob('*.py'))}")
            raise FileNotFoundError(f"Keepa script not found: {keepa_script}")
        
        cmd = [
            sys.executable, str(keepa_script),
            str(api_key_file),
            str(asins_file), 
            "--output-dir", str(self.data_dir)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Keepa script failed: {result.stderr}")
            raise RuntimeError(f"Keepa script failed with return code {result.returncode}")
        
        logger.info("Keepa data fetch completed successfully")
        
        json_files = list(self.data_dir.glob("*_history.json"))
        logger.info(f"Downloaded price data for {len(json_files)} ASINs")
    
    def step3_load_amazon_data(self, category: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Step 3: Loading Amazon Reviews data for {category}")
        
        reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}")
        reviews_df = reviews_dataset["full"].to_pandas()
        reviews_df['datetime'] = pd.to_datetime(reviews_df['timestamp'], unit='ms')
        
        try:
            meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}")
            metadata_df = meta_dataset["full"].to_pandas()
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            metadata_df = pd.DataFrame()
        
        logger.info(f"Loaded {len(reviews_df)} reviews and {len(metadata_df)} metadata records")
        return reviews_df, metadata_df
    
    def step4_combine_datasets(self, category: str, time_window_days: int = 30) -> pd.DataFrame:
        logger.info("Step 4: Combining datasets")
        
        reviews_df, metadata_df = self.step3_load_amazon_data(category)
        
        json_files = list(self.data_dir.glob("*_history.json"))
        if not json_files:
            raise RuntimeError("No Keepa data files found. Run steps 1-2 first.")
        
        combined_records = []
        
        for json_file in json_files:
            asin = json_file.stem.replace("_history", "")
            logger.info(f"Processing {asin}")
            
            try:
                history_data = load_history(json_file)
                price_df = parse_price_history(history_data)
                
                if price_df.empty:
                    logger.warning(f"No price data for {asin}")
                    continue
                
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
                
                asin_reviews = reviews_df[
                    (reviews_df['asin'] == asin) | (reviews_df['parent_asin'] == asin)
                ].copy()
                
                for _, price_row in price_df.iterrows():
                    timestamp = price_row.name
                    
                    start_date = timestamp - pd.Timedelta(days=time_window_days)
                    end_date = timestamp + pd.Timedelta(days=time_window_days)
                    
                    window_reviews = asin_reviews[
                        (asin_reviews['datetime'] >= start_date) & 
                        (asin_reviews['datetime'] <= end_date)
                    ]
                    
                    if len(window_reviews) > 0:
                        review_count_window = len(window_reviews)
                        avg_rating_window = window_reviews['rating'].mean()
                        helpful_votes = window_reviews['helpful_vote'].sum()
                        verified_ratio = window_reviews['verified_purchase'].mean()
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
                        
                        'amazon_price': price_row.get('AMAZON'),
                        'new_price': price_row.get('NEW'), 
                        'used_price': price_row.get('USED'),
                        'list_price': price_row.get('LISTPRICE'),
                        'sales_rank': price_row.get('SALES_RANK'),
                        'rating_keepa': price_row.get('RATING'),
                        'review_count_keepa': price_row.get('COUNT_REVIEWS'),
                        'new_offers_count': price_row.get('COUNT_NEW'),
                        'used_offers_count': price_row.get('COUNT_USED'),
                        
                        'review_count_window': review_count_window,
                        'avg_rating_window': avg_rating_window,
                        'helpful_votes_sum': helpful_votes,
                        'verified_purchases_ratio': verified_ratio,
                        'sample_review_text': sample_review[:500],
                        'time_window_days': time_window_days
                    }
                    
                    combined_records.append(record)
                    
            except Exception as e:
                logger.error(f"Error processing {asin}: {e}")
                continue
        
        combined_df = pd.DataFrame(combined_records)
        logger.info(f"Combined dataset created: {len(combined_df)} records, {combined_df['asin'].nunique()} unique ASINs")
        
        return combined_df
    
    def step5_save_organized_data(self, df: pd.DataFrame, category: str) -> Dict[str, Path]:
        logger.info("Step 5: Saving organized dataset")
        
        if df.empty:
            logger.warning("No data to save")
            return {}
        
        base_name = f"amazon_keepa_{category}_{datetime.now().strftime('%Y%m%d')}"
        
        output_paths = {}
        
        csv_path = self.output_dir / f"{base_name}.csv" 
        df.to_csv(csv_path, index=False)
        output_paths['csv'] = csv_path
        logger.info(f"Saved CSV: {csv_path}")
        
        parquet_path = self.output_dir / f"{base_name}.parquet"
        df.to_parquet(parquet_path, index=False)
        output_paths['parquet'] = parquet_path
        logger.info(f"Saved Parquet: {parquet_path}")
        
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
            'price_data_coverage': {
                col: f"{df[col].notna().mean():.1%}" 
                for col in ['amazon_price', 'new_price', 'used_price']
            },
            'review_data_coverage': {
                'avg_reviews_per_timepoint': df['review_count_window'].mean(),
                'timepoints_with_reviews': f"{(df['review_count_window'] > 0).mean():.1%}"
            }
        }
        
        summary_path = self.output_dir / f"{base_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        output_paths['summary'] = summary_path
        logger.info(f"Saved summary: {summary_path}")
        
        return output_paths
    
    def run_full_pipeline(self, category: str = "All_Beauty", api_key_file: str = "keepa_api_key.txt",
                         min_reviews: int = 10, max_asins: int = 100, time_window_days: int = 30,
                         sample_reviews: Optional[int] = None) -> Dict[str, Path]:
        logger.info(f"Starting full pipeline for category: {category}")
        
        api_key_path = Path(api_key_file)
        
        try:
            asins_file = self.step1_extract_asins(category, min_reviews, max_asins, sample_reviews)
            
            self.step2_fetch_keepa_data(asins_file, api_key_path)
            
            combined_df = self.step4_combine_datasets(category, time_window_days)
            
            output_paths = self.step5_save_organized_data(combined_df, category)
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Output files: {list(output_paths.values())}")
            
            return output_paths
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Amazon-Keepa Data Pipeline")
    parser.add_argument("--category", default="All_Beauty", help="Amazon category")
    parser.add_argument("--api-key-file", default="keepa_api_key.txt", help="Keepa API key file")
    parser.add_argument("--min-reviews", type=int, default=10, help="Minimum reviews per ASIN")
    parser.add_argument("--max-asins", type=int, default=100, help="Maximum ASINs to process")
    parser.add_argument("--time-window", type=int, default=30, help="Review aggregation window (days)")
    parser.add_argument("--sample-reviews", type=int, help="Sample size for testing")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    
    args = parser.parse_args()
    
    pipeline = AmazonKeepaDataPipeline(Path(args.work_dir))
    
    output_paths = pipeline.run_full_pipeline(
        category=args.category,
        api_key_file=args.api_key_file,
        min_reviews=args.min_reviews,
        max_asins=args.max_asins,
        time_window_days=args.time_window,
        sample_reviews=args.sample_reviews
    )
    
    print("\nPipeline completed! Output files:")
    for file_type, path in output_paths.items():
        print(f"  {file_type}: {path}")


if __name__ == "__main__":
    main()