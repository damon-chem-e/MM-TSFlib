# Keepa Data Pipeline

A comprehensive data pipeline for collecting, processing, and analyzing Amazon product price history data using the Keepa API. This system fetches historical price data for Amazon products and combines it with review data to create rich datasets for analysis.

## Overview

This pipeline consists of several interconnected modules that work together to:
1. Extract ASINs from Amazon review datasets
2. Fetch historical price data from Keepa API
3. Process and analyze price history data
4. Combine price data with Amazon review data
5. Generate visualizations and statistics

## Module Descriptions

### 1. `keepa.py` - Main CLI Application
**Purpose**: Primary command-line interface for downloading Keepa data with advanced features.

**Key Features**:
- Batch processing with configurable batch sizes (max 100 ASINs per request)
- Rate limiting and retry logic with exponential backoff
- Progress tracking with rich console output
- Resume capability for interrupted downloads
- Comprehensive error handling and logging
- Token consumption tracking
- Multiple data format options (raw responses, processed data)

**Main Classes**:
- `DownloadConfig`: Configuration dataclass for API settings
- `KeepaDownloader`: Core downloader class with batch processing
- CLI commands: `download`, `validate_asins`

### 2. `keepa_test.py` - Simple Keepa Client
**Purpose**: Basic, lightweight script for simple Keepa API calls.

**Features**:
- Single ASIN processing
- Basic rate limiting (0.6s delay between requests)
- Simple JSON output format
- Minimal dependencies

**Use Case**: Quick testing or small-scale data collection

### 3. `dataset_builder.py` - Modular Data Pipeline
**Purpose**: Modular pipeline for combining Keepa price data with Amazon review data using Typer CLI.

**Pipeline Commands**:
1. **`extract-asins`**: Extract ASINs from HuggingFace Amazon Reviews dataset
2. **`download-huggingface`**: Download and save HuggingFace data locally
3. **`process-huggingface-only`**: Process HuggingFace review data without Keepa price data
4. **`merge-datasets`**: Combine local Keepa data with local HuggingFace data
5. **`full-pipeline`**: Run steps 1-2, then guide user through Keepa download

**Key Features**:
- **Modular Design**: Separate download and merge steps for robustness
- **Local Data Support**: Use local Keepa and HuggingFace data
- **Rich CLI**: Typer-based interface with progress bars and rich output
- **Flexible Review Handling**: Option to include all reviews in time windows with delimiters
- **Multiple Output Formats**: CSV, Parquet, and summary JSON
- **Configurable Parameters**: Time windows, review thresholds, data sources

### 4. `analyze_history.py` - Price History Analysis
**Purpose**: Analyze and visualize Keepa price history data.

**Features**:
- Parse Keepa CSV format into pandas DataFrames
- Generate statistical summaries
- Create price history plots
- Handle multiple price types (AMAZON, NEW, USED)
- Rich console output with tables and panels

**Key Functions**:
- `load_history()`: Load JSON files with error handling
- `parse_price_history()`: Convert Keepa CSV to DataFrame
- `analyze_product()`: Generate statistics and plots

## Database Format

### Input Data Sources

#### Keepa API Response Format
```json
{
  "asin": "B0DBZFV1V2",
  "csv": [
    [keepa_timestamp, price_cents, keepa_timestamp, price_cents, ...],  // AMAZON prices
    [keepa_timestamp, price_cents, keepa_timestamp, price_cents, ...],  // NEW prices  
    [keepa_timestamp, price_cents, keepa_timestamp, price_cents, ...]   // USED prices
  ],
  "stats": {...},
  "offers": [...],
  "buybox": {...}
}
```

#### Amazon Reviews Dataset Format
- **Reviews**: ASIN, rating, review text, timestamp, helpful votes, verified purchase
- **Metadata**: Product title, brand, category, average rating, price

### Output Database Format

#### Combined Dataset Structure (with Keepa data)
```csv
asin,date,amazon_price,new_price,used_price,review_count_window,
avg_rating_window,helpful_votes,verified_ratio,sample_review,
product_title,product_category,product_brand,avg_rating,rating_count
```

**Key Fields**:
- `asin`: Amazon Standard Identification Number
- `date`: Price observation date
- `amazon_price`, `new_price`, `used_price`: Price types from Keepa
- `review_count_window`: Number of reviews in time window
- `avg_rating_window`: Average rating in time window
- `helpful_votes_sum`: Total helpful votes in window
- `verified_purchases_ratio`: Ratio of verified purchases
- `sample_review_text`: Review text(s) from window (with delimiters if multiple)
- `include_all_reviews`: Boolean flag indicating if all reviews were included
- Product metadata fields (title, brand, category, etc.)

#### HuggingFace-Only Dataset Structure
```csv
asin,timestamp,date,year,month,title,category,brand,price_metadata,avg_rating_metadata,rating_count_metadata,
amazon_price,new_price,used_price,list_price,sales_rank,rating_keepa,review_count_keepa,new_offers_count,used_offers_count,
review_count_window,avg_rating_window,helpful_votes_sum,verified_purchases_ratio,sample_review_text,window_size,windowing_strategy,include_all_reviews
```

**Key Fields** (HuggingFace-only):
- `asin`: Amazon Standard Identification Number
- `timestamp`, `date`, `year`, `month`: Time information
- `title`, `category`, `brand`: Product metadata from HuggingFace
- `price_metadata`, `avg_rating_metadata`, `rating_count_metadata`: Metadata from HuggingFace
- `amazon_price`, `new_price`, `used_price`, etc.: All set to `None` (no Keepa data)
- `review_count_window`, `avg_rating_window`, `helpful_votes_sum`, `verified_purchases_ratio`: Review statistics
- `sample_review_text`: Review text(s) from window
- `window_size`, `windowing_strategy`: Windowing configuration

## Command Line Interface

### Main CLI (`keepa.py`)

#### Download Command
```bash
python keepa.py download <api_key_file> <asin_file> [OPTIONS]
```

**Required Arguments**:
- `api_key_file`: Path to file containing Keepa API key
- `asin_file`: Path to file containing list of ASINs (one per line)

**Options**:
- `--output-dir`: Output directory (default: "keepa/data")
- `--domain`: Amazon domain (1=US, 3=UK, 4=DE, etc.) (default: 1)
- `--batch-size`: ASINs per API request, max 100 (default: 100)
- `--max-retries`: Maximum retries per request (default: 3)
- `--rate-limit-delay`: Delay between requests in seconds (default: 0.6)
- `--resume-from-batch`: Resume from specific batch number (default: 0)
- `--disable-offers`: Disable offer data collection
- `--disable-buybox`: Disable buybox data collection
- `--disable-rental`: Disable rental data collection
- `--verbose`: Enable verbose logging

**Example**:
```bash
python keepa.py download keepa-api-key.txt test-asins.txt --output-dir data --batch-size 50
```

#### Validate ASINs Command
```bash
python keepa.py validate-asins <asin_file>
```

### Simple Client (`keepa_test.py`)
```bash
python keepa_test.py <api_key_file> <asin_file> [--output-dir data]
```

### Dataset Builder (`dataset_builder.py`)

#### Extract ASINs Command
```bash
python dataset_builder.py extract-asins --category All_Beauty --min-reviews 10 --max-asins 1000
```

#### Download HuggingFace Data Command
```bash
python dataset_builder.py download-huggingface --category All_Beauty
```

#### Process HuggingFace Only Command
```bash
# Calendar windows (daily) - no Keepa data required
python dataset_builder.py process-huggingface-only --category Video_Games --windowing-strategy calendar --calendar-window-days 1

# Review frequency windows (every 20 reviews)
python dataset_builder.py process-huggingface-only --category Video_Games --windowing-strategy review_frequency --review-window-size 20

# Custom minimum reviews per ASIN
python dataset_builder.py process-huggingface-only --category Video_Games --min-reviews-per-asin 50

# Pull data from cloud and process
python dataset_builder.py process-huggingface-only --category Video_Games --pull-huggingface
```

#### Merge Datasets Command
```bash
# Price observation windows (default)
python dataset_builder.py merge-datasets --category All_Beauty --windowing-strategy price_observation --time-window-days 30

# Calendar windows (daily)
python dataset_builder.py merge-datasets --category All_Beauty --windowing-strategy calendar --calendar-window-days 1

# Review frequency windows (every 10 reviews)
python dataset_builder.py merge-datasets --category All_Beauty --windowing-strategy review_frequency --review-window-size 10

# Include all reviews with delimiters
python dataset_builder.py merge-datasets --category All_Beauty --windowing-strategy calendar --include-all-reviews
```

#### List Categories Command
```bash
# List all available categories
python dataset_builder.py list-categories

# List with detailed information
python dataset_builder.py list-categories --info
```

#### Category Information Command
```bash
# Get detailed info about a specific category
python dataset_builder.py category-info All_Beauty
```

#### Full Pipeline Command
```bash
python dataset_builder.py full-pipeline --category All_Beauty --max-asins 100
```

**Key Options**:
- `--category`: Amazon product category (default: "All_Beauty")
- `--windowing-strategy`: Window strategy: price_observation, calendar, or review_frequency (default: "price_observation")
- `--time-window-days`: Days around price observation (for price_observation strategy, default: 30)
- `--calendar-window-days`: Days per calendar window (for calendar strategy, default: 1)
- `--review-window-size`: Reviews per window (for review_frequency strategy, default: 10)
- `--include-all-reviews`: Include all reviews in time window with delimiters (default: True)
- `--pull-huggingface`: Pull HuggingFace data from cloud instead of local (default: False)
- `--min-reviews`: Minimum reviews per ASIN (default: 10)
- `--max-asins`: Maximum ASINs to process (default: 100)

**HuggingFace-Only Processing Options**:
- `--windowing-strategy`: Window strategy: calendar or review_frequency (default: "calendar")
- `--min-reviews-per-asin`: Minimum reviews required per ASIN (default: 10)

### Analysis Tool (`analyze_history.py`)
```bash
python analyze_history.py
```

**Features**:
- Automatically processes all JSON files in `data/products/`
- Generates plots in `figures/` directory
- Displays statistical summaries in console

## Installation and Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Keepa API Key
1. Sign up at [Keepa.com](https://keepa.com)
2. Navigate to API section
3. Copy your API key
4. Save to `keepa-api-key.txt` file

### 3. Prepare ASIN List
Create a text file with ASINs (one per line):
```
B0DBZFV1V2
B0C1XP5CKT
B01HJ8OOYI
```

## Usage Examples

### Quick Start - Download Price Data
```bash
# Download data for test ASINs
python keepa.py download keepa-api-key.txt test-asins.txt

# Download with custom settings
python keepa.py download keepa-api-key.txt asins.txt \
  --output-dir my_data \
  --batch-size 50 \
  --rate-limit-delay 1.0 \
  --verbose
```

### Parallel Processing Setup
```bash
# Step 0: Discover available categories
python dataset_builder.py list-categories --info

# Step 1: Extract ASINs for Keepa download
python dataset_builder.py extract-asins --category All_Beauty --max-asins 100

# Step 2: Download HuggingFace data locally
python dataset_builder.py download-huggingface --category All_Beauty

# Step 3: Download Keepa data (using robust keepa.py)
python keepa.py download keepa-api-key.txt All_Beauty_asins.txt

# Step 4: Merge datasets with different windowing strategies
# Price observation windows (default)
python dataset_builder.py merge-datasets --category All_Beauty --windowing-strategy price_observation

# Calendar windows (daily)
python dataset_builder.py merge-datasets --category All_Beauty --windowing-strategy calendar --calendar-window-days 1

# Review frequency windows (every 10 reviews)
python dataset_builder.py merge-datasets --category All_Beauty --windowing-strategy review_frequency --review-window-size 10
```

### HuggingFace-Only Processing Setup
```bash
# Process review data without Keepa price data
python dataset_builder.py process-huggingface-only --category Video_Games --windowing-strategy calendar --calendar-window-days 1

# Process with review frequency windows
python dataset_builder.py process-huggingface-only --category Video_Games --windowing-strategy review_frequency --review-window-size 20

# Process with custom filtering
python dataset_builder.py process-huggingface-only --category Video_Games --min-reviews-per-asin 50 --include-all-reviews
```

### Multi-Machine Parallel Processing
```bash
# Machine 1: Process All_Beauty category
python dataset_builder.py full-pipeline --category All_Beauty --max-asins 1000

# Machine 2: Process All_Electronics category  
python dataset_builder.py full-pipeline --category All_Electronics --max-asins 1000

# Machine 3: Process All_Home_and_Kitchen category
python dataset_builder.py full-pipeline --category All_Home_and_Kitchen --max-asins 1000
```

### HuggingFace-Only Workflow
```bash
# Step 1: Download HuggingFace data for a category
python dataset_builder.py download-huggingface --category Video_Games

# Step 2: Process review data without price data
python dataset_builder.py process-huggingface-only --category Video_Games --windowing-strategy calendar

# Step 3: Process multiple categories in parallel
python dataset_builder.py process-huggingface-only --category All_Beauty --pull-huggingface
python dataset_builder.py process-huggingface-only --category All_Electronics --pull-huggingface
python dataset_builder.py process-huggingface-only --category All_Home_and_Kitchen --pull-huggingface
```

### Analyze Downloaded Data
```bash
# Generate plots and statistics
python analyze_history.py
```

## File Structure

```
keepa/
├── keepa.py              # Main CLI application
├── keepa_test.py         # Simple Keepa client
├── dataset_builder.py    # Modular data pipeline (Typer CLI)
├── analyze_history.py    # Price analysis tool
├── requirements.txt      # Python dependencies
├── test-asins.txt       # Sample ASIN list
├── keepa-api-key.txt    # API key file (create this)
├── data/                # Downloaded Keepa data
│   ├── raw/            # Raw API responses
│   └── products/       # Processed product data
├── huggingface_data/    # Local HuggingFace data
│   └── {category}/     # Category-specific data
│       ├── reviews.parquet
│       ├── metadata.parquet
│       └── summary.json
├── figures/            # Generated plots
└── combined_dataset/   # Final combined datasets
    ├── amazon_keepa_{category}_{date}.csv      # Combined datasets with price data
    ├── amazon_keepa_{category}_{date}.parquet  # Combined datasets with price data
    ├── huggingface_only_{category}_{date}.csv  # HuggingFace-only datasets
    └── huggingface_only_{category}_{date}.parquet # HuggingFace-only datasets
```

## Rate Limiting and Best Practices

- **Keepa API Limit**: 100 requests per minute
- **Default Delay**: 0.6 seconds between requests
- **Batch Size**: Maximum 100 ASINs per request
- **Retry Logic**: Exponential backoff with configurable max retries
- **Token Tracking**: Monitor token consumption in logs

## Error Handling

The system includes comprehensive error handling for:
- Invalid API keys
- Rate limit exceeded
- Network timeouts
- Invalid ASIN formats
- File I/O errors
- Data parsing errors

## Output Formats

### Raw Data
- **JSON**: Complete API responses
- **Individual Files**: One file per ASIN

### Processed Data
- **CSV**: Tabular format for analysis
- **Parquet**: Efficient columnar storage
- **HuggingFace Dataset**: For ML workflows

### Visualizations
- **PNG Plots**: Price history charts
- **Console Tables**: Statistical summaries

## Windowing Strategies

The system supports different windowing strategies for processing review data, with or without price data:

### 1. Price Observation Windows (Default)
Windows are centered on each **price observation** from Keepa data:

- **Window Center**: Each price observation timestamp from Keepa
- **Window Size**: Configurable number of days before and after the price observation
- **Default**: 30 days before and 30 days after (60-day total window)
- **Variable Spacing**: Windows are **not** constant time periods - they're centered on each price change

**Example**:
- Price observation on 2023-01-15
- Time window: 30 days
- Window: 2022-12-16 to 2023-02-14
- All reviews in this window are aggregated for that price point

**Use Case**: Analyzing the relationship between price changes and review activity

### 2. Calendar Windows
Fixed calendar-based windows for consistent time intervals:

- **Window Size**: Configurable number of days (default: 1 day)
- **Fixed Spacing**: Regular calendar intervals regardless of price changes
- **Price Handling**: Uses the most recent price before or during each window
- **High Frequency**: Maintains high temporal resolution for review metrics
- **Temporal Continuity**: Creates observations for all calendar windows within the review date range
- **ASIN Filtering**: Only includes ASINs that have reviews
- **Rolling Statistics**: Automatically adds rolling averages for 7, 30, and 90-day windows

**Example**:
- Calendar window: 1 day
- Review date range: 2023-01-01 to 2023-01-31
- Windows: 2023-01-01, 2023-01-02, ..., 2023-01-31 (all days in range)
- Each window contains reviews from that specific day (or empty if no reviews that day)
- Price data uses the most recent price available for that day (or null if no price data)
- Rolling statistics provide trend analysis over time

**Use Case**: High-frequency analysis of review patterns and sentiment over time with complete temporal coverage for products with reviews

### 3. Review Frequency Windows
Windows based on review volume rather than time:

- **Window Size**: Configurable number of reviews per window (default: 10 reviews)
- **Volume-Based**: Each window contains exactly N reviews (except possibly the last window)
- **Price Handling**: Uses the most recent price before the midpoint of each review window
- **Consistent Volume**: Ensures each window has similar review activity

**Example**:
- Review window size: 10 reviews
- Windows: Reviews 1-10, 11-20, 21-30, ...
- Each window contains exactly 10 reviews (chronologically ordered)
- Price data uses the most recent price before the 5th review in each window (or null if no price data)

**Use Case**: Analyzing review sentiment patterns with consistent sample sizes

### Strategy Comparison

| Strategy | Window Definition | Price Handling | Frequency | Best For | Available In |
|----------|------------------|----------------|-----------|----------|--------------|
| **Price Observation** | Centered on price changes | Current price | Variable | Price-review relationship analysis | Merge datasets only |
| **Calendar** | Fixed time intervals | Most recent price (or null) | High | Temporal trend analysis | Both merge and HuggingFace-only |
| **Review Frequency** | Fixed review counts | Most recent price (or null) | Variable | Volume-based analysis | Both merge and HuggingFace-only |

### Output Fields

All strategies add these fields to track the windowing approach:
- `window_size`: The size parameter used (days or review count)
- `windowing_strategy`: Which strategy was used ("price_observation", "calendar", or "review_frequency")

**Calendar Windows Additional Fields**:
- `rolling_avg_rating_7d`: 7-day rolling average rating
- `rolling_avg_rating_30d`: 30-day rolling average rating  
- `rolling_avg_rating_90d`: 90-day rolling average rating
- `rolling_review_count_7d`: 7-day rolling review count
- `rolling_review_count_30d`: 30-day rolling review count
- `rolling_review_count_90d`: 90-day rolling review count
- `rolling_helpful_votes_7d`: 7-day rolling helpful votes
- `rolling_helpful_votes_30d`: 30-day rolling helpful votes
- `rolling_helpful_votes_90d`: 90-day rolling helpful votes
- `rolling_verified_ratio_7d`: 7-day rolling verified purchase ratio
- `rolling_verified_ratio_30d`: 30-day rolling verified purchase ratio
- `rolling_verified_ratio_90d`: 90-day rolling verified purchase ratio

## Use Cases

### Combined Dataset (Keepa + HuggingFace)
- **Price-Review Analysis**: Study the relationship between price changes and review sentiment
- **Market Research**: Analyze how pricing affects customer satisfaction
- **Competitive Analysis**: Compare price and review patterns across products
- **Time Series Forecasting**: Predict price changes based on review patterns

### HuggingFace-Only Dataset
- **Review Analysis**: Study review patterns and sentiment over time
- **Product Research**: Analyze product popularity and user satisfaction
- **Text Mining**: Process review text for NLP applications
- **Time Series**: Create review-based time series datasets
- **Category Comparison**: Compare review patterns across categories
- **Sentiment Analysis**: Analyze customer sentiment trends
- **Content Analysis**: Study review content and helpfulness patterns

## Troubleshooting

### Common Issues

1. **API Key Invalid**: Check `keepa-api-key.txt` file format
2. **Rate Limited**: Increase `--rate-limit-delay` parameter
3. **No Data Downloaded**: Verify ASINs are valid and in correct format
4. **Memory Issues**: Reduce `--batch-size` for large datasets
5. **Missing Local Data**: Ensure HuggingFace data is downloaded before merging

### Debug Mode
Use `--verbose` flag for detailed logging:
```bash
python keepa.py download api-key.txt asins.txt --verbose
```

## Contributing

When modifying the codebase:
- Keep functions under 50 lines for modularity
- Add comprehensive docstrings
- Include error handling
- Use type hints
- Follow PEP 8 style guidelines 