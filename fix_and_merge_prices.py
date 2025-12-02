#!/usr/bin/env python3
"""
Fix prices.csv by removing comma separators and merge back into GetData_with_features.csv
"""

import pandas as pd
from pathlib import Path

# Configuration
output_dir = Path('UpdatedData')
prices_file = output_dir / 'prices.csv'
main_csv_file = output_dir / 'GetData_with_features.csv'

def parse_price(price_value):
    """
    Parse price string to float, handling comma separators
    """
    if price_value is None or pd.isna(price_value):
        return None
    
    # Convert to string if not already
    price_str = str(price_value).strip()
    
    # Remove quotes if present
    price_str = price_str.strip('"').strip("'")
    
    if not price_str or price_str == '':
        return None
    
    try:
        # Remove commas (thousands separators)
        cleaned_price = price_str.replace(',', '')
        
        # Convert to float
        return float(cleaned_price)
    except (ValueError, TypeError):
        return None

def fix_prices_csv():
    """
    Fix prices.csv by removing comma separators from prices
    """
    print("Reading prices.csv...")
    prices_df = pd.read_csv(prices_file, encoding='utf-8')
    print(f"Loaded {len(prices_df)} price records")
    
    # Count prices with commas before fixing
    prices_with_commas = prices_df['price'].astype(str).str.contains(',').sum()
    print(f"Found {prices_with_commas} prices with comma separators")
    
    # Fix prices by removing commas
    print("Fixing prices...")
    fixed_prices = []
    for idx, row in prices_df.iterrows():
        price = row['price']
        fixed_price = parse_price(price)
        
        if fixed_price is not None:
            fixed_prices.append(fixed_price)
        else:
            # Keep original if can't parse
            fixed_prices.append(price)
            print(f"Warning: Could not parse price for item {row['item']}: {price}")
    
    prices_df['price'] = fixed_prices
    
    # Save fixed prices
    print(f"Saving fixed prices to {prices_file}...")
    prices_df.to_csv(prices_file, index=False, encoding='utf-8')
    print(f"Saved {len(prices_df)} price records")
    
    return prices_df

def merge_prices_to_main(prices_df):
    """
    Merge fixed prices back into GetData_with_features.csv
    """
    print(f"\nReading {main_csv_file}...")
    main_df = pd.read_csv(main_csv_file, encoding='utf-8')
    print(f"Loaded {len(main_df)} items")
    
    # Check if price column exists
    if 'price' in main_df.columns:
        print("Price column already exists, will be updated")
    else:
        print("Price column does not exist, will be added")
    
    # Merge prices on 'item' column
    print("Merging prices...")
    main_df = main_df.merge(
        prices_df[['item', 'price']],
        on='item',
        how='left',
        suffixes=('', '_new')
    )
    
    # If price column existed, update it with new values (keeping old if new is NaN)
    if 'price_new' in main_df.columns:
        # Update price where new price exists
        main_df['price'] = main_df['price_new'].fillna(main_df['price'])
        main_df = main_df.drop(columns=['price_new'])
    
    # Count items with prices
    items_with_prices = main_df['price'].notna().sum()
    print(f"Items with prices after merge: {items_with_prices} ({items_with_prices/len(main_df)*100:.1f}%)")
    
    # Save merged CSV
    print(f"\nSaving merged data to {main_csv_file}...")
    main_df.to_csv(main_csv_file, index=False, encoding='utf-8')
    print(f"Saved {len(main_df)} items")
    
    return main_df

def main():
    print("="*60)
    print("Fix Prices and Merge to GetData_with_features.csv")
    print("="*60)
    
    # Fix prices.csv
    prices_df = fix_prices_csv()
    
    # Merge back to main CSV
    main_df = merge_prices_to_main(prices_df)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print(f"Total items: {len(main_df)}")
    print(f"Items with prices: {main_df['price'].notna().sum()}")

if __name__ == '__main__':
    main()

