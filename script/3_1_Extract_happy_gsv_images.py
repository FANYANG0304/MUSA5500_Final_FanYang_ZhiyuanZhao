#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter and copy GSV street view images for the 26 happiness points
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

def main():
    # Set paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    # Input paths
    sampling_points_path = data_dir / "sampling_points" / "sampling_points.shp"
    metadata_file = data_dir / "gsv_metadata" / "gsv_metadata.csv"
    gsv_images_dir = data_dir / "gsv_images"
    
    # Output paths
    happy_images_dir = data_dir / "gsv_images_happy"
    happy_images_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Extract Happiness Points GSV Street View Images")
    print("=" * 60)
    
    # Read sampling points data
    print(f"\n[1/4] Reading sampling points data...")
    try:
        sampling_points = gpd.read_file(sampling_points_path)
        print(f"  ✓ Read {len(sampling_points)} sampling points")
        
        # Filter happiness points
        happy_points = sampling_points[sampling_points['point_type'] == 'happy_point']
        print(f"  ✓ Filtered {len(happy_points)} happiness points")
        
        # Get happiness point IDs list
        happy_point_ids = set(happy_points['point_id'].tolist())
        print(f"  ✓ Happiness point IDs: {sorted(happy_point_ids)}")
        
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
        return
    
    # Read metadata
    print(f"\n[2/4] Reading GSV metadata...")
    try:
        metadata_df = pd.read_csv(metadata_file)
        print(f"  ✓ Read {len(metadata_df)} metadata records")
        
        # Filter metadata for happiness points
        happy_metadata = metadata_df[metadata_df['point_id'].isin(happy_point_ids)]
        print(f"  ✓ Found {len(happy_metadata)} happiness point metadata records")
        
        # Get pano_id list
        happy_pano_ids = happy_metadata['pano_id'].tolist()
        print(f"  ✓ Corresponding to {len(happy_pano_ids)} panorama IDs")
        
    except Exception as e:
        print(f"  ✗ Failed to read metadata: {e}")
        return
    
    # Check image files
    print(f"\n[3/4] Checking image files...")
    existing_images = []
    missing_images = []
    
    for pano_id in happy_pano_ids:
        image_file = gsv_images_dir / f"{pano_id}.jpg"
        if image_file.exists():
            existing_images.append(image_file)
        else:
            missing_images.append(pano_id)
    
    print(f"  ✓ Found {len(existing_images)} image files")
    if missing_images:
        print(f"  ⚠ Missing {len(missing_images)} image files")
        print(f"    Missing pano_ids: {missing_images[:5]}{'...' if len(missing_images) > 5 else ''}")
    
    # Copy images
    print(f"\n[4/4] Copying images to target folder...")
    print(f"  Target directory: {happy_images_dir}")
    
    copied_count = 0
    skipped_count = 0
    
    for image_file in existing_images:
        target_file = happy_images_dir / image_file.name
        
        if target_file.exists():
            skipped_count += 1
        else:
            try:
                shutil.copy2(image_file, target_file)
                copied_count += 1
            except Exception as e:
                print(f"  ✗ Copy failed {image_file.name}: {e}")
    
    print(f"  ✓ Newly copied: {copied_count} files")
    if skipped_count > 0:
        print(f"  ⊙ Skipped existing: {skipped_count} files")
    
    # Save happiness points metadata
    happy_metadata_file = data_dir / "gsv_metadata" / "gsv_metadata_happy.csv"
    happy_metadata.to_csv(happy_metadata_file, index=False, encoding='utf-8')
    print(f"  ✓ Happiness points metadata saved to: {happy_metadata_file}")
    
    # Output statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"Number of happiness points: {len(happy_points)}")
    print(f"Corresponding metadata: {len(happy_metadata)}")
    print(f"Image files: {len(existing_images)}/{len(happy_pano_ids)}")
    print(f"Copied to: {happy_images_dir}")
    print("=" * 60)
    
    # Show detailed info for each happiness point
    print(f"\nHappiness Points Details:")
    for idx, row in happy_metadata.iterrows():
        point_id = row['point_id']
        pano_id = row['pano_id']
        image_file = happy_images_dir / f"{pano_id}.jpg"
        status = "✓" if image_file.exists() else "✗"
        print(f"  {status} Point {point_id}: {pano_id}")
    
    print("\n✓ Program execution complete!")


if __name__ == "__main__":
    main()
