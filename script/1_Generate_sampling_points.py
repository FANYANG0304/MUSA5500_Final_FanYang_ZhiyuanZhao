#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate street sampling points and happiness points merged dataset
Reference: Using Google Street View for Street-Level Urban Form Analysis
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    # Use relative paths
    # Assumes this script is in F:\MUSA5500\script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    output_dir = data_dir / "processed"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define target projected coordinate system - NAD 1983 / Pennsylvania South (ft)
    # EPSG:2272 corresponds to NAD83 / Pennsylvania South (ft)
    target_crs = "EPSG:2272"
    
    print("=" * 60)
    print("Street Sampling Points Generation Program")
    print("=" * 60)
    
    # 1. Read data
    print("\n[1/6] Reading data...")
    try:
        centerline = gpd.read_file(data_dir / "centerline" / "Street_Centerline.shp")
        print(f"  ✓ Road centerline data: {len(centerline)} records")
        
        philadelphia = gpd.read_file(data_dir / "philadelphia" / "philadelphia.shp")
        print(f"  ✓ Philadelphia boundary data: {len(philadelphia)} records")
        
        happiest = gpd.read_file(data_dir / "happiest" / "happiset.shp")
        print(f"  ✓ Happiness points data: {len(happiest)} points")
    except Exception as e:
        print(f"  ✗ Failed to read data: {e}")
        return
    
    # 2. Transform projections
    print("\n[2/6] Transforming projection to NAD 1983 / Pennsylvania South (ft)...")
    try:
        print(f"  - Road centerline original projection: {centerline.crs}")
        centerline = centerline.to_crs(target_crs)
        print(f"  ✓ Road centerline transformed")
        
        print(f"  - Philadelphia boundary original projection: {philadelphia.crs}")
        philadelphia = philadelphia.to_crs(target_crs)
        print(f"  ✓ Philadelphia boundary transformed")
        
        print(f"  - Happiness points original projection: {happiest.crs}")
        happiest = happiest.to_crs(target_crs)
        print(f"  ✓ Happiness points transformed")
    except Exception as e:
        print(f"  ✗ Projection transformation failed: {e}")
        return
    
    # 3. Clip road centerlines to Philadelphia boundary
    print("\n[3/6] Clipping road data to Philadelphia boundary...")
    try:
        centerline_clipped = gpd.clip(centerline, philadelphia)
        print(f"  ✓ Retained {len(centerline_clipped)} roads after clipping")
    except Exception as e:
        print(f"  ⚠ Clipping failed, using original data: {e}")
        centerline_clipped = centerline
    
    # 4. Generate sampling points every 200 meters along road centerlines
    print("\n[4/6] Generating sampling points along road centerlines...")
    print(f"  - Sampling interval: 200 meters (approximately 656.17 feet)")
    
    # Convert 200 meters to feet (1 meter = 3.28084 feet)
    interval_meters = 200
    interval_ft = interval_meters * 3.28084
    
    sample_points = []
    sample_attributes = []
    
    for idx, row in centerline_clipped.iterrows():
        line = row.geometry
        
        # Handle LineString
        if line.geom_type == 'LineString':
            line_length = line.length
            num_points = int(line_length / interval_ft) + 1
            
            for i in range(num_points):
                distance = i * interval_ft
                if distance <= line_length:
                    point = line.interpolate(distance)
                    sample_points.append(point)
                    sample_attributes.append({
                        'point_type': 'road_sample',
                        'street_id': idx,
                        'distance_ft': round(distance, 2),
                        'distance_m': round(distance / 3.28084, 2)
                    })
        
        # Handle MultiLineString
        elif line.geom_type == 'MultiLineString':
            for sub_idx, single_line in enumerate(line.geoms):
                line_length = single_line.length
                num_points = int(line_length / interval_ft) + 1
                
                for i in range(num_points):
                    distance = i * interval_ft
                    if distance <= line_length:
                        point = single_line.interpolate(distance)
                        sample_points.append(point)
                        sample_attributes.append({
                            'point_type': 'road_sample',
                            'street_id': f"{idx}_{sub_idx}",
                            'distance_ft': round(distance, 2),
                            'distance_m': round(distance / 3.28084, 2)
                        })
    
    # Create sampling points GeoDataFrame
    sample_gdf = gpd.GeoDataFrame(
        sample_attributes, 
        geometry=sample_points, 
        crs=target_crs
    )
    
    print(f"  ✓ Generated {len(sample_gdf)} road sampling points")
    
    # 5. Prepare happiness points data
    print("\n[5/6] Processing happiness points data...")
    happiest_processed = gpd.GeoDataFrame({
        'point_type': ['happy_point'] * len(happiest),
        'street_id': [None] * len(happiest),
        'distance_ft': [None] * len(happiest),
        'distance_m': [None] * len(happiest),
        'geometry': happiest.geometry
    }, crs=target_crs)
    
    print(f"  ✓ Processed {len(happiest_processed)} happiness points")
    
    # 6. Merge sampling points and happiness points
    print("\n[6/6] Merging data...")
    
    # Ensure column names are consistent
    common_cols = ['point_type', 'street_id', 'distance_ft', 'distance_m', 'geometry']
    
    # Merge data
    final_points = pd.concat([
        sample_gdf[common_cols],
        happiest_processed[common_cols]
    ], ignore_index=True)
    
    # Convert to GeoDataFrame
    final_points = gpd.GeoDataFrame(final_points, crs=target_crs)
    
    # Add unique ID
    final_points['point_id'] = range(1, len(final_points) + 1)
    
    # Add latitude/longitude coordinates (WGS84)
    final_points_wgs84 = final_points.to_crs("EPSG:4326")
    final_points['longitude'] = final_points_wgs84.geometry.x
    final_points['latitude'] = final_points_wgs84.geometry.y
    
    print(f"  ✓ Merge complete, total {len(final_points)} points")
    print(f"    - Road sampling points: {len(sample_gdf)}")
    print(f"    - Happiness points: {len(happiest_processed)}")
    
    # Save results
    output_file = output_dir / "sampling_points.shp"
    print(f"\nSaving results to: {output_file}")
    
    try:
        final_points.to_file(output_file, encoding='utf-8')
        print(f"  ✓ Save successful!")
        
        # Output statistics
        print("\n" + "=" * 60)
        print("Data Statistics")
        print("=" * 60)
        print(f"Total points: {len(final_points)}")
        print(f"Road sampling points: {sum(final_points['point_type'] == 'road_sample')}")
        print(f"Happiness points: {sum(final_points['point_type'] == 'happy_point')}")
        print(f"Projection CRS: {final_points.crs}")
        print(f"\nOutput file: {output_file}")
        print("=" * 60)
        
    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        return
    
    print("\n✓ Program execution complete!")

if __name__ == "__main__":
    main()
