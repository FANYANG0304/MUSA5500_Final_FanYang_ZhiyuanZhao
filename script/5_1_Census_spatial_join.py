#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download Census Tract data and perform spatial join with sampling points
Generate: Happiness points + Census + Semantic segmentation data table
          Other sampling points + Census + Semantic segmentation data table
"""

import requests
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CensusDownloader:
    """Census data downloader"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.census.gov/data"
        
        # Philadelphia: Pennsylvania (42) + Philadelphia County (101)
        self.state_fips = "42"
        self.county_fips = "101"
        
        # ACS 5-year estimate variable definitions
        self.variables = {
            'B01003_001E': ('total_pop', 'Total population'),
            'B01002_001E': ('median_age', 'Median age'),
            'B02001_002E': ('pop_white', 'White population'),
            'B02001_003E': ('pop_black', 'Black population'),
            'B02001_005E': ('pop_asian', 'Asian population'),
            'B03001_003E': ('pop_hispanic', 'Hispanic population'),
            'B19013_001E': ('median_income', 'Median household income'),
            'B17001_002E': ('pop_poverty', 'Population in poverty'),
            'B15003_001E': ('pop_25plus', 'Population 25 years and over'),
            'B15003_022E': ('pop_bachelor', 'Population with bachelor\'s degree'),
            'B15003_023E': ('pop_master', 'Population with master\'s degree'),
            'B15003_024E': ('pop_professional', 'Population with professional degree'),
            'B15003_025E': ('pop_doctorate', 'Population with doctorate'),
            'B25003_001E': ('total_housing', 'Total housing units'),
            'B25003_002E': ('owner_occupied', 'Owner-occupied units'),
            'B25077_001E': ('median_home_value', 'Median home value'),
            'B25064_001E': ('median_rent', 'Median rent'),
            'B23025_003E': ('labor_force', 'Labor force population'),
            'B23025_005E': ('unemployed', 'Unemployed population'),
        }
    
    def download_acs_data(self, year=2022):
        """Download ACS 5-year estimate data"""
        print(f"  Downloading {year} ACS 5-year estimate data...")
        
        var_codes = list(self.variables.keys())
        var_string = ','.join(var_codes)
        
        url = f"{self.base_url}/{year}/acs/acs5"
        params = {
            'get': f'NAME,{var_string}',
            'for': 'tract:*',
            'in': f'state:{self.state_fips} county:{self.county_fips}',
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"  ✗ API request failed: {e}")
            return None
        
        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)
        
        rename_dict = {'NAME': 'tract_name'}
        for var_code, (field_name, desc) in self.variables.items():
            if var_code in df.columns:
                rename_dict[var_code] = field_name
        
        df = df.rename(columns=rename_dict)
        df['GEOID'] = df['state'] + df['county'] + df['tract']
        
        # Convert numeric columns
        numeric_cols = [v[0] for v in self.variables.values()]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"  ✓ Download complete: {len(df)} Census Tracts")
        return df
    
    def calculate_derived_variables(self, df):
        """Calculate derived variables (ratios, etc.)"""
        print("  Calculating derived variables...")
        
        # Convert to float to allow NaN storage
        for col in ['total_pop', 'pop_25plus', 'total_housing', 'labor_force']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Race ratios (pandas handles division by zero automatically)
        df['pct_white'] = df['pop_white'] / df['total_pop'].where(df['total_pop'] > 0)
        df['pct_black'] = df['pop_black'] / df['total_pop'].where(df['total_pop'] > 0)
        df['pct_asian'] = df['pop_asian'] / df['total_pop'].where(df['total_pop'] > 0)
        df['pct_hispanic'] = df['pop_hispanic'] / df['total_pop'].where(df['total_pop'] > 0)
        
        # Poverty rate
        df['poverty_rate'] = df['pop_poverty'] / df['total_pop'].where(df['total_pop'] > 0)
        
        # Education ratio (bachelor's and above)
        df['pop_college_plus'] = (df['pop_bachelor'].fillna(0) + 
                                   df['pop_master'].fillna(0) + 
                                   df['pop_professional'].fillna(0) + 
                                   df['pop_doctorate'].fillna(0))
        df['pct_college'] = df['pop_college_plus'] / df['pop_25plus'].where(df['pop_25plus'] > 0)
        
        # Owner-occupancy rate
        df['pct_owner_occupied'] = df['owner_occupied'] / df['total_housing'].where(df['total_housing'] > 0)
        
        # Unemployment rate
        df['unemployment_rate'] = df['unemployed'] / df['labor_force'].where(df['labor_force'] > 0)
        
        print("  ✓ Derived variables calculated")
        return df
    
    def download_tract_boundaries(self, year=2022):
        """Download Census Tract boundary shapefile"""
        print(f"  Downloading {year} Census Tract boundaries...")
        
        url = (f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
               f"tl_{year}_{self.state_fips}_tract.zip")
        
        try:
            gdf = gpd.read_file(url)
            gdf = gdf[gdf['COUNTYFP'] == self.county_fips].copy()
            print(f"  ✓ Download complete: {len(gdf)} Tract boundaries")
            return gdf
        except Exception as e:
            print(f"  ✗ Boundary download failed: {e}")
            return None


def spatial_join_points_to_tracts(points_gdf, tracts_gdf):
    """Spatial join points data to Census Tracts"""
    if points_gdf.crs != tracts_gdf.crs:
        points_gdf = points_gdf.to_crs(tracts_gdf.crs)
    
    joined = gpd.sjoin(points_gdf, tracts_gdf, how='left', predicate='within')
    return joined


def main():
    # ============ Configuration ============
    CENSUS_API_KEY = "YOUR_CENSUS_API_KEY_HERE"
    CENSUS_YEAR = 2022
    # =======================================
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    sampling_points_path = data_dir / "sampling_points" / "sampling_points.shp"
    happy_seg_file = data_dir / "semantic_segmentation_happy" / "segmentation_results.csv"
    all_seg_file = data_dir / "semantic_segmentation_all" / "segmentation_results.csv"
    
    output_dir = data_dir / "analysis_data"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    census_dir = data_dir / "census"
    census_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("Census Data Download and Spatial Join")
    print("=" * 70)
    
    # 1. Download Census data
    print(f"\n[1/6] Downloading Census ACS data...")
    downloader = CensusDownloader(api_key=CENSUS_API_KEY)
    census_df = downloader.download_acs_data(year=CENSUS_YEAR)
    
    if census_df is None:
        print("  ✗ Census data download failed!")
        return
    
    # 2. Calculate derived variables
    print(f"\n[2/6] Calculating derived variables...")
    census_df = downloader.calculate_derived_variables(census_df)
    
    census_file = census_dir / f"census_acs_{CENSUS_YEAR}.csv"
    census_df.to_csv(census_file, index=False, encoding='utf-8')
    print(f"  ✓ Census data saved: {census_file}")
    
    # 3. Download Tract boundaries
    print(f"\n[3/6] Downloading Census Tract boundaries...")
    tracts_gdf = downloader.download_tract_boundaries(year=CENSUS_YEAR)
    
    if tracts_gdf is None:
        print("  ✗ Tract boundary download failed!")
        return
    
    tracts_gdf = tracts_gdf.merge(census_df, on='GEOID', how='left')
    
    tracts_file = census_dir / f"census_tracts_{CENSUS_YEAR}.shp"
    tracts_gdf.to_file(tracts_file, encoding='utf-8')
    print(f"  ✓ Tract boundaries saved: {tracts_file}")
    
    # 4. Read sampling points
    print(f"\n[4/6] Reading sampling points data...")
    try:
        points_gdf = gpd.read_file(sampling_points_path)
        print(f"  ✓ Read {len(points_gdf)} sampling points")
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
        return
    
    # 5. Spatial join
    print(f"\n[5/6] Executing spatial join...")
    joined_gdf = spatial_join_points_to_tracts(points_gdf, tracts_gdf)
    print(f"  ✓ Spatial join complete")
    
    # 6. Merge semantic segmentation results
    print(f"\n[6/6] Merging semantic segmentation results...")
    
    census_fields = [
        'GEOID', 'tract_name', 'total_pop', 'median_age',
        'pct_white', 'pct_black', 'pct_asian', 'pct_hispanic',
        'median_income', 'poverty_rate', 'pct_college',
        'pct_owner_occupied', 'median_home_value', 'median_rent',
        'unemployment_rate'
    ]
    
    point_fields = ['point_id', 'point_type', 'longitude', 'latitude']
    
    seg_fields = [
        'sky_ratio', 'green_view_index', 'building_ratio', 
        'road_ratio', 'vehicle_ratio', 'person_ratio',
        'tree_ratio', 'grass_ratio', 'sidewalk_ratio', 'car_ratio'
    ]
    
    keep_fields = point_fields + [f for f in census_fields if f in joined_gdf.columns]
    merged_df = joined_gdf[keep_fields].copy()
    merged_df = pd.DataFrame(merged_df.drop(columns='geometry', errors='ignore'))
    
    happy_points = merged_df[merged_df['point_type'] == 'happy_point'].copy()
    road_points = merged_df[merged_df['point_type'] == 'road_sample'].copy()
    
    print(f"  Happiness points: {len(happy_points)}")
    print(f"  Road sampling points: {len(road_points)}")
    
    # Merge happiness points semantic segmentation results
    if happy_seg_file.exists():
        happy_seg_df = pd.read_csv(happy_seg_file)
        happy_final = happy_points.merge(
            happy_seg_df[['point_id'] + seg_fields], 
            on='point_id', 
            how='left'
        )
        happy_final['is_happy'] = 1
        
        happy_output = output_dir / "happy_points_full.csv"
        happy_final.to_csv(happy_output, index=False, encoding='utf-8')
        print(f"  ✓ Happiness points data saved: {happy_output}")
    else:
        print(f"  ⚠ Happiness points semantic segmentation results not found")
        happy_final = None
    
    # Merge other sampling points semantic segmentation results
    if all_seg_file.exists():
        all_seg_df = pd.read_csv(all_seg_file)
        road_final = road_points.merge(
            all_seg_df[['point_id'] + seg_fields], 
            on='point_id', 
            how='left'
        )
        road_final['is_happy'] = 0
        
        road_output = output_dir / "road_points_full.csv"
        road_final.to_csv(road_output, index=False, encoding='utf-8')
        print(f"  ✓ Road sampling points data saved: {road_output}")
    else:
        print(f"  ⚠ All semantic segmentation results not found")
        road_final = None
    
    # Merge into complete table
    if happy_final is not None and road_final is not None:
        combined = pd.concat([happy_final, road_final], ignore_index=True)
        combined_output = output_dir / "all_points_full.csv"
        combined.to_csv(combined_output, index=False, encoding='utf-8')
        print(f"  ✓ Complete dataset saved: {combined_output}")
        print(f"    Total records: {len(combined)}")
    
    # Data overview
    print("\n" + "=" * 70)
    print("Data Overview")
    print("=" * 70)
    print(f"Census Tract count: {len(census_df)}")
    print(f"Total population: {census_df['total_pop'].sum():,.0f}")
    
    print("\nOutput files:")
    print(f"  {census_dir}")
    print(f"  {output_dir}")
    
    print("\n" + "=" * 70)
    print("✓ Program execution complete!")


if __name__ == "__main__":
    main()
