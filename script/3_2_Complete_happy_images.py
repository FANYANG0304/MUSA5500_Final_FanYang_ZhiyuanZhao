#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-acquire GSV metadata for happiness points
For points that failed to download, use original coordinates to re-query and try to find downloadable pano_ids
"""

import pandas as pd
import requests
from pathlib import Path
import time
from PIL import Image
from io import BytesIO
import shutil
import warnings
warnings.filterwarnings('ignore')


class GSVDownloader:
    def __init__(self, api_key):
        """Initialize GSV downloader"""
        self.api_key = api_key
        self.metadata_base_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    
    def get_metadata_with_radius(self, lat, lon, radius_list=[10, 50, 100, 200, 500]):
        """
        Get GSV metadata with different radii, prioritizing smaller radii
        """
        from math import radians, sin, cos, sqrt, atan2
        
        def calculate_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
        
        for radius in radius_list:
            params = {
                'location': f"{lat},{lon}",
                'radius': radius,
                'source': 'outdoor',  # Outdoor street view only
                'key': self.api_key
            }
            
            try:
                response = requests.get(self.metadata_base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'OK':
                        pano_id = data.get('pano_id')
                        
                        # Check if pano_id format is downloadable (doesn't start with CAoS)
                        if not pano_id.startswith('CAoS'):
                            found_lat = data['location']['lat']
                            found_lon = data['location']['lng']
                            distance = calculate_distance(lat, lon, found_lat, found_lon)
                            
                            return {
                                'pano_id': pano_id,
                                'lat': found_lat,
                                'lon': found_lon,
                                'date': data.get('date'),
                                'distance_m': round(distance, 2),
                                'search_radius': radius,
                                'status': 'OK'
                            }
                time.sleep(0.1)
            except Exception as e:
                continue
        
        return {'status': 'ZERO_RESULTS'}
    
    def download_panorama_tiles(self, pano_id, zoom=3, save_dir=None):
        """Download panorama tiles and stitch"""
        tile_width = 512
        tile_height = 512
        grid_sizes = {3: (7, 4)}
        cols, rows = grid_sizes.get(zoom, (7, 4))
        
        full_width = cols * tile_width
        full_height = rows * tile_height
        panorama = Image.new('RGB', (full_width, full_height))
        
        success_count = 0
        for y in range(rows):
            for x in range(cols):
                tile_url = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
                try:
                    response = requests.get(tile_url, timeout=10)
                    if response.status_code == 200:
                        tile = Image.open(BytesIO(response.content))
                        panorama.paste(tile, (x * tile_width, y * tile_height))
                        success_count += 1
                    time.sleep(0.02)
                except:
                    continue
        
        if success_count > 0:
            if save_dir:
                output_path = save_dir / f"{pano_id}.jpg"
                panorama.save(output_path, 'JPEG', quality=95)
                return output_path
        return None


def main():
    # Configuration
    API_KEY = "YOUR_API_KEY_HERE"
    
    # Set paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    happy_metadata_file = data_dir / "gsv_metadata" / "gsv_metadata_happy.csv"
    gsv_images_dir = data_dir / "gsv_images"
    happy_images_dir = data_dir / "gsv_images_happy"
    
    gsv_images_dir.mkdir(exist_ok=True, parents=True)
    happy_images_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("Re-acquire Happiness Points GSV Metadata and Download")
    print("=" * 70)
    
    # Read existing metadata
    print(f"\n[1/4] Reading existing metadata...")
    try:
        happy_metadata = pd.read_csv(happy_metadata_file)
        print(f"  ✓ Read {len(happy_metadata)} happiness point metadata records")
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
        return
    
    # Filter points with missing images
    print(f"\n[2/4] Checking points with missing images...")
    missing_points = []
    
    for idx, row in happy_metadata.iterrows():
        pano_id = row['pano_id']
        happy_image = happy_images_dir / f"{pano_id}.jpg"
        
        if not happy_image.exists():
            missing_points.append(row)
    
    print(f"  ✓ Found {len(missing_points)} points with missing images")
    
    if len(missing_points) == 0:
        print("\n✓ All images already exist!")
        return
    
    # Re-acquire metadata
    print(f"\n[3/4] Re-acquiring metadata...")
    print(f"  Strategy: Use original coordinates, search from small to large radius")
    print(f"  Search radii: 10m → 50m → 100m → 200m → 500m")
    print()
    
    downloader = GSVDownloader(api_key=API_KEY)
    
    success_list = []
    failed_list = []
    
    for idx, row in enumerate(missing_points):
        point_id = row['point_id']
        old_pano_id = row['pano_id']
        original_lat = row['original_lat']
        original_lon = row['original_lon']
        
        print(f"  [{idx+1}/{len(missing_points)}] Point {point_id}")
        print(f"    Original pano_id: {old_pano_id}")
        print(f"    Original coordinates: ({original_lat:.6f}, {original_lon:.6f})")
        
        # Re-acquire metadata
        new_metadata = downloader.get_metadata_with_radius(original_lat, original_lon)
        
        if new_metadata['status'] == 'OK':
            new_pano_id = new_metadata['pano_id']
            distance = new_metadata['distance_m']
            radius = new_metadata['search_radius']
            
            print(f"    ✓ Found new pano_id: {new_pano_id}")
            print(f"    Distance: {distance:.1f}m, Search radius: {radius}m")
            
            # Download immediately
            result = downloader.download_panorama_tiles(
                pano_id=new_pano_id,
                zoom=3,
                save_dir=gsv_images_dir
            )
            
            if result:
                print(f"    ✓ Download successful")
                # Copy to happy directory
                happy_image = happy_images_dir / f"{new_pano_id}.jpg"
                shutil.copy2(result, happy_image)
                print(f"    ✓ Copied to happy directory")
                
                success_list.append({
                    'point_id': point_id,
                    'old_pano_id': old_pano_id,
                    'new_pano_id': new_pano_id,
                    'distance_m': distance,
                    'search_radius': radius
                })
            else:
                print(f"    ✗ Download failed")
                failed_list.append({
                    'point_id': point_id,
                    'old_pano_id': old_pano_id,
                    'reason': 'Download failed'
                })
        else:
            print(f"    ✗ No available street view found")
            failed_list.append({
                'point_id': point_id,
                'old_pano_id': old_pano_id,
                'reason': 'No available street view'
            })
        
        print()
        time.sleep(0.3)
    
    # Update metadata file
    print(f"[4/4] Updating metadata...")
    
    if success_list:
        # Create mapping dictionary
        pano_id_map = {item['old_pano_id']: item['new_pano_id'] for item in success_list}
        
        # Update metadata
        for idx, row in happy_metadata.iterrows():
            old_pano_id = row['pano_id']
            if old_pano_id in pano_id_map:
                happy_metadata.at[idx, 'pano_id'] = pano_id_map[old_pano_id]
                happy_metadata.at[idx, 'old_pano_id'] = old_pano_id
        
        # Save updated metadata
        happy_metadata.to_csv(happy_metadata_file, index=False, encoding='utf-8')
        print(f"  ✓ Metadata updated")
        
        # Save replacement records
        replacement_file = data_dir / "gsv_metadata" / "pano_id_replacements.csv"
        pd.DataFrame(success_list).to_csv(replacement_file, index=False, encoding='utf-8')
        print(f"  ✓ Replacement records saved: {replacement_file}")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("Execution Results")
    print("=" * 70)
    print(f"Needed to process: {len(missing_points)}")
    print(f"Successfully acquired and downloaded: {len(success_list)}")
    print(f"Failed: {len(failed_list)}")
    
    if success_list:
        print(f"\nSuccessfully replaced pano_ids:")
        for item in success_list:
            print(f"  Point {item['point_id']}: "
                  f"{item['old_pano_id'][:20]}... → {item['new_pano_id'][:20]}... "
                  f"(distance:{item['distance_m']:.1f}m)")
    
    if failed_list:
        print(f"\nStill failed points:")
        for item in failed_list:
            print(f"  Point {item['point_id']}: {item['reason']}")
    
    # Final check
    print("\n" + "=" * 70)
    print("Final Completeness Check")
    print("=" * 70)
    
    final_complete = 0
    final_missing = 0
    
    for idx, row in happy_metadata.iterrows():
        pano_id = row['pano_id']
        happy_image = happy_images_dir / f"{pano_id}.jpg"
        
        if happy_image.exists():
            final_complete += 1
        else:
            final_missing += 1
            print(f"  ✗ Point {row['point_id']}: {pano_id}")
    
    print(f"\nTotal happiness points: {len(happy_metadata)}")
    print(f"Complete: {final_complete}")
    print(f"Still missing: {final_missing}")
    print(f"Completion rate: {final_complete/len(happy_metadata)*100:.1f}%")
    print("=" * 70)
    
    if final_missing == 0:
        print("\n✓ All happiness point images are complete!")
    elif final_missing < len(missing_points):
        print(f"\n✓ Successfully completed {len(missing_points) - final_missing} points!")
        print(f"⚠ But {final_missing} points still cannot acquire street view")
    else:
        print(f"\n✗ These points may truly have no street view coverage nearby")
    
    print("\n✓ Program execution complete!")


if __name__ == "__main__":
    main()
