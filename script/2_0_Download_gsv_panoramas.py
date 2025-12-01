#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download Google Street View Panoramas
Reference: Using Google Street View for Street-Level Urban Form Analysis
"""

import geopandas as gpd
import pandas as pd
import requests
from pathlib import Path
import time
from PIL import Image
from io import BytesIO
import json
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

class GSVDownloader:
    def __init__(self, api_key=None):
        """
        Initialize GSV downloader
        api_key: Google Maps API key (optional, but recommended to avoid rate limits)
        """
        self.api_key = api_key
        self.metadata_base_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        self.image_base_url = "https://maps.googleapis.com/maps/api/streetview"
        # Legacy API (used in the reference paper)
        self.old_metadata_url = "http://maps.google.com/cbk?output=xml"
        
    def get_metadata_old(self, lat, lon):
        """
        Get GSV metadata using legacy API (method from the paper)
        """
        url = f"{self.old_metadata_url}&ll={lat},{lon}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                pano_id = root.find('.//pano_id')
                pano_yaw = root.find('.//pano_yaw_deg')
                data_properties = root.find('.//data_properties')
                
                if pano_id is not None:
                    metadata = {
                        'pano_id': pano_id.text,
                        'pano_yaw_deg': float(pano_yaw.text) if pano_yaw is not None else 0,
                        'lat': lat,
                        'lon': lon,
                        'status': 'OK'
                    }
                    
                    if data_properties is not None:
                        image_date = data_properties.get('image_date')
                        if image_date:
                            metadata['date'] = image_date
                    
                    return metadata
            return {'status': 'ZERO_RESULTS'}
        except Exception as e:
            print(f"    Failed to get metadata: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def get_metadata(self, lat, lon):
        """
        Get GSV metadata using new API
        """
        if self.api_key:
            params = {
                'location': f"{lat},{lon}",
                'key': self.api_key
            }
            try:
                response = requests.get(self.metadata_base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'OK':
                        return {
                            'pano_id': data.get('pano_id'),
                            'lat': data['location']['lat'],
                            'lon': data['location']['lng'],
                            'date': data.get('date'),
                            'status': 'OK'
                        }
                return {'status': data.get('status', 'ERROR')}
            except Exception as e:
                print(f"    Failed to get metadata: {e}")
                return {'status': 'ERROR', 'error': str(e)}
        else:
            # If no API key, use legacy API
            return self.get_metadata_old(lat, lon)
    
    def download_panorama_tiles(self, pano_id, zoom=3, save_dir=None):
        """
        Download panorama tiles and stitch them
        zoom: Zoom level (0-5, higher means clearer but longer download time)
              zoom=3 produces image size approximately 3328x1664
        """
        # Calculate tile grid size
        tile_width = 512
        tile_height = 512
        
        # Grid sizes for different zoom levels
        grid_sizes = {
            0: (1, 1),
            1: (2, 1),
            2: (4, 2),
            3: (7, 4),
            4: (13, 7),
            5: (26, 13)
        }
        
        cols, rows = grid_sizes.get(zoom, (7, 4))
        
        # Create blank canvas
        full_width = cols * tile_width
        full_height = rows * tile_height
        panorama = Image.new('RGB', (full_width, full_height))
        
        # Download and stitch tiles
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
                    time.sleep(0.02)  # Avoid requesting too fast
                except Exception as e:
                    print(f"      Tile ({x},{y}) download failed: {e}")
                    continue
        
        if success_count > 0:
            if save_dir:
                output_path = save_dir / f"{pano_id}.jpg"
                panorama.save(output_path, 'JPEG', quality=95)
                return output_path
            return panorama
        return None
    
    def download_panorama_api(self, lat, lon, heading=0, fov=90, pitch=0, size="640x640", save_dir=None, filename=None):
        """
        Download image directly using Street View API (requires API key)
        This method is simpler but image quality and field of view are limited
        """
        if not self.api_key:
            print("    API key required for this method")
            return None
        
        params = {
            'size': size,
            'location': f"{lat},{lon}",
            'heading': heading,
            'fov': fov,
            'pitch': pitch,
            'key': self.api_key
        }
        
        try:
            response = requests.get(self.image_base_url, params=params, timeout=15)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                if save_dir and filename:
                    output_path = save_dir / filename
                    image.save(output_path, 'JPEG', quality=95)
                    return output_path
                return image
        except Exception as e:
            print(f"    Failed to download image: {e}")
        return None


def main():
    # Set paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    # Input/output paths
    sampling_points_path = data_dir / "sampling_points" / "sampling_points.shp"
    output_dir = data_dir / "gsv_images"
    metadata_dir = data_dir / "gsv_metadata"
    
    # Create output directories
    output_dir.mkdir(exist_ok=True, parents=True)
    metadata_dir.mkdir(exist_ok=True, parents=True)
    
    # Google API Key
    API_KEY = "YOUR_API_KEY_HERE"
    
    print("=" * 60)
    print("Google Street View Panorama Download Program")
    print("=" * 60)
    
    # Read sampling points
    print(f"\n[1/5] Reading sampling points data...")
    try:
        points = gpd.read_file(sampling_points_path)
        # Convert to WGS84 to get lat/lon
        points_wgs84 = points.to_crs("EPSG:4326")
        print(f"  ✓ Read {len(points)} sampling points")
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
        return
    
    # Initialize downloader
    print(f"\n[2/5] Initializing GSV downloader...")
    downloader = GSVDownloader(api_key=API_KEY)
    if API_KEY:
        print(f"  ✓ Using API Key mode")
    else:
        print(f"  ✓ Using free mode (legacy API, may have limitations)")
        print(f"  ⚠ Recommend applying for API Key for more stable service")
    
    # Get metadata
    print(f"\n[3/5] Getting GSV metadata...")
    
    # Check if metadata file exists (resume capability)
    metadata_file = metadata_dir / "gsv_metadata.csv"
    if metadata_file.exists():
        print(f"  Found existing metadata file, loading...")
        existing_metadata = pd.read_csv(metadata_file)
        existing_point_ids = set(existing_metadata['point_id'].tolist())
        print(f"  ✓ Found {len(existing_metadata)} existing metadata records")
        metadata_list = existing_metadata.to_dict('records')
    else:
        existing_point_ids = set()
        metadata_list = []
    
    success_count = len(existing_point_ids)
    skipped_count = 0
    
    for idx, row in points_wgs84.iterrows():
        lat = row.geometry.y
        lon = row.geometry.x
        point_id = row.get('point_id', idx)
        
        # Skip points with existing metadata
        if point_id in existing_point_ids:
            skipped_count += 1
            if skipped_count % 1000 == 0:
                print(f"  Skipped {skipped_count} points with existing metadata...")
            continue
        
        if (idx + 1 - skipped_count) % 100 == 0:
            print(f"  [{idx+1}/{len(points)}] Processed {idx+1-skipped_count} new points...")
        
        metadata = downloader.get_metadata(lat, lon)
        
        if metadata['status'] == 'OK':
            metadata['point_id'] = point_id
            metadata['original_lat'] = lat
            metadata['original_lon'] = lon
            metadata_list.append(metadata)
            success_count += 1
        
        time.sleep(0.05)  # Avoid requesting too fast
        
        # Save every 1000 points (prevent data loss on interruption)
        if (idx + 1) % 1000 == 0:
            temp_df = pd.DataFrame(metadata_list)
            temp_df.to_csv(metadata_file, index=False, encoding='utf-8')
            print(f"  ✓ Saved {len(metadata_list)} metadata records (checkpoint)")
    
    print(f"\n  Total: {success_count}/{len(points)} points have available GSV images")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count} points with existing metadata")
        print(f"  Newly acquired: {success_count - len(existing_point_ids)} point metadata")
    
    # Save metadata
    if metadata_list:
        metadata_df = pd.DataFrame(metadata_list)
        metadata_file = metadata_dir / "gsv_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False, encoding='utf-8')
        print(f"  ✓ Metadata saved to: {metadata_file}")
    
    # Download panoramas
    print(f"\n[4/5] Downloading panoramas...")
    print(f"  Target directory: {output_dir}")
    print(f"  Download mode: Tile stitching (high quality)")
    print(f"  Zoom level: 3 (image size approximately 3328x1664)")
    
    download_count = 0
    skip_count = 0
    
    for idx, meta in enumerate(metadata_list):
        pano_id = meta['pano_id']
        point_id = meta['point_id']
        
        # Show progress every 100 images
        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(metadata_list)}] Processed {download_count} successful, {skip_count} already exist")
        
        # Check if already exists
        output_file = output_dir / f"{pano_id}.jpg"
        if output_file.exists():
            skip_count += 1
            download_count += 1
            continue
        
        # Download panorama
        result = downloader.download_panorama_tiles(
            pano_id=pano_id,
            zoom=3,
            save_dir=output_dir
        )
        
        if result:
            download_count += 1
        
        time.sleep(0.1)  # Avoid requesting too fast
    
    # Update metadata with download status
    print(f"\n[5/5] Updating metadata...")
    for meta in metadata_list:
        image_file = output_dir / f"{meta['pano_id']}.jpg"
        meta['image_downloaded'] = image_file.exists()
        meta['image_path'] = str(image_file) if image_file.exists() else None
    
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(metadata_dir / "gsv_metadata.csv", index=False, encoding='utf-8')
    print(f"  ✓ Metadata updated")
    
    # Output statistics
    print("\n" + "=" * 60)
    print("Download Statistics")
    print("=" * 60)
    print(f"Total sampling points: {len(points)}")
    print(f"With GSV images: {success_count}")
    print(f"Successfully downloaded: {download_count}")
    print(f"Skipped (existing): {skip_count}")
    print(f"Coverage rate: {success_count/len(points)*100:.1f}%")
    print(f"Download rate: {download_count/success_count*100:.1f}%" if success_count > 0 else "")
    print(f"\nImages saved to: {output_dir}")
    print(f"Metadata file: {metadata_dir / 'gsv_metadata.csv'}")
    print("=" * 60)
    
    print("\n✓ Program execution complete!")


if __name__ == "__main__":
    main()
