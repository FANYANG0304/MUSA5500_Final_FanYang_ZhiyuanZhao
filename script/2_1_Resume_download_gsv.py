#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single script for automatic 20-process parallel GSV image download
Run once to automatically start 20 parallel download processes
"""

import pandas as pd
import requests
from pathlib import Path
import time
from PIL import Image
from io import BytesIO
import multiprocessing
from multiprocessing import Process, Queue
import warnings
warnings.filterwarnings('ignore')


class GSVImageDownloader:
    def __init__(self):
        """Initialize GSV image downloader"""
        pass
    
    def download_panorama_tiles(self, pano_id, zoom=3, save_dir=None):
        """
        Download panorama tiles and stitch them
        zoom: Zoom level (0-5), zoom=3 produces image size approximately 3328x1664
        """
        tile_width = 512
        tile_height = 512
        
        grid_sizes = {
            0: (1, 1),
            1: (2, 1),
            2: (4, 2),
            3: (7, 4),
            4: (13, 7),
            5: (26, 13)
        }
        
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
                except Exception as e:
                    continue
        
        if success_count > 0:
            if save_dir:
                output_path = save_dir / f"{pano_id}.jpg"
                panorama.save(output_path, 'JPEG', quality=95)
                return output_path
            return panorama
        return None


def worker_process(worker_id, tasks, output_dir, result_queue):
    """
    Worker process function
    """
    downloader = GSVImageDownloader()
    
    download_count = 0
    failed_count = 0
    start_time = time.time()
    
    print(f"Worker {worker_id}: Starting to process {len(tasks)} tasks")
    
    for idx, item in enumerate(tasks):
        pano_id = item['pano_id']
        point_id = item['point_id']
        
        # Show progress every 100 items
        if (idx + 1) % 100 == 0 or idx == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1) if idx > 0 else 0
            remaining = avg_time * (len(tasks) - idx - 1)
            remaining_min = int(remaining / 60)
            remaining_sec = int(remaining % 60)
            progress_pct = (idx + 1) / len(tasks) * 100
            print(f"Worker {worker_id}: [{idx+1:5d}/{len(tasks):5d}] {progress_pct:5.1f}% "
                  f"Success:{download_count:5d} Failed:{failed_count:4d} "
                  f"Remaining:{remaining_min:3d}m{remaining_sec:02d}s")
        
        # Download panorama
        try:
            result = downloader.download_panorama_tiles(
                pano_id=pano_id,
                zoom=3,
                save_dir=output_dir
            )
            
            if result:
                download_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
        
        time.sleep(0.05)  # Avoid requesting too fast
    
    # Completion statistics
    total_time = time.time() - start_time
    result = {
        'worker_id': worker_id,
        'total_tasks': len(tasks),
        'success': download_count,
        'failed': failed_count,
        'time': total_time
    }
    
    result_queue.put(result)
    print(f"Worker {worker_id}: ✓ Complete! Success:{download_count} Failed:{failed_count} "
          f"Time:{int(total_time/60)}m{int(total_time%60)}s")


def main():
    # Configuration
    NUM_WORKERS = 20  # Number of parallel processes
    
    # Set paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    metadata_file = data_dir / "gsv_metadata" / "gsv_metadata.csv"
    output_dir = data_dir / "gsv_images"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print(f"Automatic 20-Process Parallel GSV Image Download")
    print("=" * 70)
    
    # Read metadata
    print(f"\n[1/5] Reading metadata...")
    try:
        metadata_df = pd.read_csv(metadata_file)
        print(f"  ✓ Read {len(metadata_df)} metadata records")
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
        return
    
    # Check already downloaded images (resume capability)
    print(f"\n[2/5] Checking download progress (resume capability)...")
    already_downloaded = 0
    to_download = []
    
    for idx, row in metadata_df.iterrows():
        pano_id = row['pano_id']
        image_file = output_dir / f"{pano_id}.jpg"
        
        if image_file.exists():
            already_downloaded += 1
        else:
            to_download.append({
                'pano_id': pano_id,
                'point_id': row.get('point_id', idx),
                'index': idx
            })
    
    total_images = len(metadata_df)
    need_download = len(to_download)
    
    print(f"  ✓ Already downloaded: {already_downloaded} images")
    print(f"  ✓ Pending download: {need_download} images")
    print(f"  ✓ Completion rate: {already_downloaded/total_images*100:.1f}%")
    
    if need_download == 0:
        print("\n✓ All images already downloaded!")
        return
    
    # Distribute tasks
    print(f"\n[3/5] Distributing tasks to {NUM_WORKERS} processes...")
    
    # Calculate tasks per worker
    tasks_per_worker = need_download // NUM_WORKERS
    remainder = need_download % NUM_WORKERS
    
    worker_tasks = []
    start_idx = 0
    
    for worker_id in range(NUM_WORKERS):
        # First 'remainder' workers get one extra task
        if worker_id < remainder:
            end_idx = start_idx + tasks_per_worker + 1
        else:
            end_idx = start_idx + tasks_per_worker
        
        tasks = to_download[start_idx:end_idx]
        worker_tasks.append(tasks)
        
        print(f"  Worker {worker_id:2d}: {len(tasks):5d} tasks "
              f"(index {start_idx:5d} - {end_idx-1:5d})")
        
        start_idx = end_idx
    
    # Start parallel download
    print(f"\n[4/5] Starting {NUM_WORKERS} parallel download processes...")
    print(f"  Target directory: {output_dir}")
    print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create result queue
    result_queue = Queue()
    
    # Create and start all worker processes
    processes = []
    overall_start_time = time.time()
    
    for worker_id in range(NUM_WORKERS):
        p = Process(
            target=worker_process,
            args=(worker_id, worker_tasks[worker_id], output_dir, result_queue)
        )
        p.start()
        processes.append(p)
        time.sleep(0.5)  # Stagger start times
    
    print(f"✓ Started {NUM_WORKERS} processes")
    print(f"Downloading, please wait...\n")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    print(f"\n[5/5] Collecting statistics...")
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Calculate overall statistics
    total_time = time.time() - overall_start_time
    total_success = sum(r['success'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_processed = sum(r['total_tasks'] for r in results)
    
    # Output detailed results
    print("\n" + "=" * 70)
    print("Download Complete - Detailed Statistics")
    print("=" * 70)
    
    # Sort by worker_id
    results.sort(key=lambda x: x['worker_id'])
    
    print(f"\n{'Worker':>6} {'Tasks':>8} {'Success':>8} {'Failed':>8} {'Time':>12}")
    print("-" * 70)
    
    for r in results:
        time_str = f"{int(r['time']/60)}m{int(r['time']%60)}s"
        print(f"{r['worker_id']:6d} {r['total_tasks']:8d} {r['success']:8d} "
              f"{r['failed']:8d} {time_str:>12}")
    
    print("-" * 70)
    total_time_str = f"{int(total_time/60)}m{int(total_time%60)}s"
    print(f"{'Total':>6} {total_processed:8d} {total_success:8d} "
          f"{total_failed:8d} {total_time_str:>12}")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("Overall Statistics")
    print("=" * 70)
    print(f"Total images: {total_images}")
    print(f"Previously downloaded: {already_downloaded}")
    print(f"Downloaded this run: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Current total: {already_downloaded + total_success}")
    print(f"Completion rate: {(already_downloaded + total_success)/total_images*100:.1f}%")
    print(f"Total time: {int(total_time/60)}m{int(total_time%60)}s")
    
    if total_processed > 0:
        print(f"Success rate this run: {total_success/total_processed*100:.1f}%")
        avg_speed = total_time / total_success if total_success > 0 else 0
        print(f"Average speed: {avg_speed:.2f} sec/image")
    
    print(f"\nImages saved to: {output_dir}")
    print("=" * 70)
    
    # If there are failures, prompt to re-run
    if total_failed > 0:
        print(f"\n⚠ {total_failed} images failed to download")
        print(f"You can re-run this script to continue downloading")
    
    print("\n✓ Program execution complete!")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
