#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSV Panorama Semantic Segmentation - GPU Batch Processing Accelerated Version
Uses batch processing + multi-threaded data loading to maximize GPU utilization

Install dependencies:
    pip install transformers --break-system-packages
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Error: Please install transformers library first")


class GSVDataset(Dataset):
    """GSV image dataset, supports multi-threaded loading"""
    
    def __init__(self, image_list, processor):
        """
        Parameters:
            image_list: [{'pano_id': ..., 'point_id': ..., 'image_path': ...}, ...]
            processor: SegformerImageProcessor
        """
        self.image_list = image_list
        self.processor = processor
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        item = self.image_list[idx]
        image_path = item['image_path']
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            # Return empty data, filter later
            return {
                'pixel_values': torch.zeros(3, 512, 512),
                'pano_id': item['pano_id'],
                'point_id': item['point_id'],
                'valid': False,
                'height': 0,
                'width': 0
            }
        
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'pano_id': item['pano_id'],
            'point_id': item['point_id'],
            'valid': True,
            'height': h,
            'width': w
        }


def collate_fn(batch):
    """Custom batch collation function, filters invalid data"""
    # Filter invalid data
    valid_batch = [item for item in batch if item['valid']]
    
    if len(valid_batch) == 0:
        return None
    
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in valid_batch]),
        'pano_ids': [item['pano_id'] for item in valid_batch],
        'point_ids': [item['point_id'] for item in valid_batch],
        'heights': [item['height'] for item in valid_batch],
        'widths': [item['width'] for item in valid_batch]
    }


class BatchSegmentation:
    """Batch processing semantic segmentation model"""
    
    def __init__(self, batch_size=8, num_workers=4):
        """
        Parameters:
            batch_size: Batch size, adjust based on GPU memory (recommend 8-16 for 8GB VRAM)
            num_workers: Number of data loading threads
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        print("Initializing batch segmentation model...")
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ VRAM: {gpu_mem:.1f} GB")
            
            # Auto-adjust batch_size based on VRAM
            if gpu_mem < 6:
                self.batch_size = min(batch_size, 4)
            elif gpu_mem < 10:
                self.batch_size = min(batch_size, 8)
            else:
                self.batch_size = min(batch_size, 16)
            
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
            self.batch_size = 1
            print("  ⚠ No GPU detected")
        
        print(f"  ✓ Batch size: {self.batch_size}")
        print(f"  ✓ Data loading threads: {self.num_workers}")
        
        # Load model
        print("  Loading SegFormer-B0 model...")
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("  ✓ Model loaded successfully")
        
        # Class definitions
        self.sky_classes = [2]
        self.vegetation_classes = [4, 9, 17, 66, 72]
        self.building_classes = [1, 25, 48, 84]
        self.road_classes = [6, 11, 52]
        self.vehicle_classes = [20, 80, 83, 102, 127]
        self.person_classes = [12]
    
    def calculate_ratios_batch(self, predictions, heights, widths):
        """Batch calculate ratios"""
        results = []
        
        for pred, h, w in zip(predictions, heights, widths):
            # Upsample to original size
            pred_resized = cv2.resize(pred.astype(np.uint8), (w, h), 
                                      interpolation=cv2.INTER_NEAREST)
            
            total_pixels = pred_resized.size
            
            ratios = {
                'sky_ratio': np.sum(np.isin(pred_resized, self.sky_classes)) / total_pixels,
                'green_view_index': np.sum(np.isin(pred_resized, self.vegetation_classes)) / total_pixels,
                'building_ratio': np.sum(np.isin(pred_resized, self.building_classes)) / total_pixels,
                'road_ratio': np.sum(np.isin(pred_resized, self.road_classes)) / total_pixels,
                'vehicle_ratio': np.sum(np.isin(pred_resized, self.vehicle_classes)) / total_pixels,
                'person_ratio': np.sum(np.isin(pred_resized, self.person_classes)) / total_pixels,
                'tree_ratio': np.sum(pred_resized == 4) / total_pixels,
                'grass_ratio': np.sum(pred_resized == 9) / total_pixels,
                'sidewalk_ratio': np.sum(pred_resized == 11) / total_pixels,
                'car_ratio': np.sum(pred_resized == 20) / total_pixels,
            }
            results.append(ratios)
        
        return results
    
    def process_batch(self, batch):
        """Process one batch"""
        pixel_values = batch['pixel_values'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
        
        # Get predictions
        predictions = logits.argmax(dim=1).cpu().numpy()
        
        # Calculate ratios
        ratios_list = self.calculate_ratios_batch(
            predictions, 
            batch['heights'], 
            batch['widths']
        )
        
        return ratios_list


def main():
    """Main function"""
    
    if not HAS_TRANSFORMERS:
        print("Please install transformers library first!")
        return
    
    # ============ Adjustable Parameters ============
    BATCH_SIZE = 8      # Batch size: recommend 8 for 8GB VRAM, 12 for 12GB, 16 for 24GB
    NUM_WORKERS = 4     # Data loading threads: recommend half of CPU cores
    SAVE_INTERVAL = 500 # Save interval
    # ===============================================
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    gsv_images_dir = data_dir / "gsv_images"
    metadata_file = data_dir / "gsv_metadata" / "gsv_metadata.csv"
    
    output_dir = data_dir / "semantic_segmentation_all"
    output_dir.mkdir(exist_ok=True, parents=True)
    results_file = output_dir / "segmentation_results.csv"
    
    print("=" * 70)
    print("GSV Semantic Segmentation - GPU Batch Processing Accelerated")
    print("=" * 70)
    
    # Read metadata
    print(f"\n[1/5] Reading metadata...")
    try:
        metadata_df = pd.read_csv(metadata_file)
        print(f"  ✓ Read {len(metadata_df)} records")
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
        return
    
    # Resume capability check
    print(f"\n[2/5] Checking processed results...")
    processed_pano_ids = set()
    existing_results = []
    
    if results_file.exists():
        try:
            existing_df = pd.read_csv(results_file)
            processed_pano_ids = set(existing_df['pano_id'].tolist())
            existing_results = existing_df.to_dict('records')
            print(f"  ✓ Already processed: {len(processed_pano_ids)} images")
        except:
            pass
    
    # Filter images to process
    print(f"\n[3/5] Checking images to process...")
    to_process = []
    
    for idx, row in metadata_df.iterrows():
        pano_id = row['pano_id']
        if pano_id in processed_pano_ids:
            continue
        
        image_file = gsv_images_dir / f"{pano_id}.jpg"
        if image_file.exists():
            to_process.append({
                'pano_id': pano_id,
                'point_id': row['point_id'],
                'image_path': image_file
            })
    
    print(f"  ✓ Pending: {len(to_process)} images")
    
    if not to_process:
        print("\n✓ All images already processed!")
        return
    
    # Initialize model
    print(f"\n[4/5] Initializing model...")
    segmenter = BatchSegmentation(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    # Create data loader
    print(f"\n[5/5] Creating data loader...")
    dataset = GSVDataset(to_process, segmenter.processor)
    dataloader = DataLoader(
        dataset,
        batch_size=segmenter.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )
    
    total_batches = len(dataloader)
    print(f"  ✓ Total batches: {total_batches}")
    
    # Batch processing
    print(f"\nStarting batch processing...")
    print(f"  Output file: {results_file}\n")
    
    results = existing_results.copy()
    start_time = time.time()
    processed_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        batch_start = time.time()
        
        # Batch inference
        ratios_list = segmenter.process_batch(batch)
        
        # Save results
        for pano_id, point_id, ratios in zip(batch['pano_ids'], batch['point_ids'], ratios_list):
            result = {'point_id': point_id, 'pano_id': pano_id}
            result.update(ratios)
            results.append(result)
            processed_count += 1
        
        # Show progress
        elapsed = time.time() - start_time
        batch_time = time.time() - batch_start
        speed = processed_count / elapsed if elapsed > 0 else 0
        remaining = (len(to_process) - processed_count) / speed if speed > 0 else 0
        remaining_min = int(remaining // 60)
        remaining_sec = int(remaining % 60)
        progress = processed_count / len(to_process) * 100
        
        print(f"\r  Batch [{batch_idx+1:4d}/{total_batches}] "
              f"Progress: {progress:5.1f}% "
              f"Speed: {speed:.1f} img/s "
              f"Batch time: {batch_time:.2f}s "
              f"Remaining: {remaining_min}m{remaining_sec:02d}s", end='', flush=True)
        
        # Periodic save
        if (batch_idx + 1) % (SAVE_INTERVAL // segmenter.batch_size) == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(results_file, index=False, encoding='utf-8')
    
    print()
    
    # Final save
    total_time = time.time() - start_time
    
    print(f"\nSaving final results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    
    # Statistics
    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"Processed this run: {processed_count} images")
    print(f"Total: {len(results)} images")
    print(f"Time elapsed: {int(total_time//60)}m{int(total_time%60)}s")
    print(f"Average speed: {processed_count/total_time:.2f} images/sec")
    
    if len(results) > 0:
        print(f"\nUrban Feature Statistics:")
        for metric in ['sky_ratio', 'green_view_index', 'building_ratio', 'road_ratio']:
            values = results_df[metric].values
            print(f"  {metric:20s}: mean={np.mean(values):.3f}")
    
    print(f"\nOutput: {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
