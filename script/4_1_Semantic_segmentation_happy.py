#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSV Panorama Semantic Segmentation - Happiness Points Only (with Visualization)
Uses SegFormer-B0 (ADE20K) for urban scene semantic segmentation
Output: Visualization images + numerical results

Install dependencies:
    pip install transformers --break-system-packages
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import torch
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
    print("Run: pip install transformers --break-system-packages")


class UrbanSegmentation:
    """Urban scene semantic segmentation model"""
    
    def __init__(self):
        """Initialize segmentation model"""
        print("Initializing semantic segmentation model...")
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"  ✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("  ⚠ No GPU detected, using CPU")
        
        print("  Loading SegFormer-B0 model...")
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("  ✓ Model loaded successfully")
        
        # Define class groups
        self.sky_classes = [2]
        self.vegetation_classes = [4, 9, 17, 66, 72]
        self.building_classes = [1, 25, 48, 84]
        self.road_classes = [6, 11, 52]
        self.vehicle_classes = [20, 80, 83, 102, 127]
        self.person_classes = [12]
        
        # Class names and colors
        self.legend_classes = {
            2: ("Sky", [135, 206, 235]),
            4: ("Tree", [34, 139, 34]),
            9: ("Grass", [124, 252, 0]),
            17: ("Plant", [0, 128, 0]),
            1: ("Building", [128, 128, 128]),
            6: ("Road", [64, 64, 64]),
            11: ("Sidewalk", [192, 192, 192]),
            20: ("Car", [255, 0, 0]),
            12: ("Person", [255, 192, 203]),
            80: ("Bus", [255, 165, 0]),
            83: ("Truck", [139, 69, 19]),
            127: ("Bicycle", [255, 255, 0]),
            25: ("House", [160, 82, 45]),
            48: ("Skyscraper", [105, 105, 105]),
            52: ("Path", [169, 169, 169]),
            72: ("Palm", [0, 100, 0]),
            102: ("Van", [178, 34, 34]),
        }
        
        self.color_map = self._create_color_map()
        print("  ✓ Initialization complete")
    
    def _create_color_map(self):
        """Create color mapping"""
        np.random.seed(42)
        color_map = np.random.randint(0, 255, (150, 3), dtype=np.uint8)
        for class_id, (name, color) in self.legend_classes.items():
            color_map[class_id] = color
        return color_map
    
    def segment_image(self, image_path):
        """Perform semantic segmentation on a single image"""
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            return None, None
        
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        h, w = original_image.shape[:2]
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode='bilinear', align_corners=False
        )
        
        prediction = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        return prediction, original_image
    
    def calculate_ratios(self, segmentation):
        """Calculate ratios of various urban features"""
        total_pixels = segmentation.size
        
        ratios = {
            'sky_ratio': np.sum(np.isin(segmentation, self.sky_classes)) / total_pixels,
            'green_view_index': np.sum(np.isin(segmentation, self.vegetation_classes)) / total_pixels,
            'building_ratio': np.sum(np.isin(segmentation, self.building_classes)) / total_pixels,
            'road_ratio': np.sum(np.isin(segmentation, self.road_classes)) / total_pixels,
            'vehicle_ratio': np.sum(np.isin(segmentation, self.vehicle_classes)) / total_pixels,
            'person_ratio': np.sum(np.isin(segmentation, self.person_classes)) / total_pixels,
        }
        
        ratios['tree_ratio'] = np.sum(segmentation == 4) / total_pixels
        ratios['grass_ratio'] = np.sum(segmentation == 9) / total_pixels
        ratios['sidewalk_ratio'] = np.sum(segmentation == 11) / total_pixels
        ratios['car_ratio'] = np.sum(segmentation == 20) / total_pixels
        
        return ratios
    
    def find_label_positions(self, segmentation, min_area_ratio=0.005):
        """Find label positions for each class"""
        h, w = segmentation.shape
        total_pixels = h * w
        min_area = total_pixels * min_area_ratio
        
        label_positions = []
        
        for class_id, (class_name, color) in self.legend_classes.items():
            mask = (segmentation == class_id).astype(np.uint8)
            area = np.sum(mask)
            
            if area < min_area:
                continue
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            
            for i in range(1, num_labels):
                region_area = stats[i, cv2.CC_STAT_AREA]
                if region_area < min_area:
                    continue
                
                cx, cy = centroids[i]
                cx, cy = int(cx), int(cy)
                # Increased margins for larger labels
                cx = max(150, min(w - 150, cx))
                cy = max(80, min(h - 80, cy))
                
                label_positions.append((class_id, class_name, (cx, cy), region_area))
        
        label_positions.sort(key=lambda x: x[3], reverse=True)
        return label_positions
    
    def draw_labels_on_image(self, image, label_positions, segmentation):
        """Draw class labels on image"""
        total_pixels = segmentation.size
        font = cv2.FONT_HERSHEY_SIMPLEX
        used_positions = []
        
        # Font settings - increased for better visibility on high-res panoramas (3328x1664)
        label_font_scale = 2      # Class name font size
        ratio_font_scale = 1.7    # Percentage font size
        label_thickness = 7       # Class name thickness
        ratio_thickness = 3       # Percentage thickness
        border_thickness = 7      # Box border thickness
        
        for class_id, class_name, (cx, cy), area in label_positions:
            too_close = False
            for ux, uy in used_positions:
                # Increased spacing to avoid overlap with larger text
                if abs(cx - ux) < 250 and abs(cy - uy) < 100:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            ratio = area / total_pixels
            label_text = f"{class_name}"
            ratio_text = f"{ratio:.1%}"
            
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, label_font_scale, label_thickness)
            (ratio_w, ratio_h), _ = cv2.getTextSize(ratio_text, font, ratio_font_scale, ratio_thickness)
            
            box_w = max(text_w, ratio_w) + 30      # Increased padding
            box_h = text_h + ratio_h + 40          # Increased padding
            
            x1 = cx - box_w // 2
            y1 = cy - box_h // 2
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            h, w = image.shape[:2]
            x1 = max(0, min(w - box_w, x1))
            y1 = max(0, min(h - box_h, y1))
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)  # Darker background for better contrast
            
            rgb_color = self.legend_classes[class_id][1]
            bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, border_thickness)
            
            text_x = x1 + 15
            text_y = y1 + text_h + 12
            cv2.putText(image, label_text, (text_x, text_y), font, label_font_scale, (255, 255, 255), label_thickness)
            
            ratio_y = text_y + ratio_h + 15
            cv2.putText(image, ratio_text, (text_x, ratio_y), font, ratio_font_scale, bgr_color, ratio_thickness)
            
            used_positions.append((cx, cy))
        
        return image
    
    def create_visualization(self, original_image, segmentation, point_id, ratios):
        """Create segmentation result visualization image"""
        h, w = segmentation.shape
        
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in range(150):
            mask = segmentation == class_id
            if np.any(mask):
                color_mask[mask] = self.color_map[class_id]
        
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_image, 0.6, color_mask_bgr, 0.4, 0)
        
        label_positions = self.find_label_positions(segmentation)
        overlay_labeled = self.draw_labels_on_image(overlay.copy(), label_positions, segmentation)
        
        stats_lines = [
            f"Point {point_id}",
            f"Sky: {ratios['sky_ratio']:.1%}",
            f"Green: {ratios['green_view_index']:.1%}",
            f"Building: {ratios['building_ratio']:.1%}",
            f"Road: {ratios['road_ratio']:.1%}",
        ]
        
        # Stats overlay font settings - increased for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        stats_font_scale = 1.7
        stats_thickness_bg = 7
        stats_thickness_fg = 3
        line_height = 70
        y_offset = 50
        x_offset = 20
        
        for i, line in enumerate(stats_lines):
            y_pos = y_offset + i * line_height
            # Draw shadow/outline for better visibility
            cv2.putText(overlay_labeled, line, (x_offset + 2, y_pos), font, stats_font_scale, (0, 0, 0), stats_thickness_bg)
            cv2.putText(overlay_labeled, line, (x_offset, y_pos), font, stats_font_scale, (255, 255, 255), stats_thickness_fg)
        
        visualization = np.hstack([original_image, color_mask_bgr, overlay_labeled])
        return visualization


def main():
    """Main function: Process happiness point images"""
    
    if not HAS_TRANSFORMERS:
        print("Please install transformers library first!")
        return
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    happy_images_dir = data_dir / "gsv_images_happy"
    happy_metadata_file = data_dir / "gsv_metadata" / "gsv_metadata_happy.csv"
    
    output_base = data_dir / "semantic_segmentation_happy"
    segmentation_dir = output_base / "segmentations"
    visualization_dir = output_base / "visualizations"
    
    segmentation_dir.mkdir(exist_ok=True, parents=True)
    visualization_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("GSV Semantic Segmentation - Happiness Points (with Visualization)")
    print("Model: SegFormer-B0 (ADE20K)")
    print("=" * 70)
    
    print(f"\n[1/4] Reading metadata...")
    try:
        metadata_df = pd.read_csv(happy_metadata_file)
        print(f"  ✓ Read {len(metadata_df)} records")
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
        return
    
    print(f"\n[2/4] Checking image files...")
    available_images = []
    for idx, row in metadata_df.iterrows():
        pano_id = row['pano_id']
        image_file = happy_images_dir / f"{pano_id}.jpg"
        if image_file.exists():
            available_images.append({
                'pano_id': pano_id,
                'point_id': row['point_id'],
                'image_path': image_file
            })
    
    print(f"  ✓ Found {len(available_images)} image files")
    
    if not available_images:
        print("  ✗ No image files found!")
        return
    
    print(f"\n[3/4] Loading segmentation model...")
    segmenter = UrbanSegmentation()
    
    print(f"\n[4/4] Processing images...")
    print(f"  Output directory: {output_base}\n")
    
    results = []
    start_time = time.time()
    
    for idx, item in enumerate(available_images):
        pano_id = item['pano_id']
        point_id = item['point_id']
        image_path = item['image_path']
        
        progress = (idx + 1) / len(available_images) * 100
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1) if idx > 0 else 0
        remaining = avg_time * (len(available_images) - idx - 1)
        
        print(f"  [{idx+1:3d}/{len(available_images)}] {progress:5.1f}% Point {point_id} ", end='', flush=True)
        
        segmentation, original_image = segmenter.segment_image(image_path)
        
        if segmentation is None:
            print("✗")
            continue
        
        ratios = segmenter.calculate_ratios(segmentation)
        
        # Save segmentation result
        seg_path = segmentation_dir / f"{pano_id}.npy"
        np.save(seg_path, segmentation.astype(np.uint8))
        
        # Save visualization
        vis = segmenter.create_visualization(original_image, segmentation, point_id, ratios)
        vis_path = visualization_dir / f"{pano_id}_seg.jpg"
        cv2.imwrite(str(vis_path), vis)
        
        result = {'point_id': point_id, 'pano_id': pano_id}
        result.update(ratios)
        results.append(result)
        
        print(f"✓ Sky:{ratios['sky_ratio']:.1%} Green:{ratios['green_view_index']:.1%} ({remaining:.0f}s)")
    
    total_time = time.time() - start_time
    
    results_df = pd.DataFrame(results)
    results_file = output_base / "segmentation_results.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    
    print("\n" + "=" * 70)
    print(f"Complete! Processed {len(results)} images, time elapsed {total_time:.1f} seconds")
    print(f"Output directory: {output_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
