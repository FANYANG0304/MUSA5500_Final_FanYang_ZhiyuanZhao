#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Delete happiness point images from main directory that already exist in happy directory
Free up disk space, avoid duplicate storage
"""

from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')


def main():
    # Set paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    gsv_images_dir = data_dir / "gsv_images"
    happy_images_dir = data_dir / "gsv_images_happy"
    
    print("=" * 70)
    print("Delete Happiness Point Images from Main Directory")
    print("=" * 70)
    
    # Check if directories exist
    if not gsv_images_dir.exists():
        print(f"\n✗ Main directory does not exist: {gsv_images_dir}")
        return
    
    if not happy_images_dir.exists():
        print(f"\n✗ Happy directory does not exist: {happy_images_dir}")
        return
    
    # Get all image files in happy directory
    print(f"\n[1/3] Scanning happy directory...")
    happy_images = list(happy_images_dir.glob("*.jpg"))
    happy_filenames = set(img.name for img in happy_images)
    
    print(f"  ✓ Found {len(happy_filenames)} happiness point images")
    
    # Find these files in main directory
    print(f"\n[2/3] Finding corresponding files in main directory...")
    to_delete = []
    
    for filename in happy_filenames:
        main_image = gsv_images_dir / filename
        if main_image.exists():
            to_delete.append(main_image)
    
    print(f"  ✓ Found {len(to_delete)} files to delete")
    
    if len(to_delete) == 0:
        print("\n✓ No files need to be deleted from main directory!")
        return
    
    # Calculate space to be freed
    total_size = sum(f.stat().st_size for f in to_delete)
    size_mb = total_size / (1024 * 1024)
    
    print(f"  ✓ Space to be freed: {size_mb:.2f} MB")
    
    # Show files to be deleted (max 10)
    print(f"\nFiles to delete (total {len(to_delete)}):")
    for i, file in enumerate(to_delete[:10]):
        file_size = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({file_size:.2f} MB)")
    if len(to_delete) > 10:
        print(f"  ... and {len(to_delete) - 10} more files")
    
    # Confirm deletion
    print(f"\n" + "=" * 70)
    print("⚠ Warning: About to delete happiness point images from main directory")
    print("=" * 70)
    print(f"Target directory: {gsv_images_dir}")
    print(f"Files to delete: {len(to_delete)}")
    print(f"Space to free: {size_mb:.2f} MB")
    print(f"These files are still preserved in {happy_images_dir}")
    print("=" * 70)
    
    confirm = input("\nConfirm deletion? (type 'yes' to continue): ").strip().lower()
    
    if confirm != 'yes':
        print("\n✗ Deletion cancelled")
        return
    
    # Execute deletion
    print(f"\n[3/3] Deleting files...")
    deleted_count = 0
    failed_count = 0
    
    for idx, file in enumerate(to_delete):
        try:
            file.unlink()
            deleted_count += 1
            
            # Show progress every 100 files
            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(to_delete)}] Deleted {deleted_count} files")
        except Exception as e:
            print(f"  ✗ Failed to delete {file.name}: {e}")
            failed_count += 1
    
    # Statistics
    print("\n" + "=" * 70)
    print("Deletion Complete")
    print("=" * 70)
    print(f"Successfully deleted: {deleted_count} files")
    print(f"Failed to delete: {failed_count} files")
    print(f"Space freed: {size_mb:.2f} MB")
    print(f"\nHappiness point images preserved in: {happy_images_dir}")
    print("=" * 70)
    
    print("\n✓ Program execution complete!")


if __name__ == "__main__":
    main()
