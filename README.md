# Traces of Joy

**Exploring Urban Happiness through Machine Vision and Human Feeling**

[![Project Website](https://img.shields.io/badge/Project-Website-green)](https://fanyang0304.github.io/MUSA5500_Final_FanYang_ZhiyuanZhao/)
[![Final Report](https://img.shields.io/badge/Final-Report-blue)](Final_Report-Fan_Yang&Zhiyuan_Zhao.pdf)

---

## Overview

What makes a person truly happy? It might be a shared meal with friends, or simply standing in a place that feels right. This project maps the emotional geography of Philadelphia—combining street-view imagery, semantic segmentation, and machine learning to understand what makes urban environments feel "happy."

> *"A lot of things need fixing in Philly, but there are a lot of good things here."*

Using 28 happiness points identified by Drexel University students as positive samples and approximately 40,000 road sampling points across Philadelphia streets, we implement a **Positive-Unlabeled (PU) Learning** framework to predict urban happiness scores city-wide.

## Key Metrics

| Metric | Value |
|--------|-------|
| Happiness Points | 28 |
| Sampling Points | 40,000+ |
| Model AUC | 0.968 |
| Features | 13 (Visual + Socioeconomic) |

## Authors

- **Fan Yang** - University of Pennsylvania, Weitzman School of Design
- **Zhiyuan Zhao** - University of Pennsylvania, Weitzman School of Design

Course: MUSA 5500 - Geospatial Data Science in Python

---

## Repository Structure

```
MUSA5500_Final_FanYang_ZhiyuanZhao/
│
├── script/                          # Analysis scripts (Python)
│   ├── 1_Generate_sampling_points.py
│   ├── 2_0_Download_gsv_panoramas.py
│   ├── 2_1_Resume_download_gsv.py
│   ├── 3_1_Extract_happy_gsv_images.py
│   ├── 3_2_Complete_happy_images.py
│   ├── 3_3_Cleanup_main_happy_images.py
│   ├── 4_1_Semantic_segmentation_happy.py
│   ├── 4_2_semantic_segmentation_all.py
│   ├── 5_1_Census_spatial_join.py
│   └── 6_Happiness_modeling.py
│
├── analysis/                        # Quarto analysis pages
│   ├── 1-data-collection.qmd
│   ├── 2-semantic-segmentation.qmd
│   ├── 3-pu-learning.qmd
│   └── 4-results.qmd
│
├── data/                            # Local data files
├── images/                          # Images and visualizations
├── docs/                            # Generated website files
│
├── _quarto.yml                      # Quarto configuration
├── index.qmd                        # Homepage
├── introduction.qmd                 # Introduction page
├── references.qmd                   # References page
├── styles.css                       # Custom CSS styling
│
├── Final_Report-Fan_Yang&Zhiyuan_Zhao.pdf  # Final report
└── README.md                        # This file
```

## Data

**Note:** Due to the large volume of Google Street View imagery data (~40,000 panoramas), all data files are stored on Google Drive:

📁 [**Google Drive Data Folder**](https://drive.google.com/file/d/1kpzX5mKrjTFOktLkOwL1HTqwxIsuO3at/view?usp=drive_link)

The data folder contains:
- `sampling_points/` - Generated sampling points shapefile
- `gsv_images/` - Google Street View panorama images
- `gsv_images_happy/` - Street view images for happiness points
- `gsv_metadata/` - Panorama metadata (pano_id, coordinates, dates)
- `semantic_segmentation_happy/` - Segmentation results for happiness points
- `semantic_segmentation_all/` - Segmentation results for all points
- `census/` - Census tract data and boundaries
- `analysis_data/` - Merged datasets for modeling
- `modeling_results/` - Model outputs and predictions

---

## Analysis Pipeline

The analysis is organized into 6 stages, implemented in the `script/` folder:

### Stage 1: Generate Sampling Points
**`1_Generate_sampling_points.py`**

Generates sampling points along Philadelphia's street network at 200-meter intervals and merges them with the 28 happiness points identified by Drexel University students. Outputs a unified shapefile with point type labels.

### Stage 2: Download Google Street View Images
**`2_0_Download_gsv_panoramas.py`** | **`2_1_Resume_download_gsv.py`**

Downloads Google Street View panorama images using the Street View API. The main script handles initial metadata collection and image downloading, while the resume script implements 20-process parallel downloading with checkpoint capability for large-scale data collection.

### Stage 3: Extract and Process Happiness Point Images
**`3_1_Extract_happy_gsv_images.py`** | **`3_2_Complete_happy_images.py`** | **`3_3_Cleanup_main_happy_images.py`**

Extracts street view images corresponding to happiness points into a separate folder, attempts to re-acquire images for any missing points using expanded search radii, and cleans up duplicate files to save disk space.

### Stage 4: Semantic Segmentation
**`4_1_Semantic_segmentation_happy.py`** | **`4_2_semantic_segmentation_all.py`**

Performs semantic segmentation on street view images using the **SegFormer-B0** model (pre-trained on ADE20K). Extracts urban visual features including:
- Sky ratio
- Green View Index (vegetation coverage)
- Building ratio
- Road ratio
- Vehicle ratio
- Person ratio

The happiness points script includes visualization outputs, while the batch script uses GPU acceleration for processing all 40,000+ images.

### Stage 5: Census Data Integration
**`5_1_Census_spatial_join.py`**

Downloads Census ACS 5-year estimates via the Census API, performs spatial joins between sampling points and Census tracts, and merges socioeconomic variables with visual features. Census features include:
- Median income
- Poverty rate
- Education level (% college graduates)
- Racial composition
- Home ownership rate
- Unemployment rate

### Stage 6: PU Learning Modeling
**`6_Happiness_modeling.py`**

Implements **Positive-Unlabeled (PU) Learning** to predict happiness scores across the city. Since we only have confirmed "happy" locations (positive samples) and unlabeled points (which may or may not be happy), traditional supervised learning isn't applicable. The PU Learning approach:

1. Identifies reliable negative samples based on feature distance from positive centroid
2. Trains Logistic Regression and Random Forest classifiers
3. Evaluates model performance using 5-fold cross-validation
4. Generates city-wide happiness predictions

---

## Website Structure

The project website is built with **Quarto** and includes:

| File | Description |
|------|-------------|
| `index.qmd` | Homepage with project overview and key findings |
| `introduction.qmd` | Research background and motivation |
| `analysis/1-data-collection.qmd` | Data collection methodology |
| `analysis/2-semantic-segmentation.qmd` | Semantic segmentation approach |
| `analysis/3-pu-learning.qmd` | PU Learning model explanation |
| `analysis/4-results.qmd` | Results and interactive maps |
| `references.qmd` | Data sources and references |
| `styles.css` | Custom styling |
| `_quarto.yml` | Site configuration |

---

## Technologies

- **Python** - Data processing and analysis
- **Google Street View API** - Street-level imagery
- **SegFormer (Hugging Face)** - Semantic segmentation
- **scikit-learn** - Machine learning models
- **GeoPandas** - Spatial data processing
- **Census API** - Socioeconomic data
- **Quarto** - Website generation
- **Mapbox** - Interactive visualizations

---

## References

- Google Street View API: https://developers.google.com/maps/documentation/streetview
- U.S. Census Bureau ACS Data: https://www.census.gov/programs-surveys/acs
- OpenDataPhilly Street Centerlines: https://opendataphilly.org/
- SegFormer Model: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512

---

## License

This project is developed for academic purposes as part of MUSA 5500 at the University of Pennsylvania.

---

<p align="center">
  <img src="images/upenn.jpg" alt="UPenn Logo" height="40">
  <br>
  <em>University of Pennsylvania | Weitzman School of Design</em>
</p>
