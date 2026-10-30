# 🛰️ Building Footprint Extraction and Tree Canopy Detection Using Deep Learning

An AI-powered geospatial analysis project that uses satellite imagery, deep learning, and computer vision to extract building footprints and identify tree canopy regions. The project aims to support urban mapping, environmental monitoring, and Tree Equity Score (TES) assessment.

## 📌 Project Overview

Buildings and trees are important components of urban environments. Accurately identifying building footprints and tree canopy coverage from satellite imagery can help in urban planning, green-space assessment, and environmental sustainability.

This project applies deep learning-based image segmentation techniques to satellite imagery to identify buildings and tree-covered regions. The extracted information can be used for geospatial visualization, urban vegetation analysis, and further environmental assessment.

## 🎯 Objectives

- Extract building footprints from satellite imagery using deep learning.
- Detect and segment tree canopy regions from high-resolution images.
- Apply semantic segmentation techniques for pixel-level classification.
- Process geospatial raster imagery for urban feature extraction.
- Generate building masks and tree canopy masks for spatial analysis.
- Support Tree Equity Score assessment through urban tree canopy mapping.
- Explore applications in sustainable urban development and environmental monitoring.

## 🧠 Methodology

The project follows a deep learning-based image segmentation pipeline.

### 1. Satellite Image Acquisition
Satellite imagery and relevant geospatial datasets are collected for building extraction and tree canopy analysis.

### 2. Image Preprocessing
Input imagery is prepared for model inference or training through operations such as image tiling, resizing where appropriate, normalization, and mask preparation.

### 3. Building Footprint Extraction
A deep learning-based segmentation pipeline identifies building regions in satellite images.

The building extraction workflow includes:
- Satellite image preprocessing.
- Building segmentation using a pretrained model and/or a U-Net++ refinement model.
- Prediction mask generation.
- Morphological post-processing and removal of small connected regions.
- Conversion of segmented regions into building polygons when required.

### 4. Tree Canopy Detection
A U-Net++-based semantic segmentation model identifies tree canopy pixels in satellite imagery.

The workflow includes:
- Tree dataset preparation.
- Image and mask preprocessing.
- Deep learning model inference.
- Tree canopy mask generation.
- Evaluation using segmentation metrics.

### 5. Geospatial Analysis
Extracted masks can be integrated with geospatial raster and vector data for visualization and further spatial analysis.

Potential outputs include:
- Building footprint masks.
- Building polygon layers.
- Tree canopy segmentation masks.
- Tree canopy coverage maps.
- Spatial datasets for urban environmental assessment.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Segmentation Architecture:** U-Net++
- **Feature Extraction:** EfficientNet-B4 encoder for the building refinement model
- **Computer Vision:** OpenCV
- **Geospatial Processing:** Rasterio, GeoPandas, Shapely
- **Model Inference:** ONNX Runtime for the ONNX-based building detection pipeline
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Geospatial Software:** QGIS
- **Satellite Data Processing:** Google Earth Engine and geospatial raster workflows

*The technologies used may differ between the building extraction and tree canopy detection pipelines.*

## 🌳 Tree Equity Score (TES) Application

Tree Equity Score assessment aims to understand how tree canopy is distributed across urban areas and how that distribution relates to community needs.

The building and tree detection outputs can contribute to environmental assessment by supporting:

- Estimation of tree canopy coverage.
- Identification of built-up regions.
- Visualization of green and built environments.
- Analysis of urban vegetation distribution.
- Identification of areas for further green-space assessment.

**Note:** Building and tree segmentation alone do not calculate a complete Tree Equity Score. Additional socioeconomic, demographic, temperature, and other relevant datasets may be required, depending on the assessment methodology.

## 📊 Model Evaluation

The models can be evaluated using appropriate segmentation metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified pixels. |
| Precision | Measures how many predicted positive pixels are correct. |
| Recall | Measures how many actual positive pixels are detected. |
| Dice Score | Measures the overlap between predicted and ground-truth masks. |
| Intersection over Union (IoU) | Measures the intersection-to-union ratio between predicted and ground-truth regions. |

Evaluation results should be reported separately for building extraction and tree canopy segmentation using their respective test datasets.

## 📁 Repository Structure

A suggested repository structure is:

```text
Building-and-Tree-Detection/
│
├── building_detection/
│   ├── scripts/
│   ├── models/
│   ├── input_tiff/
│   ├── output_masks/
│   └── output_polygons/
│
├── tree_detection/
│   ├── scripts/
│   ├── checkpoints/
│   ├── dataset/
│   └── output_masks/
│
├── notebooks/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Adjust this structure to match the files and folders in your actual repository.*

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Building-and-Tree-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies required by your particular scripts and model checkpoints.

## 🚀 Usage

The general workflow is:

1. Prepare satellite imagery in the format expected by the selected pipeline.
2. Configure the appropriate trained model or checkpoint.
3. Run the building extraction pipeline to generate building segmentation masks.
4. Run the tree canopy segmentation pipeline to generate tree masks.
5. Apply post-processing where required.
6. Visualize the results and perform geospatial analysis.

The exact commands depend on the scripts included in the repository. Add the relevant commands once the final scripts and configuration files are organized.

## 📍 Applications

- Urban planning and development.
- Building footprint mapping.
- Tree canopy monitoring.
- Urban green-space assessment.
- Environmental sustainability studies.
- Geospatial mapping and analysis.
- Tree Equity Score-related research.
- Satellite image segmentation research.

## 🔮 Future Enhancements

- Improve building boundary accuracy and separation of adjacent buildings.
- Evaluate performance on additional satellite imagery and geographic regions.
- Improve tree canopy segmentation under different lighting and seasonal conditions.
- Integrate building and tree outputs into an interactive WebGIS interface.
- Incorporate population, temperature, demographic, and plantable-area datasets.
- Develop automated geospatial reporting and urban vegetation assessment workflows.

## 👩‍💻 Author

**Jakka Vishnu Priya**

B.Tech — Computer Science and Engineering (Artificial Intelligence and Machine Learning)

SRKR Engineering College, Bhimavaram

## 📄 License

Add an appropriate open-source license if you intend to distribute this project publicly. Ensure that the licenses of the datasets, pretrained models, and third-party components permit your intended use.

## 🙌 Acknowledgements

This project builds on deep learning, computer vision, and geospatial technologies for satellite image analysis. Acknowledgements and citations for the specific datasets, pretrained models, and research papers used should be added here.
