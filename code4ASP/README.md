# 📊 Automated and Scalable Pipeline (ASP) for Multimodal Financial Report Dataset Construction

This repository provides an automated and scalable pipeline for constructing a multimodal financial dataset from Chinese listed companies' annual reports. 
The goal is to enable financial multimodal reasoning tasks such as financial chart understanding and multimodal question answering.

## 📌 Project Overview

The full pipeline consists of four stages:

1. **Data Extraction (`1_data_extraction.py`)**  
   Extracts both machine-readable text and embedded visual elements (charts, tables, figures) from PDF financial reports using `PyMuPDF` and `pdfplumber`.

2. **Data Cleaning (`2_data_cleaning.py`)**  
   Applies rule-based strategies to refine extracted content. Filters irrelevant text/images, removes duplicates, and merges fragmented segments to ensure semantic consistency.

3. **Question-Answer Generation (`3_qa_generation.py`)**  
   Uses a large language model (DeepSeek-V3) to generate chart-related multiple-choice questions based on cleaned textual content. Questions span four categories: Arithmetic Reasoning, Statistical Reasoning, Financial Explanation, and Financial Knowledge.

4. **Image-Question Alignment (`4_iq_alignment.py`)**  
   Employs a vision-language model (Qwen-VL 2.5-3B) to align questions with relevant images in the same document. Only unambiguous, semantically matched image-question pairs are retained.

---

## 🗂️ Directory Structure

```bash
.
├── 1_data_extraction.py      # Extract text and images from PDFs
├── 2_data_cleaning.py        # Clean and filter extracted data
├── 3_qa_generation.py        # Generate QA pairs from text
├── 4_iq_alignment.py         # Align images with QA pairs
├── requirements.txt          # Python package dependencies
└── README.md                 # Project overview
```


## 🚀 Quick Start
📦 Requirements
Install dependencies via:
```bash
pip install -r requirements.txt
```
Run each component in the following order:

```bash
# Step 1: Extract text and images
python 1_data_extraction.py --filename example.pdf --input_dir ./raw_pdfs --output_dir ./extracted

# Step 2: Clean extracted data
python 2_data_cleaning.py --filename example --data_dir ./extracted --dest_dir ./cleaned

# Step 3: Generate QA pairs
python 3_qa_generation.py --filename example --data_dir ./cleaned --dest_dir ./qa_pairs

# Step 4: Align questions with images
python 4_iq_alignment.py --model_path ./path_to_Qwen2.5-VL --pdf_folder_path example --dest_dir ./aligned
```
📁 Data Path Structure
Below is the expected data organization before and after each pipeline step.
🔹 Before Step 1
```
./raw_pdfs/
└── example.pdf
```
🔹 After Step 1 (1_data_extraction.py)
```
./extracted/
└── example/
    ├── content_info.json
    └── images/
        ├── image_001.png
        ├── image_002.png
        └── ...
```
🔹 After Step 2 (2_data_cleaning.py)
```
./cleaned/
└── example/
    ├── content_info.json
    └── images/
        ├── image_001.png
        ├── image_002.png
        └── ...
```
🔹 After Step 3 (3_qa_generation.py)
```
./qa_pairs/
└── example/
    ├── example.json
    └── images/
        ├── image_001.png
        ├── image_002.png
        └── ...
```
🔹 After Step 4 (4_iq_alignment.py)
```
./aligned/
└── example/
    ├── example.json
    └── images/
        ├── image_001.png
        ├── image_002.png
        └── ...
```
✅ Note: Each step uses the output of the previous step as its input. Ensure directory names and file formats remain consistent throughout.