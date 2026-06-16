import os  
import fitz  # PyMuPDF
import pdfplumber
import json
import hashlib
from collections import defaultdict
import argparse




def hash_image(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def extract_images(doc, output_folder, min_size=50):
    images_info = []
    image_hashes = defaultdict(list)

    for page_num in range(len(doc) - 2):  
        
        page = doc.load_page(page_num)
    
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            
            base_image = doc.extract_image(xref)
           
            image_bytes = base_image["image"]
            width, height = base_image["width"], base_image["height"]

            if width > min_size and height > min_size:
                img_hash = hash_image(image_bytes)
                filename = f"page_{page_num+1}_image_{img_index}.png"
                image_hashes[img_hash].append(filename)
                bbox=page.get_image_rects(xref)
                if len(bbox)==0:
                    continue
                mat = fitz.Matrix(2.0, 2.0)
                #Initial filtering of image size
                bbox=(bbox[0][0],bbox[0][1],bbox[0][2],bbox[0][3])
                if(bbox[2]-bbox[0]) < 100 and (bbox[3]-bbox[1]) < 100:
                    continue
                try:
                    bbox1 = (bbox[0]-50,bbox[1]-50,bbox[2]+50,bbox[3]+50)
                    pix = doc[page_num].get_pixmap(clip=bbox1,matrix = mat)

                except Exception as e:
                    try:
                        bbox1 = (bbox[0]-30,bbox[1]-30,bbox[2]+30,bbox[3]+30)
                        pix = doc[page_num].get_pixmap(clip=bbox1,matrix = mat)
                    except Exception as e:
                        pix = doc[page_num].get_pixmap(clip=bbox,matrix = mat)
                pix.save(os.path.join(output_folder, 'images', filename))

                images_info.append({
                    'page_number': page_num + 1,
                    'filename': filename,
                    'relative_path': f"images/{filename}"
                })
    return images_info, image_hashes



def extract_tables(pdf_path, output_folder, min_rows=1, min_cols=1):
    table_bboxes = defaultdict(list)
    table_info = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(len(pdf.pages) - 2):  
            page = pdf.pages[page_num]
            tables = page.find_tables()
            doc = fitz.open(pdf_path)
            j = 0
            
            for i, table in enumerate(tables):
                if len(table.rows) > min_rows and len(table.columns) > min_cols:
                    bbox = (table.bbox[0], table.bbox[1], table.bbox[2], table.bbox[3])
                    table_bboxes[page_num].append(bbox)
                    mat = fitz.Matrix(2.0, 2.0)
                    
                    if(bbox[2]-bbox[0] < 100) and (bbox[3]-bbox[1] < 100):
                        continue
                    
                    try:
                        bbox1 = (table.bbox[0]-50, table.bbox[1]-50, table.bbox[2]+50, table.bbox[3]+50)
                        pix = doc[page_num].get_pixmap(clip=bbox1,matrix = mat)
                    except Exception as e:
                        try:
                            bbox1 = (table.bbox[0]-30, table.bbox[1]-30, table.bbox[2]+30, table.bbox[3]+30)
                            pix = doc[page_num].get_pixmap(clip=bbox1,matrix = mat)
                        except Exception as e:
                            pix = doc[page_num].get_pixmap(clip=bbox,matrix = mat)
                        
                    if pix.width>50 and pix.height>50:
                        filename = f"page_{page_num+1}_table_{j+1}.png"
                        j=j+1
                        pix.save(os.path.join(output_folder, 'images', filename))
                        table_info.append({
                            'page_number': page_num + 1,
                            'filename': filename,
                            'relative_path': f"images/{filename}"
                        })

    return table_bboxes, table_info

def is_valid_first_char(char):
    return char.isprintable() and not char.isspace()

def clean_text(text):
    while text and not is_valid_first_char(text[0]):
        text = text[1:]
    return text

def is_invalid_text(text, invalid_starts):
    if not text or len(text.strip()) == 0:  
        return True
    for start in invalid_starts:
        if text.startswith(start):
            return True
    return False

def extract_non_table_content(doc, table_bboxes, invalid_starts):
    """Extract content (text) from non-table areas and filter out invalid texts"""
    content_info = []

    for page_num in range(len(doc) - 2):  
        page = doc.load_page(page_num)
        text_blocks = []

        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] == 0:  
                block_rect = fitz.Rect(b["bbox"])
                intersects_table = any(block_rect.intersects(fitz.Rect(*bbox)) for bbox in table_bboxes[page_num])
                if not intersects_table:
                    text = ''.join([s["text"] for l in b["lines"] for s in l["spans"]])
                    cleaned_text = clean_text(text)
                    if not is_invalid_text(cleaned_text, invalid_starts):
                        text_blocks.append(cleaned_text)

        content_info.append({
            'page_number': page_num + 1,
            'texts': text_blocks
        })

    return content_info

def delete_duplicate_images(images_info, image_hashes, output_folder):
    """Delete all duplicate images"""
    valid_images = set()

    for img_hash, filenames in image_hashes.items():
        if len(filenames) == 1:
            valid_images.add(filenames[0])
        else:
            for filename in filenames:
                file_path = os.path.join(output_folder, 'images', filename)
                if os.path.exists(file_path):
                    os.remove(file_path)

    return [info for info in images_info if info['filename'] in valid_images]

def address_pdf(filename, output_dir, directory_path):
    folder_name = os.path.splitext(filename)[0]
    output_folder = os.path.join(output_dir, folder_name)
    pdf_path = os.path.join(directory_path, filename)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, 'images')):
        os.makedirs(os.path.join(output_folder, 'images'))

    doc = fitz.open(pdf_path)
    
    images_info, image_hashes = extract_images(doc, output_folder, min_size=50)
      
    table_bboxes, table_info = extract_tables(pdf_path, output_folder, min_rows=1, min_cols=1)
    
    valid_images = delete_duplicate_images(images_info, image_hashes, output_folder)

    invalid_starts = [
        
    ]

    content_info = extract_non_table_content(doc, table_bboxes, invalid_starts)
    
    for page_info in content_info:
        page_num = page_info['page_number']
        page_info['images'] = [img for img in valid_images if img['page_number'] == page_num]
        page_info['tables'] = [table for table in table_info if table['page_number'] == page_num]

    
    with open(os.path.join(output_folder, 'content_info.json'), 'w', encoding='utf-8') as f:
        json.dump(content_info, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True, help='PDF file name')
    parser.add_argument('--input_dir', required=True, help='Directory where PDF is stored')
    parser.add_argument('--output_dir', required=True, help='Where to save results')
    args = parser.parse_args()
    
    address_pdf(args.filename, args.output_dir, args.input_dir)



