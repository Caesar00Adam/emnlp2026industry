import json
import os
import shutil
import re
import argparse

def is_chinese_char(c):
    return '\u4e00' <= c <= '\u9fff'

def count_chinese_chars(text):
    return sum(1 for c in text if is_chinese_char(c))

def merge_texts_on_page(page_texts):
    merged_texts = []
    total_texts = len(page_texts)
    

    is_suspicious_page = total_texts > 35
    
    i = len(page_texts) - 1
    while i >= 0:
        current_text = page_texts[i].rstrip()  
        
        
        if current_text and (current_text[-1] == '。' or current_text[-1] == '；'):
            j = i - 1
            merged_paragraph = current_text
            while j >= 0:
                previous_text = page_texts[j].rstrip()  
                if (len(previous_text) > 30 and (not previous_text or (previous_text[-1] != '。' and previous_text[-1] != '；'))):
                    
                    merged_paragraph = f"{previous_text}{merged_paragraph}"
                    j -= 1
                else:
                    break
            
           
            merged_texts.insert(0, merged_paragraph)
            i = j  
        else:
           
            if len(current_text) < 60 and (current_text.startswith('Figure') or current_text.startswith('Table')):
                merged_texts.insert(0, current_text)
                i -= 1
            else:
                
                if is_suspicious_page:
                    
                    i -= 1
                else:
                    
                    chinese_count = count_chinese_chars(current_text)
                    if chinese_count / len(current_text) >= 0.5:
                        merged_texts.insert(0, current_text)
                    i -= 1
    
    return merged_texts

def get_figure_num(content_info):
    max_figures = {'Figure': 0, 'Table': 0, 'Chart': 0}
    patterns = {
        'Figure': re.compile(r'^Figure\s*(\d+)'),
        'Table': re.compile(r'^Table\s*(\d+)'),
        'Chart': re.compile(r'^Chart\s*(\d+)')
    }
    
    for page in content_info:
        for text in page['texts']:
            for key, pattern in patterns.items():
                match = pattern.match(text)
                if match:
                    max_figures[key] = max(max_figures[key], int(match.group(1)))
    
    return sum(max_figures.values())

def figure_filter(content_info, directory, target_dir):
    for page in content_info:
        is_figure_suspicious_page = all(not text.startswith(('Figure', 'Table')) for text in page['texts'])
        if is_figure_suspicious_page:
            page['images'] = []
            page['tables'] = []

def delete_directory_if_conditions_met(directory, actual_figure_num=None):
    
    content_path = os.path.join(directory, 'content_info.json')
    images_dir = os.path.join(directory, 'images')
    
    if not os.path.exists(content_path):
        should_delete = True
    elif os.path.exists(images_dir):
        num_images = len([name for name in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, name))])
        
        #
        if num_images < 1:
            should_delete = True
       
        elif actual_figure_num is not None and num_images < 0.7 * actual_figure_num:
            should_delete = True
        else:
            should_delete = False
    else:
        should_delete = False
    
    if should_delete:
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        
        os.rmdir(directory)

def copy_files_based_on_relative_paths(directory, target_dir, content_info):
   
    images_dir = os.path.join(target_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for page in content_info:
        for image in page.get('images', []):
            src_image_path = os.path.join(directory, image['relative_path'])
            dst_image_path = os.path.join(target_dir, image['relative_path'])
            os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)
        
        for table in page.get('tables', []):
            src_table_path = os.path.join(directory, table['relative_path'])
            dst_table_path = os.path.join(target_dir, table['relative_path'])
            os.makedirs(os.path.dirname(dst_table_path), exist_ok=True)
            if os.path.exists(src_table_path):
                shutil.copy2(src_table_path, dst_table_path)

def process_content(directory, target_dir):
    content_path = os.path.join(directory, 'content_info.json')
    target_content_path = os.path.join(target_dir, 'content_info.json')
    
    with open(content_path, 'r', encoding='utf-8') as file:
        original_content_info = json.load(file)
    
 
    content_info = json.loads(json.dumps(original_content_info))
    
    
    for page in content_info:
        page['texts'] = merge_texts_on_page(page['texts'])
    
    
    figure_filter(content_info, directory, target_dir)
    
    
    with open(target_content_path, 'w', encoding='utf-8') as file:
        json.dump(content_info, file, ensure_ascii=False, indent=4)
    
 
    copy_files_based_on_relative_paths(directory, target_dir, content_info)

def process_pdf(filename, data_dir, dest_dir):
    directory = os.path.join(data_dir, filename)
    target_dir = os.path.join(dest_dir, filename)  

    
    content_path_data = os.path.join(directory, 'content_info.json')
    if not os.path.exists(content_path_data):
        return 

   
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
 
    process_content(directory, target_dir)

    
    content_path = os.path.join(target_dir, 'content_info.json')
    if os.path.exists(content_path):
        with open(content_path, 'r', encoding='utf-8') as file:
            content_info = json.load(file)
        actual_figure_num = get_figure_num(content_info)
    else:
        actual_figure_num = None
    
 
    delete_directory_if_conditions_met(target_dir, actual_figure_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean PDF-extracted data by processing images, tables, and text.")
    parser.add_argument("--filename", type=str, required=True, help="PDF filename to be cleaned.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory where original PDF or extracted data is stored.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Output directory for cleaned data.")

    args = parser.parse_args()

    process_pdf(args.filename, args.data_dir, args.dest_dir)

