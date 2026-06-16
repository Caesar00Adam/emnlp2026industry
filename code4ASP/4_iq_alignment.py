import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import shutil
import argparse


def generate_messages(image_path, caption):
    prompt_text = (
        "Does the uploaded image contain the text from the provided caption?"
        "Please respond only with 'Yes' or 'No', output a single word without any other characters including punctuation"
    )
    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    messages[0]["content"].append(
        {
            "type": "image",
            "image": image_path,
        }
    )

    full_text = f"{prompt_text}\n\ncaption:\n[{caption}]"
    messages[0]["content"].append(
        {
            "type": "text",
            "text": full_text,
        }
    )

    return messages

def extract_chart_info(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return [item["chart_info"] for item in data.get("response", []) if "chart_info" in item]


def get_png_images(folder_path):
    return sorted(
        [os.path.join(folder_path, file) for file in os.listdir(folder_path)
         if file.lower().endswith('.png')],
        key=lambda path: os.path.basename(path).lower()  # Sort by filename case-insensitively
    )


def check(chart_info, image_path, model, processor):
    """
    Custom matching verification function.

    Args:
        chart_info (str): An element from the chart_info list.
        image_path (str): An element from the image path list.

    Returns:
        bool: Return True if matched, otherwise False.
    """
    messages = generate_messages(image_path, chart_info)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    
    if output_text[0] == "yes":
        return True
    else:
        return False


def post_process_data(valid_chart_info_list, unique_matches, pdf_folder_path, dest_dir):
    """
    Post-processing function including copying images, generating new JSON files, etc.

    Args:
        valid_chart_info_list (list): List of valid chart_info entries.
        unique_matches (dict): Dictionary mapping chart_info to image paths uniquely.
        pdf_folder_path (str): Original PDF folder path.
        dest_dir (str): Target directory path.

    Returns:
        None
    """
    if not valid_chart_info_list:
        print("Valid chart_info list is empty. Skipping post-processing.")
        return

    # Create target subfolder under dest_dir with the same name as pdf_folder_path
    target_subfolder = os.path.basename(pdf_folder_path)
    target_dir = os.path.join(dest_dir, target_subfolder)
    os.makedirs(target_dir, exist_ok=True)

    # 1. Copy images folder and its contents to the target folder
    source_images_folder = os.path.join(pdf_folder_path, "images")
    dest_images_folder = os.path.join(target_dir, "images")
    os.makedirs(dest_images_folder, exist_ok=True)

    # Copy all PNG images
    for img_name in os.listdir(source_images_folder):
        img_path = os.path.join(source_images_folder, img_name)
        if img_path.lower().endswith('.png'):
            shutil.copy(img_path, dest_images_folder)

    # 2. Load original JSON file
    json_file = os.path.join(pdf_folder_path, os.path.basename(pdf_folder_path) + ".json")
    with open(json_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # Build new JSON data structure
    new_json_data = {}
    for chart_info in valid_chart_info_list:
        img_path = unique_matches[chart_info]
        img_name = os.path.basename(img_path)  # Store only the image name

        # Filter QA info matching current chart_info
        filtered_questions = []
        for qa in original_data.get("response", []):
            if qa.get("chart_info") == chart_info:
                filtered_questions.append({
                    "question_type": qa.get("question_type"),
                    "question": qa.get("question"),
                    "options": qa.get("options"),
                    "correct_answer": qa.get("correct_answer")
                })

        # Save image name and questions into new JSON
        new_json_data[chart_info] = {
            "img_path": img_name,
            "questions": filtered_questions
        }

    # 3. Save the new JSON file to the target folder
    new_json_file = os.path.join(target_dir, os.path.basename(pdf_folder_path) + ".json")
    with open(new_json_file, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, ensure_ascii=False, indent=4)


def process_pdf_folder(pdf_folder_path, dest_dir, model, processor):
    """
    Process the PDF folder, extract chart_info and image paths, then perform matching.

    Args:
        pdf_folder_path (str): Path to the PDF folder.

    Returns:
        dict: A dictionary containing matched results, keys are chart_info entries, values are matched image paths.
    """
    json_file = os.path.join(pdf_folder_path, os.path.basename(pdf_folder_path) + ".json")
    images_folder = os.path.join(pdf_folder_path, "images")

    chart_info_list = extract_chart_info(json_file)

    image_paths_list = get_png_images(images_folder)

    chart_info2_list = list(dict.fromkeys(chart_info_list)) 

    # Matching logic
    match_results = {}
    for chart_info in chart_info2_list:
        matched_images = []
        for image_path in image_paths_list:
            if check(chart_info, image_path, model, processor):  
                matched_images.append(image_path)
        match_results[chart_info] = matched_images

    
    # Post-process: filter unique matches
    unique_matches = {}
    used_images = set()  

    for chart_info in chart_info2_list:
        matched_images = match_results.get(chart_info, [])
        if len(matched_images) == 1 and matched_images[0] not in used_images:
            unique_match_image = matched_images[0]
            unique_matches[chart_info] = unique_match_image
            used_images.add(unique_match_image)

    
    valid_chart_info_list = list(unique_matches.keys())

    # Data post-processing
    post_process_data(valid_chart_info_list, unique_matches, pdf_folder_path, dest_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA alignment using Qwen2.5-VL.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5-VL model directory.")
    parser.add_argument("--pdf_folder_path", type=str, required=True, help="Path to folder containing PDF JSONs.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory for aligned results.")

    args = parser.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float32, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    process_pdf_folder(args.pdf_folder_path, args.dest_dir, model, processor)
