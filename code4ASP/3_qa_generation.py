
from openai import OpenAI
import json
import os
import shutil
import re
import argparse

def extract_date_id(path):
    parts = path.split('/')
    if len(parts) >= 5:
        return '/'.join(parts[-5:-1])
    else:
        return None

def copy_folder_contents(src, dest):
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dest_item = os.path.join(dest, item)

        if os.path.isdir(src_item):
            shutil.copytree(src_item, dest_item)
        else:
            shutil.copy2(src_item, dest_item)


# Copy PDF data information
def copy_pdf(src_path, dest_folder):
    base_name = os.path.basename(src_path)

    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, base_name)
    shutil.copy(src_path, dest_path)


def copy_doc(json_path: str, dest_path: str):
    os.makedirs(dest_path, exist_ok=True)
    data_id = extract_date_id(json_path)  
    pdf_allyear_path = ""
    info_allyear_path = ""

    pdf_name = data_id + ".PDF"
    pdf_path = os.path.join(pdf_allyear_path, pdf_name)  
    info_path = os.path.join(info_allyear_path, data_id)  

    copy_pdf(pdf_path, dest_path)  # Copy PDF
    copy_folder_contents(info_path, dest_path)  # Copy info


#merge texts from get_json_text
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def merge_texts_into_string(data):
    merged_text = ""
    for page in data:
        texts = page.get("texts", [])
        for text in texts:
            merged_text += text + " "
    return merged_text.strip()  # Remove the trailing space


def get_json_text(json_path: str) -> str:
    data = load_json(json_path)
    result = merge_texts_into_string(data)
    return result.encode('utf-8', errors='ignore').decode('utf-8')


# check whether the PDF meets requirements
# Count the maximum figure records and use it as the actual number of figures
def get_figure_num(content_info):
    max_figures = {'Figure': 0, 'Table': 0, 'Chart': 0}
    patterns = [
        ('Chart', re.compile(r'Chart\s*(\d+)')),
        ('Figure', re.compile(r'Figure\s*(\d+)')),
        ('Table', re.compile(r'Table\s*(\d+)'))
    ]

    for page in content_info:
        for text in page['texts']:
            matched_positions = set()  
            for key, pattern in patterns:
                for match in pattern.finditer(text):
                    start, end = match.span()
                    if any(start < pos < end for pos in matched_positions):
                        continue  
                    number = int(match.group(1))
                    max_figures[key] = max(max_figures[key], number)
                    for pos in range(start, end):
                        matched_positions.add(pos)

    return sum(max_figures.values())


def get_pic_num(folder_path: str) -> int:
    try:
        return len(os.listdir(folder_path))
    except FileNotFoundError:
        print("Path not found. Please check the input folder path.")
        return None


def get_text_num(file_path: str) -> int:
    return len(get_json_text(file_path))


def check(pdf_path: str):
    folder_path = os.path.join(pdf_path, "images")
    file_path = os.path.join(pdf_path, "content_info.json")
    pic_num = get_pic_num(folder_path)

    with open(file_path, 'r') as file:
        content_info = json.load(file)
        actual_count = get_figure_num(content_info)

    if (pic_num == 1) or (pic_num > 10):
        return False
    if pic_num != actual_count:
        return False
    text_num = get_text_num(file_path)
    if (text_num / pic_num) < 300:
        return False
    return True  # Return True if none of the above conditions are met

#Main function to process a single PDF folder
def process_pdf(filename, data_dir, dest_dir):
    pdf_path = os.path.join(data_dir, filename)
    if not check(pdf_path):
        return False
    dest_dir = os.path.join(dest_dir, filename)

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")


    system_prompt = '''
    Users will provide text from financial reports or research reports of an industry or company. This text may contain descriptions of various types of financial charts (such as tables, bar graphs, line graphs, pie charts, etc.). Please note that most of the text does not involve image descriptions. Determine and select out the text content describing images. If the entire text does not describe any images, output "None".

For content containing image descriptions, propose zero, one, or more questions for each mentioned chart (the number depends on the complexity and amount of information of the chart). Ensure that each question falls into one of four categories and clearly indicates which chart it pertains to (including chart number and title). Each question must meet the following conditions:

1. **Arithmetic reasoning question**: Ask about numerical attributes and relationships within financial charts, requiring simple numerical operations and logical reasoning.
2. **Statistical reasoning question**: Focus on statistical measures within financial charts, such as mean, median, mode, variance, standard deviation, correlation coefficient, etc., involving relatively complex calculations.
3. **Financial knowledge question**: Involves professional terms related to finance, using full Chinese names for these terms.
4. **Financial Explanation question**: Combine information from trends, multiple values, and other aspects to ask about specific financial issues and scenarios.

All questions must meet the following criteria:

- Each question should clearly specify which chart it targets (including chart number and title like Chart 1: Stock Trend), whether using "Chart 1", "Figure 1", or "Table 1" based on textual information.
- Construct questions based on provided descriptions without using hypothetical language like "if" or "a certain company"; directly reference actual data and concepts from the description.
- The question's description should not include phrases like "according to chart X"; just construct the question directly. Information about chart numbers and titles should be included in chart_info.
- Ensure clear question phrasing without ambiguity and concise correct answers that are easy to judge.
- Provide five options, detailed explanations, and the correct answer for each question. Answers should only rely on provided description, not external sources.
- Question answers cannot be directly or indirectly calculated from the question description since this information comes from the images.
- Explanations must be thorough, including complete reasoning and calculation processes, explaining each step as detailed as possible.
- Vary questioning angles as much as possible but choose only one type of question from the four categories for each question, covering specific values, trends, fluctuations, descriptions, etc.
- Finally, check if question answers can be directly or indirectly calculated from the question description; if so, the question is disqualified.
- If the question's answers and explanations are incomplete or correctness is not guaranteed, the question is disqualified.
- Question answers must be definite with no uncertainty. If the answer is uncertain or the explanation mentions needing to determine based on chart information, the question is disqualified.
- Try to evenly distribute question types, avoiding all being of one type.
- For questions involving numerical options, ensure there is a noticeable difference between option values based on text or answer value scales, and eliminate any numerically ambiguous questions.
- For questions with time-sensitive answers, if no specific time is mentioned, use "most recent"; otherwise, use the specific time provided.
- Perform a final check to ensure there are no options or answers with uncertainty. If they exist, they should be excluded from the final output.

Carefully review each generated question and its answer to ensure accuracy and applicability. Ensure that questions can be answered based on the given textual description without being evident in the question description.
The response format should be nested JSON, with the output JSON being a list of questions, each containing the following parts: "chart_info": records chart number and title; "question_type": question category (arithmetic reasoning, statistical calculation, financial knowledge, financial analysis); "question": the posed question; "options": question options; "correct_answer": the correct answer; "explanation": detailed explanation of the question.

Ensure each part within a question is strictly ordered as listed above. Reference the following JSON structure:
{
    [
        {
            "chart_info": "Chart 4 LLDPE Profit Comparison by Different Raw Materials",
            "question_type": "Arithmetic Reasoning",
            "question": "How much did the profit of oil-based LLDPE increase per ton in the most recent period compared to last week?",
            "options": [
                "A. 90 yuan/ton",
                "B. 110 yuan/ton",
                "C. 120 yuan/ton",
                "D. 130 yuan/ton",
                "E. 140 yuan/ton"
            ],
            "explanation": "According to the description, the profit of oil-based LLDPE increased by 110 yuan/ton compared to last week.",
            "correct_answer": "B. 110 yuan/ton"
        },
        ...
    ]
}

If the entire text does not describe any images, output the JSON format as:
{
    "response": "None"
}
    '''
    json_file_path = os.path.join(pdf_path, "content_info.json")
    user_prompt = get_json_text(json_file_path)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        frequency_penalty=0,
        max_tokens=4096,
        presence_penalty=0,
        response_format={
            'type': 'json_object'},
        stream=False
    )

    data = json.loads(response.choices[0].message.content)
    if data == {"response": "None"}:
        return False

    os.makedirs(dest_dir, exist_ok=True)

    filename1 = filename + ".json"
    file_path = os.path.join(dest_dir, filename1)

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    copy_doc(json_file_path, dest_dir)
    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA pairs from extracted PDF content.")
    parser.add_argument("--filename", type=str, required=True, help="Filename of the PDF to process.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing extracted data.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Directory to save the generated QA pairs.")

    args = parser.parse_args()

    process_pdf(args.filename, args.data_dir, args.dest_dir)
