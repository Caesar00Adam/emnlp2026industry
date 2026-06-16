
import re
from typing import Dict

from mathruler.grader import grade_answer


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        ground_truth = ground_truth.strip()
        

        predict_option_match = re.match(r'^([A-Fa-f])[\.\)\s\:]+(.*)$', given_answer)
        ground_option_match = re.match(r'^([A-Fa-f])[\.\)\s\:]+(.*)$', ground_truth)
        
        predict_option = predict_option_match.group(1).upper() if predict_option_match else None
        predict_content = predict_option_match.group(2).strip() if predict_option_match else None
        ground_option = ground_option_match.group(1).upper() if ground_option_match else None
        ground_content = ground_option_match.group(2).strip() if ground_option_match else None
        
        if predict_option and ground_option:
            if predict_option == ground_option:
                return 1.0
        

        elif ground_option and not predict_option_match and given_answer.upper() in ['A', 'B', 'C', 'D','E','F']:
            if given_answer.upper() == ground_option:
                return 1.0
        

        elif ground_option and ground_content and not predict_option:

            normalized_predict = normalize_math_text(given_answer)
            normalized_ground = normalize_math_text(ground_content)
            if normalized_predict == normalized_ground:
                return 1.0

        elif predict_option and not ground_option_match and ground_truth.upper() in ['A', 'B', 'C', 'D','E','F']:
            if predict_option == ground_truth.upper():
                return 1.0
        

        elif predict_option and predict_content and not ground_option:
            normalized_predict = normalize_math_text(predict_content)
            normalized_ground = normalize_math_text(ground_truth)
            if normalized_predict == normalized_ground:
                return 1.0
        

        if grade_answer(given_answer, ground_truth):
            return 1.0
            

        normalized_predict = normalize_math_text(given_answer)
        normalized_ground = normalize_math_text(ground_truth)
        if normalized_predict == normalized_ground:
            return 1.0

    except Exception as e:
        print(e)

    return 0.0

def normalize_math_text(text):

    text = re.sub(r'[\$\\\{\}]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'^(the answer is|answer is|result is|equals|equal to|is equal to)\s*', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s*(units?|degrees?|meters?|m|kg|grams?|cm|mm)$', '', text, flags=re.IGNORECASE)

    text = re.sub(r'(\d+)/(\d+)', lambda m: str(float(m.group(1))/float(m.group(2))), text)

    text = text.replace('π', 'pi').replace('×', '*').replace('÷', '/')
    return text.lower() 


def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.5) -> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward(predict_str, ground_truth)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
