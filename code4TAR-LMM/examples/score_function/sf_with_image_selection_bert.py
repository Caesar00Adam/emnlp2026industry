
import re
from typing import Dict, Optional, Union


from mathruler.grader import grade_answer


def format_reward(predict_str: str) -> float:

    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>\s*<image_selection>.*?</image_selection>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:

        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        ground_truth = ground_truth.strip()
        

        predict_option_match = re.match(r'^([A-Ea-e])[\.\)\s\:]+(.*)$', given_answer)
        ground_option_match = re.match(r'^([A-Ea-e])[\.\)\s\:]+(.*)$', ground_truth)
        
        predict_option = predict_option_match.group(1).upper() if predict_option_match else None
        predict_content = predict_option_match.group(2).strip() if predict_option_match else None
        ground_option = ground_option_match.group(1).upper() if ground_option_match else None
        ground_content = ground_option_match.group(2).strip() if ground_option_match else None
        

        if predict_option and ground_option:
            if predict_option == ground_option:
                return 1.0
        

        elif ground_option and not predict_option_match and given_answer.upper() in ['A', 'B', 'C', 'D','E']:
            if given_answer.upper() == ground_option:
                return 1.0
        

        elif ground_option and ground_content and not predict_option:

            normalized_predict = normalize_math_text(given_answer)
            normalized_ground = normalize_math_text(ground_content)
            if normalized_predict == normalized_ground:
                return 1.0
        

        elif predict_option and not ground_option_match and ground_truth.upper() in ['A', 'B', 'C', 'D','E']:
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


def extract_thinking_process(predict_str: str) -> str:

    try:
        thinking_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
        return thinking_match.group(1).strip() if thinking_match else ""
    except Exception:
        return ""


def image_selection_reward(predict_str: str, correct_image_index: int) -> float:

    try:
        selection_match = re.search(r"<image_selection>(\d+)</image_selection>", predict_str)
        if not selection_match:
            return 0.0
        
        selected_image_index = int(selection_match.group(1))
        
        if selected_image_index == correct_image_index:
            return 1.0
    
    except Exception:
        pass
    
    return 0.0


def thinking_length_reward(predict_str: str, min_length: int = 100, optimal_length: int = 300, max_length: int = 600) -> float:

    try:

        thinking = extract_thinking_process(predict_str)
        

        thinking_length = len(thinking)
        

        if thinking_length == 0:
            return 0.0

        if thinking_length < min_length:
            return 0.6 * (thinking_length / min_length) ** 0.5
            

        if thinking_length >= min_length and thinking_length <= optimal_length:

            progress = (thinking_length - min_length) / (optimal_length - min_length)
            return 0.6 + 0.4 * progress
            
        if thinking_length > optimal_length and thinking_length <= max_length:
            return 1.0

        excess = (thinking_length - max_length) / max_length
        return max(0.9, 1.0 - 0.1 * excess)
        
    except Exception as e:
        print(f"thinking_length_reward error: {e}")
        return 0.0


def compute_score(
    predict_str: str, 
    ground_truth: str, 
    image_selection_index: Optional[int] = None,
    original_problem: Optional[str] = None,
    bert_score: Optional[Union[float, int]] = None,
    format_weight: float = 0.10, 
    accuracy_weight: float = 0.6,
    image_selection_weight: float = 0.15,
    thinking_length_weight: float = 0.05,  
    bert_reward_weight: float = 0.0,  
    min_thinking_length: int = 300,  
    optimal_thinking_length: int = 450, 
    max_thinking_length: int = 600  
) -> Dict[str, float]:

    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward(predict_str, ground_truth)
    thinking_length_score = thinking_length_reward(
        predict_str, min_thinking_length, optimal_thinking_length, max_thinking_length
    )
    

    if bert_score is not None and bert_reward_weight > 0:
        bert_reward = float(bert_score)  
        
        image_score = 0.0
        if image_selection_index is not None:
            image_score = image_selection_reward(predict_str, image_selection_index)

            total_weight = format_weight + accuracy_weight + image_selection_weight + thinking_length_weight + bert_reward_weight
            norm_format_weight = format_weight / total_weight
            norm_accuracy_weight = accuracy_weight / total_weight
            norm_image_selection_weight = image_selection_weight / total_weight
            norm_thinking_length_weight = thinking_length_weight / total_weight
            norm_bert_reward_weight = bert_reward_weight / total_weight
            
            overall_score = (
                norm_format_weight * format_score + 
                norm_accuracy_weight * accuracy_score + 
                norm_image_selection_weight * image_score +
                norm_thinking_length_weight * thinking_length_score +
                norm_bert_reward_weight * bert_reward
            )
        else:
            total_weight = format_weight + accuracy_weight + thinking_length_weight + bert_reward_weight
            norm_format_weight = format_weight / total_weight
            norm_accuracy_weight = accuracy_weight / total_weight
            norm_thinking_length_weight = thinking_length_weight / total_weight
            norm_bert_reward_weight = bert_reward_weight / total_weight
            
            overall_score = (
                norm_format_weight * format_score + 
                norm_accuracy_weight * accuracy_score + 
                norm_thinking_length_weight * thinking_length_score +
                norm_bert_reward_weight * bert_reward
            )
            image_score = -1
            
        return {
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "image_selection": image_score,
            "thinking_length": thinking_length_score, 
            "bert": bert_reward  
        }
    else:

        image_score = 0.0
        if image_selection_index is not None:
            image_score = image_selection_reward(predict_str, image_selection_index)
            
            total_weight = format_weight + accuracy_weight + image_selection_weight + thinking_length_weight
            norm_format_weight = format_weight / total_weight
            norm_accuracy_weight = accuracy_weight / total_weight
            norm_image_selection_weight = image_selection_weight / total_weight
            norm_thinking_length_weight = thinking_length_weight / total_weight
            
            overall_score = (
                norm_format_weight * format_score + 
                norm_accuracy_weight * accuracy_score + 
                norm_image_selection_weight * image_score +
                norm_thinking_length_weight * thinking_length_score
            )
        else:
            total_weight = format_weight + accuracy_weight + thinking_length_weight
            norm_format_weight = format_weight / total_weight
            norm_accuracy_weight = accuracy_weight / total_weight
            norm_thinking_length_weight = thinking_length_weight / total_weight
            
            overall_score = (
                norm_format_weight * format_score + 
                norm_accuracy_weight * accuracy_score +
                norm_thinking_length_weight * thinking_length_score
            )
            image_score = -1
        
        return {
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "image_selection": image_score,
            "thinking_length": thinking_length_score 
        } 