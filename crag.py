INSTRUCTIONS = """Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction indicates it is not sure about the answer, "score" should be "0"; otherwise, go the next step.
3: If the Prediction exactly matches the Ground Truth, "score" is 1.
4: If the Prediction does not exactly match the Ground Truth, go through the following steps and likely give a score as 0.
5: If the Ground Truth is a number, "score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
6: If the Prediction is self-contradictory, "score" must be 0.
7: If the prediction is not answering the question, "score" must be 0.
8: If the prediction is a concise and correct summary of the ground truth, "score" is 1.
9: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
10: Otherwise, "score" is 0.

### Output a JSON blob with an "explanation" field explaining your answer as short as possible and an "score" field with value 1 or 0."""

IN_CONTEXT_EXAMPLES = """You should make the judgment based on provided examples.
Examples:

Question: "who were the last 3 prime ministers of uk?"
Ground Truth: "rishi sunak, liz truss, and boris johnson"
Prediction: "1. liz truss, 2. boris johnson, 3.theresa may"
Output: {"score": 0, "explanation": "The prediction does not contain item, rishi sunak, that is in the grount truth."}

Question: "on which days did am distribute dividends in the last year?"
Ground Truth: "2023-01-24, 2023-04-25, 2023-07-25, 2023-10-24"
Prediction: "the company distributed dividends on october 24, 2023 and january 24, 2023."
Output: {"score": 0, "explanation": "The prediction partially matches the ground truth, but it is incomplete."}

Question: "on which days did xxx distribute dividends in the last year?"
Ground Truth: "2023-01-13, 2023-03-25, 2023-11-21"
Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
Output: {"score": 0, "explanation": "The prediction contains an item, 2023-10-21, which is not in ground truth."}

Question: "who are current band members of maroon 5?"
Ground Truth: "adam Levine, jesse carmichael, james valentine, matt flynn, pj morton, sam farrar"
Prediction: "they are jesse carmichael, matt flynn, adam levine, pj morton, sam farrar, and james valentine."
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "which movies comprise the matrix franchise?"
Ground Truth: "the matrix, the matrix reloaded, the matrix revolutions, the matrix resurrections"
Prediction: "the matrix, the matrix reloaded, the matrix revolutions, the animatrix, and the matrix resurrections."
Output: {"score": 0, "explanation": "The prediction covers more items than what are given by the ground truth."}

Question: "what was the worldwide box office haul for movie x?"
Ground Truth: "101756123."
Prediction: "102 million"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "how much has spotify's user base increased by since 2020 in na?"
Ground Truth: "spotify's user base increased by 34 million since 2020."
Prediction: "spotify's north american user base increased from 36 million in 2020 to 85 million by 2021"
Output: {"score": 0, "explanation": "The prediction is not answering the question as it only gives the increase from 2020 to 2021."}
"""

from tqdm import tqdm
import re


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES
    
def attempt_api_call(client, model_name, messages, max_retries=1):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return None
def parse_response(response: str):
    """
    Return a tuple of (explanation, score) from the response, 
    where score is 0 if the prediction is wrong, 1 if the prediction is correct.

    Need to handle
    Corner case 1:
        {"explanation": ...}
        Wait, no! I made a mistake. The prediction does not exactly match the ground truth. ...
        {...}

    Corner case 2:
        {"score": 0, "explanation": "The prediction does not contain item, nick "goose" bradshaw, that is in the ground truth."}
        return a tuple of (explanation, score)
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        # Pattern to match the explanation
        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1
def evaluate_predictions(client , queries, ground_truths_list, predictions, evaluation_model_name):
    """
    Evaluates the predictions generated by a model against ground truth answers.
    
    Args:
    queries (List[str]): List of queries.
    ground_truths_list (List[List[str]]): List of lists of ground truth answers. 
        Note each query can have multiple ground truth answers.
    predictions (list): List of predictions generated by the model.
    evaluation_model_name (str): Name of the evaluation model.
    
    Returns:
    dict: A dictionary containing evaluation results.
    """
    openai_client = client
    n_miss, n_correct = 0, 0
    system_message = get_system_message()

    for _idx, prediction in enumerate(tqdm( predictions, total=len(predictions), desc="Evaluating Predictions")):
        query = queries[_idx]
        ground_truths = ground_truths_list[_idx]

        if "i don't know" in prediction.lower():
            n_miss += 1
            continue

        accuracy = -1

        for ground_truth in ground_truths:
            ground_truth_lowercase = ground_truth.lower()
            prediction_lowercase = prediction.lower()
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
                },
            ]
            if "i don't know" in prediction_lowercase:
                n_miss += 1
                continue
            elif prediction_lowercase == ground_truth_lowercase:
                n_correct_exact += 1
                n_correct += 1
                continue
            else:
                # need to use the OpenAI evaluation model to get the accuracy result (0 means wrong, 1 means correct)
                response = attempt_api_call(openai_client, evaluation_model_name, messages)
                if response:
                    _, accuracy = parse_response(response)
                    if accuracy == 1:
                        break

        if accuracy == 1:
            n_correct += 1

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_hallucination": n - n_correct - n_miss,
        "total": n,
    }
    return results

import openai
import pandas as pd
import json

HYPERBOLIC_API_KEY = "YOUR-API-KEY"
client_crag = openai.OpenAI(
    api_key=HYPERBOLIC_API_KEY,
    base_url="https://api.hyperbolic.xyz/v1",
    )

df = pd.read_feather("/kaggle/input/crag-eval-2/data.feather") # Replace with your dataset
logs = []

for i in range(200):
    log = {}
    log["query"] = df[i ,"question"]
    log["gt"] = df[i, "gt"]
    log["llm_response"] = df[i, "answer"]
    
    score = evaluate_predictions(client_crag, [log["query"]], [[log["gt"]]], [log["llm_response"]], "meta-llama/Meta-Llama-3-70B-Instruct")
    log["crag_score"] = score
    
    print(score)
    
    logs.append(log)
    with open("output.json", "w") as json_file:
        json.dump(logs, json_file, indent=4)
    