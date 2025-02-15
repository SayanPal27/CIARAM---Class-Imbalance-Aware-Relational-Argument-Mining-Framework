import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from collections import defaultdict
import os

# Load the pre-trained LLaMA model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "../../llama-3.1-8b-instruct"  # Use a generative model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

# Define labels (Including "None")
LABELS = ["Support", "Attack", "None"]

# Define dataset path
DATASET_PATH = "../datasets/presidential_final/presidential_final_test.json"

def generate_target_sentence(input_sentence):
    """
    Generate the relationship between argument pairs and return only the relation word (Support, Attack, or None).
    """
    prompt = f"""
    Classify the relationship between the arguments in the format [Arg1][Rel][Arg2].
    Use only one of these labels: Support, Attack, None.

    ## Example Format: 
    ###Input: [Arg1][][Arg2]
    ###Output: [Arg1][Rel][Arg2]

    Here are few examples:

    #Example 1:
    Input sentence :  [They’re not there.][][Number two, every three and four and five-year-old will go to school, school, not daycare, school, and with all the great universities, including the one you’ve gone to, go to, or went to, in fact, talks about in the last eight years, what’s happened.]
    Target sentence :  [They’re not there.][Support][Number two, every three and four and five-year-old will go to school, school, not daycare, school, and with all the great universities, including the one you’ve gone to, go to, or went to, in fact, talks about in the last eight years, what’s happened.]

    #Example 2:
    Input sentence :  [It’s a fact.][][It’s been totally discredited.]
    Target sentence :  [It’s a fact.][Attack][It’s been totally discredited.]

    #Example 3:
    Input sentence :  [We have an election coming up.][][You think she would rule for you?]
    Target sentence :  [We have an election coming up.][None][You think she would rule for you?]

    #Example 4:
    Input sentence :  [Mexico is heavily infected, as you know.][][And we’ve made it very, very difficult to come in because of the pandemic, and other reasons, and crime.]
    Target sentence :  [Mexico is heavily infected, as you know.][Support][And we’ve made it very, very difficult to come in because of the pandemic, and other reasons, and crime.]

    #Example 5:
    Input sentence :  [Now he says I should have closed it earlier.][][I didn’t say either of those things.]
    Target sentence :  [Now he says I should have closed it earlier.][Attack][I didn’t say either of those things.]

    ###Input: {input_sentence}

    ###Output:
    #NOTE: Only give the output in the same format. No unnecessary texts or explanations please.
    """

    # Encode input and generate response
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=1024,
            temperature=0.3,
            top_p=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and extract the target sentence
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded_output = decoded_output.split("#NOTE: Only give the output in the same format. No unnecessary texts or explanations please.")[-1].strip()
    decoded_output = decoded_output.split("\n\n")[0].strip()
    
    # Use regex to capture the relation word
    match = re.search(r"\[.*?\]\[(Support|Attack|None)\]\[.*?\]", decoded_output)
    if match:
        relation = match.group(1).strip()
        return relation  
    else:
        return "ERROR: Relation not found"

def calculate_f1_metrics(results):
    """
    Calculate precision, recall, and F1-score for each relation and the macro F1-score.
    """
    metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for result in results:
        expected = result["expected_relation"]
        generated = result["generated_relation"]

        if generated == expected:
            metrics[expected]["tp"] += 1
        else:
            metrics[generated]["fp"] += 1
            metrics[expected]["fn"] += 1

    scores = {}
    for label in LABELS:
        tp = metrics[label]["tp"]
        fp = metrics[label]["fp"]
        fn = metrics[label]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        scores[label] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(scores[label]["f1"] for label in LABELS) / len(LABELS)

    print("\n--- F1 Score Report ---")
    for label in LABELS:
        print(f"{label}: Precision = {scores[label]['precision']:.4f}, Recall = {scores[label]['recall']:.4f}, F1 = {scores[label]['f1']:.4f}")
    print(f"Macro F1 = {macro_f1:.4f}")

    return scores, macro_f1

def process_dataset(dataset_path):
    """
    Process a dataset and generate outputs for each input pair.
    """
    dataset_name = os.path.basename(dataset_path).replace("_test.json", "")
    dataset = load_data(dataset_path)
    results = []

    for example in tqdm(dataset, desc=f"Processing {dataset_name}"):
        arg1 = example["Arg1"]
        arg2 = example["Arg2"]
        rel = example["rel"]

        input_sentence = f"[{arg1}][][ {arg2} ]"
        generated_relation = generate_target_sentence(input_sentence)

        results.append({
            "dataset": dataset_name,
            "id": example["id"],
            "input_sentence": input_sentence,
            "expected_relation": rel,
            "generated_relation": generated_relation
        })

    # Calculate and display evaluation metrics
    scores, macro_f1 = calculate_f1_metrics(results)

    return {
        "dataset": dataset_name,
        "results": results,
        "f1_scores": scores,
        "macro_f1": macro_f1
    }

def load_data(json_filename):
    """
    Load the dataset from a JSON file.
    """
    with open(json_filename, "r") as json_file:
        dataset = json.load(json_file)
    return dataset

# Run the process for the dataset
dataset_result = process_dataset(DATASET_PATH)

# Save the result to a JSON file
with open("output_resultsfewpres.json", "w") as output_file:
    json.dump(dataset_result, output_file, ensure_ascii=False, indent=4)
