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

# Define labels (Only Support and Attack)
LABELS = ["Support", "Attack"]

# Define dataset path (Only one dataset)
DATASET_PATH = "../datasets/debate_normative_pairs-bi/debate_normative_pairs-bi_test.json"

def generate_target_sentence(input_sentence):
    """
    Generate the relationship between argument pairs and return only the relation word (Support or Attack).
    """
    prompt = f"""
    Classify the relationship between the arguments in the format [Arg1][Rel][Arg2].
    Use only one of these labels: Support, Attack.

    ## Example Format: 
    ###Input: [Arg1][][Arg2]
    ###Output: [Arg1][Rel][Arg2]

    Here are some examples:

    #Example 1:
    Input sentence :  [The United States should lower its drinking age from 21 to 18.][][Lowering the drinking age increases use and problems]
    Target sentence :  [The United States should lower its drinking age from 21 to 18.][Attack][Lowering the drinking age increases use and problems]

    #Example 2:
    Input sentence :  [Plastic shopping bags should be banned.][][That wouldn"t do anything to help the environment.]
    Target sentence :  [Plastic shopping bags should be banned.][Attack][That wouldn"t do anything to help the environment.]

    #Example 3:
    Input sentence :  [Palestinian statehood should be recognized in the United Nations.][][Push for UN recognition IS diplomacy, doesn"t undermine it.]
    Target sentence :  [Palestinian statehood should be recognized in the United Nations.][Support][Push for UN recognition IS diplomacy, doesn"t undermine it.]

    #Example 4:
    Input sentence :  [ICC is good. The US and other countries should join the ICC.][][UN SC can block ICC prosecutions for security reasons]
    Target sentence :  [ICC is good. The US and other countries should join the ICC.][Support][UN SC can block ICC prosecutions for security reasons]

    #Example 5:
    Input sentence :  [Compulsory national service is a good idea.][][Military service gives youth great responsibilities]
    Target sentence :  [Compulsory national service is a good idea.][Support][Military service gives youth great responsibilities]

    #Example 6:
    Input sentence :  [Prostitution should be legal.][][Prostitution is no more immoral than a one night stand.]
    Target sentence :  [Prostitution should be legal.][Support][Prostitution is no more immoral than a one night stand.]

    #Example 7:
    Input sentence :  [Hugo Chavez should focus on the private sector more than social spending.][][The experience of the USSR shows that mutual support helps to achieve outstanding results]
    Target sentence :  [Hugo Chavez should focus on the private sector more than social spending.][Attack][The experience of the USSR shows that mutual support helps to achieve outstanding results]

    #Example 8:
    Input sentence :  [The US government should fund embryonic stem cell research.][][There may be major obstacles to stem-cell use in therapy and transplant treatments,]
    Target sentence :  [The US government should fund embryonic stem cell research.][Attack][There may be major obstacles to stem-cell use in therapy and transplant treatments,]

    #Example 9:
    Input sentence :  [We should have regular "Earth Hours" and they should become compulsory.][][People also use torches.]
    Target sentence :  [We should have regular "Earth Hours" and they should become compulsory.][Attack][People also use torches.]

    #Example 10:
    Input sentence :  [The US should create a public insurance option and hybrid public-private health care system.][][Private insurers offer sufficient choices]
    Target sentence :  [The US should create a public insurance option and hybrid public-private health care system.][Attack][Private insurers offer sufficient choices]

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
    match = re.search(r"\[.*?\]\[(Support|Attack)\]\[.*?\]", decoded_output)
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
    Process the dataset and generate outputs for each input pair.
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

# Run the process for the single dataset
dataset_result = process_dataset(DATASET_PATH)

# Save results to a JSON file
with open("output_results_norm10.json", "w") as output_file:
    json.dump(dataset_result, output_file, ensure_ascii=False, indent=4)
