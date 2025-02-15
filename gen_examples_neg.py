import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
# Load the pre-trained Flan-T5 model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "../llama-3.1-8b-instruct"  # Use a generative model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

def generate_examples(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(inputs.input_ids,  max_length=2048, num_return_sequences=1, repetition_penalty=1.2, temperature = 0.1)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)
    return explanation

def generate_counterargument(argument):
    """
    Generate a counterargument for the given argument using the Flan-T5 model.

    Args:
        argument (str): The argument for which to generate a counterargument.

    Returns:
        str: The generated counterargument.
    """
    # Format the input prompt for Flan-T5
    input_text = f"""Generate exact opposite of the following argument: {argument}  \n
    The format of the generated statement should be ["Generated Statement"] (answer should be within square brackets)\n
    Please do not provide any explaination or code\n\n"""
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)  # Move input IDs to the device

    # Generate counterargument
    with torch.no_grad():
        outputs = model.generate(input_ids,max_length=128)

    counterargument = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Regular expression to match the first occurrence of text within square brackets
    pattern = r'\[(.*?)\]'

# Find the first match
    matches = re.findall(pattern, counterargument)

# Display the first match
    if len(matches)>1:
        return matches[1]

    return ""

def process_dataset():
    """
    Process the dataset to balance the number of 'Support' and 'Attack' examples by generating new examples for the underrepresented class.
    """
    dataset = load_data()

    support_count = 0
    attack_count = 0
    none_count = 0

    # First, count the number of Support and Attack examples
    for example in dataset:
        if example["rel"] == "Support":
            support_count += 1
        elif example["rel"] == "Attack":
            attack_count += 1
        else:
            none_count += 1

    # Determine which class is underrepresented
    if support_count > attack_count:
        generate_for = "Attack"
        deficit = support_count - attack_count
    else:
        generate_for = "Support"
        deficit = attack_count - support_count

    # Generate examples to balance the dataset
    new_examples = []
    for example in tqdm(dataset):
        id_ = example["id"]
        arg1 = example["Arg1"]
        arg2 = example["Arg2"]
        rel = example["rel"]

        if generate_for == "Attack" and rel == "Support":
            # Generate counterargument for Arg1
            counter_arg1 = generate_counterargument(arg2)
            new_example = {
                "id": id_,
                "Arg1": arg1,
                "Arg2": counter_arg1,
                "rel": "Attack"
            }
            new_examples.append(new_example)
        elif generate_for == "Support" and rel == "Attack":
            # Generate counterargument for Arg2
            counter_arg1 = generate_counterargument(arg2)
            new_example = {
                "id": id_,
                "Arg1": arg1,
                "Arg2": counter_arg1,
                "rel": "Support"
            }
            new_examples.append(new_example)

        # Stop when enough examples are generated
        if len(new_examples) >= deficit:
            break

    # Print the new examples in JSON format, without any additional information
    for example in new_examples:
        print(json.dumps(example, ensure_ascii=False, indent=4))


def load_data():
    json_filename = f"./datasets/presidential_final/presidential_final_test.json"
    with open(json_filename, "r") as json_file:
        dataset = json.load(json_file)
    return dataset

process_dataset()
