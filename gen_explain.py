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
'''
def generate_explaination(argument):
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
'''

# def generate_explanation(arg1, arg2, rel):
#     """
#     Generate an explanation (exactly 10 words) for the relationship between arg1 and arg2.

#     Args:
#         arg1 (str): The first argument.
#         arg2 (str): The second argument.
#         rel (str): The relationship between arg1 and arg2.

#     Returns:
#         str: The explanation generated by the model (10 words max).
#     """
#     # Input prompt formatted as specified
#     input_text = (f"""Mention in only 1 short sentence words why Argument 1 and Argument 2 have relationship "{rel}": \n"""
#               f"""Argument 1: {arg1}\nArgument 2: {arg2}\n\n"""
#               f"""Provide only the relevant words, not a full explanation. Keep the answer within 10 words. the short answer should start with Explanation: (your answer)"""
#                   f"""Explanation:""")

#     # Encode the input prompt
#     input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

#     # Generate explanation with strict constraints
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids, 
#             max_new_tokens=15,  # Allow space for up to 10 words and punctuation
#             temperature=0.7, 
#             top_p=0.9,
#             repetition_penalty=1.2  # Discourage repetitive outputs
#         )

#     # Decode the model's output
#     explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Extract the explanation after "Explanation:"
#     if "Explanation:" in explanation:
#         explanation = explanation.split("Explanation:", 1)[1].strip()

#     print("--------------------------------------------------------------------")
#     print(explanation)

#     return explanation
def generate_explanation(arg1, arg2, rel): 
    """
    Generate an explanation (exactly 10 words) for the relationship between arg1 and arg2.

    Args:
        arg1 (str): The first argument.
        arg2 (str): The second argument.
        rel (str): The relationship between arg1 and arg2.

    Returns:
        str: The explanation generated by the model (one sentence after "Explanation:").
    """
    # Input prompt formatted as specified
    input_text = (f"""Mention in only 1 short sentence words why Argument 1 and Argument 2 have relationship "{rel}": \n"""
                  f"""Argument 1: {arg1}\nArgument 2: {arg2}\n\n"""
                  f"""Provide only the relevant words, not a full explanation. Keep the answer within 10 words."""
                  f"""Explanation:""")

    # Encode the input prompt
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate explanation with strict constraints
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=25,  # Allow space for up to 10 words and punctuation
            temperature=0.7, 
            top_p=0.9,
            repetition_penalty=1.2  # Discourage repetitive outputs
        )

    # Decode the model's output
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the explanation after "Explanation:"
    if "Explanation:" in explanation:
        explanation = explanation.split("Explanation:", 1)[1].strip()
    
    # Keep only the first sentence
    explanation = explanation.split(".")[0].strip() + "."

    # print("--------------------------------------------------------------------")
    # print(explanation)

    return explanation



def process_dataset():
    """
    Process the dataset to balance the number of 'Support' and 'Attack' examples by generating new examples for the underrepresented class.
    """
    dataset = load_data()

    # Generate examples to balance the dataset
    new_examples = []
    
    for example in tqdm(dataset):
        id_ = example["id"]
        arg1 = example["Arg1"]
        arg2 = example["Arg2"]
        rel = example["rel"]

        # Generate explanation for Arg1
        exp = generate_explanation(arg1,arg2,rel)
        new_example = {
            "id": id_,
            "Arg1": arg1,
            "Arg2": arg2,
            "rel": rel,
            "exp": exp
        }
        
        #print(new_example)
        
        new_examples.append(new_example)
        
        
    # Print the new examples in JSON format, without any additional information
    for example in new_examples:
        print(json.dumps(example, ensure_ascii=False, indent=4))


def load_data():
    json_filename = f"./datasets/presidential_final_aug/presidential_final_aug_test.json"
    with open(json_filename, "r") as json_file:
        dataset = json.load(json_file)
    return dataset

process_dataset()
