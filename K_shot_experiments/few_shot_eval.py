import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

# Load the pre-trained LLaMA model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "../../llama-3.1-8b-instruct"  # Use a generative model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

def generate_target_sentence(input_sentence):
    """
    Given a pair of arguments of the format [Arg1][][Arg2] classify the relation between them as "Support", "Attack" or "None" and populate in the output with this format: [Arg1][Rel][Arg2]. Please note, the argument span boundaries must be same for both input and output for all components. Spans must be extractive, not generative. Here are the following examples:

    Args:
        input_sentence (str): The formatted ###Input.

    Returns:
        str: The generated ###Output.
    """
    # Format the prompt for LLaMA
    testdata = "debate"

    debex = f"""
    # Example 1:
    
    ###Input:  [ Not passing $700b bailout risks sending economy into major recession ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]
    
    ###Output:  [ Not passing $700b bailout risks sending economy into major recession ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    # Example 2:

    ###Input :  [ $700b bailout helps avoid widespread bankruptcies/layoffs ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    ###Output :  [ $700b bailout helps avoid widespread bankruptcies/layoffs ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    # Example 3:

    ###Input :  [ $700b bailout is generally well designed to solve US economic crisis ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    ###Output :  [ $700b bailout is generally well designed to solve US economic crisis ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    # Example 4:
    
    ###Input :  [ $700b bailout offers buyer for frozen mortgages; restores liquidity ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    ###Output :  [ $700b bailout offers buyer for frozen mortgages; restores liquidity ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    # Example 5:
    
    ###Input :  [ Most economists support the $700b US economic bailout plan ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    ###Output :  [ Most economists support the $700b US economic bailout plan ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    # Example 6:

    ###Input :  [ $700b bailout is important to stabilize volatile global markets ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    ###Output :  [ $700b bailout is important to stabilize volatile global markets ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    # Example 7:
    
    ###Input :  [ $700b bailout must be implemented immediately to avoid crisis ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    ###Output :  [ $700b bailout must be implemented immediately to avoid crisis ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    # Example 8:
    
    ###Input :  [ $700 it is more important to pass a plan than for it to be perfect. ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    ###Output :  [ $700 it is more important to pass a plan than for it to be perfect. ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    # Example 9:
    
    ###Input :  [ $700b bailout is consistent with US government interventionism ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]
    
    ###Output :  [ $700b bailout is consistent with US government interventionism ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    # Example 10:
    
    ###Input :  [ Deregulation and free-market ideologies caused US economic crisis ][][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]

    ###Output :  [ Deregulation and free-market ideologies caused US economic crisis ][Support][ The $700 billion bailout plan for the 2008 US financial crisis is a good idea. ]


    """

    input_prompt = f"""Given a pair of arguments of the format [Arg1][][Arg2] classify the relation between them as "Support", "Attack" or "None" and populate in the output with this format: [Arg1][Rel][Arg2]. Please note, the argument span boundaries must be same for both input and output for all components. Spans must be extractive, not generative. Here are the following examples:
    
    """

    real_data = f"""

    ### Real Data ###

    ###Input: {input_sentence}
  
    ### Output:

    #NOTE: Only give the output in the same format. No unnecessary texts or explanations please.
    """

    input_text = input_prompt + debex + real_data
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate ###Output
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=2048)

    target_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    target_sentence = target_sentence.split("#NOTE: Only give the output in the same format. No unnecessary texts or explanations please.")[-1].strip()
    # target_sentence = target_sentence.split("\n\n")[0].strip()
    print (target_sentence)
    print (f"--------------------------------------------------------------------------------------------------------\n")


    # # Extract the sentence within square brackets
    # pattern = r'\[(.*?)\]'
    # matches = re.findall(pattern, target_sentence)

    # return matches[0] if matches else target_sentence  # Return extracted text or raw output
    return target_sentence

def process_dataset():
    """
    Process the dataset to format input-output sentence pairs and generate the ###Output using LLaMA.
    """
    dataset = load_data()
    
    results = []
    for example in tqdm(dataset):
        arg1 = example["Arg1"]
        arg2 = example["Arg2"]
        rel = example["rel"]

        input_sentence = f"[{arg1}][][ {arg2} ]"
        expected_target = f"[{arg1}][{rel}][ {arg2} ]"

        # Generate the ###Output using LLaMA
        generated_target = generate_target_sentence(input_sentence)

        results.append({
            "id": example["id"],
            "input_sentence": input_sentence,
            "expected_target": expected_target,
            "generated_target": generated_target
        })

    # Print results in JSON format
    print(json.dumps(results, ensure_ascii=False, indent=4))

def load_data():
    """
    Load the dataset from JSON.
    """
    json_filename = "../datasets/debate/debate_test.json"
    with open(json_filename, "r") as json_file:
        dataset = json.load(json_file)
    return dataset

# Run the process
process_dataset()
