import re

def prepare_data(dataset):
    input_sentences = []
    target_sentences = []

    for example in dataset:
        input_sen = "[" + example["Arg1"] + "][][" + example["Arg2"] + "]" 
        target_sen = "[" + example["Arg1"] + "][" +example["rel"] +"][" + example["Arg2"] + "]"

        print("Input sentence : ", input_sen)
        print("\n")
        input_sentences.append(input_sen)
        print("Target sentence : ", target_sen)
        print("\n")
        # print(len(target_sen))
        target_sentences.append(target_sen)
    return input_sentences, target_sentences


def decode_anl(pred):
    match = re.search(r'\[(.*?)\]', pred)
    if match:
        return match.group(1)  # Extract the content inside the second brackets
    else:
        return None  # Return None if the format doesn't match
    



