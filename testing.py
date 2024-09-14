import json
import re

def add_batch(self, true_rel, pred_rel):
    for i in range(len(true_rel)):
        true_rel_tuples = true_rel[i]
        pred_rel_tuples = pred_rel[i]

        self.all_true_rel_tuples.extend(true_rel_tuples)
        self.all_pred_rel_tuples.extend(pred_rel_tuples)


def evaluate(self):
        # Convert to sets to remove duplicates and enable direct comparison
    all_true_rel_tuples = list(self.all_true_rel_tuples)
    all_pred_rel_tuples = list(self.all_pred_rel_tuples)

    correctSupport = 0
    totalSupport = 0
    predictedSupport = 0

    correctAttack = 0
    totalAttack = 0
    predictedAttack = 0

    correctNone = 0
    totalNone = 0
    predictedNone = 0

    for i in range(len(all_true_rel_tuples)):
        truevalue = all_true_rel_tuples[i]
        predvalue = all_pred_rel_tuples[i]

        if(truevalue == "Support" and predvalue == "Support"):
            correctSupport+=1
            totalSupport+=1
            predictedSupport+=1
        elif(truevalue == "Support" and predvalue == "Attack"):
            totalSupport+=1
            predictedAttack+=1
        elif(truevalue == "Support" and predvalue == "None"):
            totalSupport+=1
            predictedNone+=1
        
        elif(truevalue=="Attack" and predvalue == "Support"):
            totalAttack+=1
            predictedSupport+=1

        elif(truevalue=="Attack" and predvalue == "Attack"):
            correctAttack+=1
            totalAttack+=1
            predictedAttack+=1

        elif(truevalue == "Attack" and predvalue == "None"):
            totalAttack+=1
            predictedNone+=1

        elif(truevalue == "None" and predvalue == "Support"):
            totalNone+=1
            predictedSupport+=1

        elif(truevalue == "None" and predvalue == "Attack"):
            totalNone+=1
            predictedAttack+=1

        elif(truevalue == "None" and predvalue == "None"):
            correctNone+=1
            totalNone+=1
            predictedNone+=1

        supportPrecision = correctSupport/predictedSupport if predictedSupport>0 else 0
        supportRecall = correctSupport/totalSupport if totalSupport>0 else 0
        supportF1 = (2*supportPrecision*supportRecall)/(supportPrecision+supportRecall) if (supportPrecision+supportRecall) > 0 else 0

        attackPrecision = correctAttack/predictedAttack if predictedAttack > 0 else 0
        attackRecall = correctAttack/totalAttack if totalAttack > 0 else 0
        attackF1 = (2*attackPrecision*attackRecall)/(attackPrecision+attackRecall) if (attackPrecision+attackRecall) > 0 else 0

        NonePrecision = correctNone/predictedNone if predictedNone>0 else 0
        NoneRecall = correctNone/totalNone if totalNone>0 else 0
        NoneF1 = (2*NonePrecision*NoneRecall)/(NonePrecision+NoneRecall) if(NonePrecision+NoneRecall) > 0 else 0

        macroF1 = 0
        if(totalNone> 0):
            macroF1 = (supportF1+attackF1)/2
        else:
            macroF1 = (supportF1+attackF1+NoneF1)/3
       
        return {
            'support_precision': supportPrecision,
            'support_recall': supportRecall,
            'support_f1': supportF1,
            'attack_precision': attackPrecision,
            'attack_recall': attackRecall,
            'attack_f1': attackF1,
            'none_precision': NonePrecision,
            'none_recall': NoneRecall,
            'none_f1': NoneF1,
            'macro_f1': macroF1

        }


# def generate_anl_end_to_end(text, components, relations, entities) -> str:
#     # Add IDs to components
#     index_counter = 0
#     for component in components:
#         component['id'] = index_counter
#         index_counter += 1

#     # Sort components by their start index
#     sorted_components = sorted(components, key=lambda x: x['start'])

#     # Create a dictionary to track which component has which relations
#     relation_dict = {}
#     for relation in relations:
#         relation_type = relation['type']
#         head = relation['head']
#         tail = relation['tail']
#         if head not in relation_dict:
#             relation_dict[head] = []
#         relation_dict[head].append((relation_type, tail))

#     # Generate the formatted output
#     formatted_output = ""
#     prev_end = 0  # Track the end of the previous span

#     for comp in sorted_components:
#         comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

#         # Add text before the component span
#         formatted_output += text[prev_end:comp_start]

#         component_text = text[comp_start:comp_end]
#         formatted_output += f"[ {component_text} | {comp_type} "

#         # Add relations if any
#         if comp_index in relation_dict:
#             for relation_type, tail in relation_dict[comp_index]:
#                 tail_component = next(filter(lambda x: x['id'] == tail, components))
#                 tail_text = text[tail_component['start']:tail_component['end']]
#                 formatted_output += f"{relation_type} | {tail_text} "

#         formatted_output += "]"
#         prev_end = comp_end

#     # Add any remaining text after the last component
#     formatted_output += text[prev_end:]

#     target_output = " ".join(formatted_output.split())  # Remove extra spaces

#     # ADD OUTCOMES ENTITIES
#     outcome_entities = [entity for item in entities if item['type'] == 'outcome' for entity in item['entity']]
#     outcome_string = ', '.join(outcome_entities)
#     outcomes = f"Outcomes: (({outcome_string}))"

#     target_output = f"{outcomes}\n{target_output}"

#     print(target_output)
#     print("Target \n")
#     decode_anl(target_output)
#     print("Predicted \n")
#     return target_output

def decode_anl(pred):
    relationship = pred.split('is : ')[1]
    relationship = relationship.strip('"')

    print(relationship)

    return relationship
    

# def evaluate(self):
#         # Convert to sets to remove duplicates and enable direct comparison
#         # all_true_comp_tuples = list(set(self.all_true_comp_tuples))
#         # all_pred_comp_tuples = list(set(self.all_pred_comp_tuples))
#         all_true_rel_tuples = list(set(self.all_true_rel_tuples))
#         all_pred_rel_tuples = list(set(self.all_pred_rel_tuples))

#         # Calculate precision, recall, and F1 for components
#         # correct_comp_tuples = set(all_true_comp_tuples) & set(all_pred_comp_tuples)
#         # comp_precision = len(correct_comp_tuples) / len(all_pred_comp_tuples) if all_pred_comp_tuples else 0
#         # comp_recall = len(correct_comp_tuples) / len(all_true_comp_tuples) if all_true_comp_tuples else 0
#         # comp_f1 = (2 * comp_precision * comp_recall / (comp_precision + comp_recall)) if (comp_precision + comp_recall) else 0

#         # True positives (correctly predicted relations)
#         correct_rel_tuples = set(all_true_rel_tuples) & set(all_pred_rel_tuples)

#         # Precision, recall, F1 for class 1 (relations predicted)
#         rel_precision = len(correct_rel_tuples) / len(all_pred_rel_tuples) if all_pred_rel_tuples else 0
#         rel_recall = len(correct_rel_tuples) / len(all_true_rel_tuples) if all_true_rel_tuples else 0
#         rel_f1 = (2 * rel_precision * rel_recall / (rel_precision + rel_recall)) if (rel_precision + rel_recall) else 0

#         # For class 2 (non-relations, the complement of relations)
#         # These are relations that were missed (false negatives) or falsely predicted (false positives)
#         false_true_rel_tuples = set(all_true_rel_tuples) - set(all_pred_rel_tuples)
#         false_pred_rel_tuples = set(all_pred_rel_tuples) - set(all_true_rel_tuples)

#         # Precision, recall, F1 for class 2 (non-relations)
#         non_rel_precision = len(false_true_rel_tuples) / len(all_pred_rel_tuples) if all_pred_rel_tuples else 0
#         non_rel_recall = len(false_true_rel_tuples) / len(all_true_rel_tuples) if all_true_rel_tuples else 0
#         non_rel_f1 = (2 * non_rel_precision * non_rel_recall / (non_rel_precision + non_rel_recall)) if (non_rel_precision + non_rel_recall) else 0

#         # Macro F1 score is the average of F1 scores for both classes
#         macro_f1 = (rel_f1 + non_rel_f1) / 2

#         # Store macro F1 score
#         # print(f"Macro F1 Score: {macro_f1}")


#         return {
#             # 'component_precision': comp_precision,
#             # 'component_recall': comp_recall,
#             # 'component_f1': comp_f1,
#             'relation_precision': rel_precision,
#             'relation_recall': rel_recall,
#             'relation_f1': rel_f1,
#             'relation_macro_f1': macro_f1
#         }


def prepare_data(dataset):
    input_sentences = []
    target_sentences = []

    for example in dataset:
        input_sen = "The relationship between \"" + example["Arg1"] + "\" and \"" + example["Arg2"] + "\" is : " 
        target_sen = "The relationship between \"" + example["Arg1"] + "\" and \"" + example["Arg2"] + "\" is : \"" + example["rel"] + "\""

        # print("Input sentence : ", input_sen)
        # print("\n")
        input_sentences.append(input_sen)
        # print("Target sentence : ", target_sen)
        # print("\n")
        target_sentences.append(target_sen)

        # print("Decoded :", decode_anl(target_sen))

    
    return input_sentences, target_sentences

def load_data(split):
    json_filename = f"./datasets/debate/debate_{split}.json"
    with open(json_filename, "r") as json_file:
        dataset = json.load(json_file)
    return dataset

# def prepare_data(dataset):
#     input_sentences = []
#     target_sentences = []

#     for example in dataset:
#         input_sen = example["paragraph"]
#         target_sen = generate_anl_end_to_end(example["paragraph"], example["components"], example["relations"], example["entities"])

#         input_sentences.append(input_sen)
#         target_sentences.append(target_sen)

#     return input_sentences, target_sentences

# # Load data
# def load_data(split):
#     json_filename = f"./datasets/abstrct/abstrct_{split}.json"
#     with open(json_filename, "r") as json_file:
#         dataset = json.load(json_file)
#     return dataset


train_dataset = load_data("train")
train_input_sentences, train_target_sentences = prepare_data(train_dataset)