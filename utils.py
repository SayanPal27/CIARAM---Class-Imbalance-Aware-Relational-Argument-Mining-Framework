import re

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
#                 formatted_output += f"| {relation_type.capitalize()} = {tail_text} "

#         formatted_output += "]"
#         prev_end = comp_end

#     # Add any remaining text after the last component
#     formatted_output += text[prev_end:]

#     target_output = " ".join(formatted_output.split())  # Remove extra spaces

#     # ADD OUTCOMES ENTITIES
#     outcome_entities = [entity for item in entities if item['type'] == 'outcome' for entity in item['entity']]
#     outcome_string = ', '.join(outcome_entities)
#     outcomes = f"Outcomes: (({outcome_string}))"

#     target_output = f"{target_output}\n{outcomes}"

#     print (target_output)
#     print ("\n")
#     return target_output

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
#     print("\n")
#     return target_output




# def prepare_data(dataset):
#     input_sentences = []
#     target_sentences = []

#     for example in dataset:
#         input_sen = example["paragraph"]
#         target_sen = generate_anl_end_to_end(example["paragraph"], example["components"], example["relations"], example["entities"])

#         input_sentences.append(input_sen)
#         target_sentences.append(target_sen)

#     return input_sentences, target_sentences


def prepare_data(dataset):
    input_sentences = []
    target_sentences = []

    for example in dataset:
        input_sen = "The relationship between \"" + example["Arg1"] + "\" and \"" + example["Arg2"] + "\" is : " 
        target_sen = "The relationship between \"" + example["Arg1"] + "\" and \"" + example["Arg2"] + "\" is : \"" + example["rel"] + "\""

        print("Input sentence : ", input_sen)
        print("\n")
        input_sentences.append(input_sen)
        print("Target sentence : ", target_sen)
        print("\n")
        target_sentences.append(target_sen)

# # Post-processing the anl structure------------------------------------------------------------

# def decode_anl(formatted_text):

#     formatted_text = re.sub(r'\](\W)', r'] \1', formatted_text)

#     comp_pattern = re.compile(r'\[(.*?)\|(.*?)\]')
    
#     # Find all matches of the component pattern in the formatted text
#     matches = comp_pattern.findall(formatted_text)
    
#     components = []
#     relations = []

#     for match in matches:
#         comp_str = match[0].strip()  # Text inside the brackets
#         comp_type_relations = match[1].strip().split('|')  # Type and relations

#         comp_type = comp_type_relations[0].strip()
#         if (len(comp_type.split(" ")) > 1):
#             comp_type = comp_type.split(" ")[0]
            
#         comp_relations = [rel.strip() for rel in comp_type_relations[1:]]

#         # Store the component details
#         components.append({
#             'span': comp_str,
#             'type': comp_type,
#             'relations': comp_relations
#         })

#     # Create relations based on extracted components and relations
#     for component in components:
#         for rel in component['relations']:
#             rel_match = re.match(r'(\w+)\s*=\s*(.*)', rel)
#             if rel_match:
#                 rel_type = rel_match.group(1).strip()
#                 rel_target_span = rel_match.group(2).strip()
                
#                 # Find the target component by span
#                 target_component = next((comp for comp in components if comp['span'] == rel_target_span), None)
#                 if target_component:
#                     relations.append((
#                         (component['span'], component['type']),
#                         rel_type,
#                         (target_component['span'], target_component['type'])
#                     ))

#     component_tuples = [(comp['type'], comp['span']) for comp in components]

#     # print("Extracted components:", component_tuples)
#     # print("Extracted relations:", relations)
#     return component_tuples, relations


def decode_anl(formatted_text):
    formatted_text = re.sub(r'\](\W)', r'] \1', formatted_text)

    comp_pattern = re.compile(r'\[(.*?)\|(.*?)\]')
    
    # Find all matches of the component pattern in the formatted text
    matches = comp_pattern.findall(formatted_text)
    
    components = []
    relations = []

    for match in matches:
        comp_str = match[0].strip()  # Text inside the brackets
        comp_type_relations = match[1].strip().split(' ')  # Type and relations

        comp_type = comp_type_relations[0].strip()
        comp_relations = [" ".join([rel.strip() for rel in comp_type_relations[1:]])]

        # print (comp_relations)

        # Store the component details
        components.append({
            'span': comp_str,
            'type': comp_type,
            'relations': comp_relations
        })

    # Create relations based on extracted components and relations
    for component in components:
        for rel in component['relations']:
            rel_match = re.match(r'(\w+)\s*\|\s*(.*)', rel)
            if rel_match:
                rel_type = rel_match.group(1).strip()
                rel_target_span = rel_match.group(2).strip()
                
                # Find the target component by span
                target_component = next((comp for comp in components if comp['span'] == rel_target_span), None)
                if target_component:
                    relations.append((
                        (component['span'], component['type']),
                        rel_type,
                        (target_component['span'], target_component['type'])
                    ))

    component_tuples = [(comp['type'], comp['span']) for comp in components]

    # print("Extracted components:", component_tuples)
    # print("Extracted relations:", relations)
    return component_tuples, relations


