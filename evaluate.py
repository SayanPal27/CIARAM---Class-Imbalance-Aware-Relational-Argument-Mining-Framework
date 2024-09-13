class BatchEvaluator:
    def __init__(self):
        # self.all_true_comp_tuples = []
        # self.all_pred_comp_tuples = []
        self.all_true_rel_tuples = []
        self.all_pred_rel_tuples = []


    # def add_batch(self, true_comp, true_rel, pred_comp, pred_rel):
    #     for i in range(len(true_comp)):
    #         # true_comp_tuples = true_comp[i]
    #         # pred_comp_tuples = pred_comp[i]

    #         # self.all_true_comp_tuples.extend(true_comp_tuples)
    #         # self.all_pred_comp_tuples.extend(pred_comp_tuples)

    #         true_rel_tuples = true_rel[i]
    #         pred_rel_tuples = pred_rel[i]

    #         self.all_true_rel_tuples.extend(true_rel_tuples)
    #         self.all_pred_rel_tuples.extend(pred_rel_tuples)

    def add_batch(self, true_rel, pred_rel):
        for i in range(len(true_rel)):
            # true_comp_tuples = true_comp[i]
            # pred_comp_tuples = pred_comp[i]

            # self.all_true_comp_tuples.extend(true_comp_tuples)
            # self.all_pred_comp_tuples.extend(pred_comp_tuples)

            true_rel_tuples = true_rel[i]
            pred_rel_tuples = pred_rel[i]

            self.all_true_rel_tuples.extend(true_rel_tuples)
            self.all_pred_rel_tuples.extend(pred_rel_tuples)

    # def evaluate(self):
    #     # Convert to sets to remove duplicates and enable direct comparison
    #     # all_true_comp_tuples = list(set(self.all_true_comp_tuples))
    #     # all_pred_comp_tuples = list(set(self.all_pred_comp_tuples))
    #     all_true_rel_tuples = list(set(self.all_true_rel_tuples))
    #     all_pred_rel_tuples = list(set(self.all_pred_rel_tuples))

    #     # Calculate precision, recall, and F1 for components
    #     # correct_comp_tuples = set(all_true_comp_tuples) & set(all_pred_comp_tuples)
    #     # comp_precision = len(correct_comp_tuples) / len(all_pred_comp_tuples) if all_pred_comp_tuples else 0
    #     # comp_recall = len(correct_comp_tuples) / len(all_true_comp_tuples) if all_true_comp_tuples else 0
    #     # comp_f1 = (2 * comp_precision * comp_recall / (comp_precision + comp_recall)) if (comp_precision + comp_recall) else 0

    #     # Calculate precision, recall, and F1 for relations
    #     correct_rel_tuples = set(all_true_rel_tuples) & set(all_pred_rel_tuples)
    #     rel_precision = len(correct_rel_tuples) / len(all_pred_rel_tuples) if all_pred_rel_tuples else 0
    #     rel_recall = len(correct_rel_tuples) / len(all_true_rel_tuples) if all_true_rel_tuples else 0
    #     rel_f1 = (2 * rel_precision * rel_recall / (rel_precision + rel_recall)) if (rel_precision + rel_recall) else 0

    #     return {
    #         'component_precision': comp_precision,
    #         'component_recall': comp_recall,
    #         'component_f1': comp_f1,
    #         'relation_precision': rel_precision,
    #         'relation_recall': rel_recall,
    #         'relation_f1': rel_f1
    #     }

    def calculate_f1_for_class(true_rel_tuples, pred_rel_tuples):
        correct_rel_tuples = set(true_rel_tuples).intersection(set(pred_rel_tuples))
    
        # Precision
        if pred_rel_tuples:
            precision = len(correct_rel_tuples) / len(pred_rel_tuples)
        else:
            precision = 0
    
        # Recall
        if true_rel_tuples:
            recall = len(correct_rel_tuples) / len(true_rel_tuples)
        else:
            recall = 0
    
        # F1 Score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
    
        return precision, recall, f1


    def evaluate(self):
        # Convert to sets to remove duplicates and enable direct comparison
        # all_true_comp_tuples = list(set(self.all_true_comp_tuples))
        # all_pred_comp_tuples = list(set(self.all_pred_comp_tuples))
        all_true_rel_tuples = list(set(self.all_true_rel_tuples))
        all_pred_rel_tuples = list(set(self.all_pred_rel_tuples))

        true_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)

        for rel in all_true_rel_tuples:
            true_by_class[rel[0]].append(rel)

        for rel in all_pred_rel_tuples:
            pred_by_class[rel[0]].append(rel)

        # Calculate F1 for each class
        f1_scores = []
        for class_label in set(true_by_class.keys()).union(set(pred_by_class.keys())):
            precision, recall, f1 = calculate_f1_for_class(true_by_class[class_label], pred_by_class[class_label])
            f1_scores.append(f1)
    
        # Output for each class (optional)
        print(f"Class: {class_label}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall: {recall:.4f}")
        print(f" F1 Score: {f1:.4f}")
        print('')

        # Calculate precision, recall, and F1 for components
        # correct_comp_tuples = set(all_true_comp_tuples) & set(all_pred_comp_tuples)
        # comp_precision = len(correct_comp_tuples) / len(all_pred_comp_tuples) if all_pred_comp_tuples else 0
        # comp_recall = len(correct_comp_tuples) / len(all_true_comp_tuples) if all_true_comp_tuples else 0
        # comp_f1 = (2 * comp_precision * comp_recall / (comp_precision + comp_recall)) if (comp_precision + comp_recall) else 0

        # True positives (correctly predicted relations)
        # correct_rel_tuples = set(all_true_rel_tuples) & set(all_pred_rel_tuples)

        # # Precision, recall, F1 for class 1 (relations predicted)
        # rel_precision = len(correct_rel_tuples) / len(all_pred_rel_tuples) if all_pred_rel_tuples else 0
        # rel_recall = len(correct_rel_tuples) / len(all_true_rel_tuples) if all_true_rel_tuples else 0
        # rel_f1 = (2 * rel_precision * rel_recall / (rel_precision + rel_recall)) if (rel_precision + rel_recall) else 0

        # # For class 2 (non-relations, the complement of relations)
        # # These are relations that were missed (false negatives) or falsely predicted (false positives)
        # false_true_rel_tuples = set(all_true_rel_tuples) - set(all_pred_rel_tuples)
        # false_pred_rel_tuples = set(all_pred_rel_tuples) - set(all_true_rel_tuples)

        # # Precision, recall, F1 for class 2 (non-relations)
        # non_rel_precision = len(false_true_rel_tuples) / len(all_pred_rel_tuples) if all_pred_rel_tuples else 0
        # non_rel_recall = len(false_true_rel_tuples) / len(all_true_rel_tuples) if all_true_rel_tuples else 0
        # non_rel_f1 = (2 * non_rel_precision * non_rel_recall / (non_rel_precision + non_rel_recall)) if (non_rel_precision + non_rel_recall) else 0

        # # Macro F1 score is the average of F1 scores for both classes
        # macro_f1 = (rel_f1 + non_rel_f1) / 2

        # # Store macro F1 score
        # # print(f"Macro F1 Score: {macro_f1}")

        # Calculate Macro F1 Score
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0



        return {
            # 'component_precision': comp_precision,
            # 'component_recall': comp_recall,
            # 'component_f1': comp_f1,
            'relation_precision': rel_precision,
            'relation_recall': rel_recall,
            'relation_f1': rel_f1,
            'relation_macro_f1': macro_f1
        }

# end


