

class BatchEvaluator:
    def __init__(self):
        self.all_true_rel_tuples = []
        self.all_pred_rel_tuples = []

    def add_batch(self, true_rel, pred_rel):
        self.all_true_rel_tuples.extend(true_rel)
        self.all_pred_rel_tuples.extend(pred_rel)

        # print(self.all_true_rel_tuples,"\n")
        # print(self.all_pred_rel_tuples ,"\n")

            
    def evaluate(self):
        # Convert to sets to remove duplicates and enable direct comparison
        all_true_rel_tuples = self.all_true_rel_tuples
        all_pred_rel_tuples = self.all_pred_rel_tuples

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

            # print(truevalue , " -> ", predvalue , "\n")

            if(truevalue == 'Support' and predvalue == 'Support'):
                correctSupport+=1
                totalSupport+=1
                predictedSupport+=1
            elif(truevalue == 'Support' and predvalue == 'Attack'):
                totalSupport+=1
                predictedAttack+=1
            elif(truevalue == 'Support' and predvalue == 'None'):
                totalSupport+=1
                predictedNone+=1
        
            elif(truevalue=='Attack' and predvalue == 'Support'):
                totalAttack+=1
                predictedSupport+=1

            elif(truevalue=='Attack' and predvalue == 'Attack'):
                correctAttack+=1
                totalAttack+=1
                predictedAttack+=1

            elif(truevalue == 'Attack' and predvalue == 'None'):
                totalAttack+=1
                predictedNone+=1

            elif(truevalue == 'None' and predvalue == 'Support'):
                totalNone+=1
                predictedSupport+=1

            elif(truevalue == 'None' and predvalue == 'Attack'):
                totalNone+=1
                predictedAttack+=1

            elif(truevalue == 'None' and predvalue == 'None'):
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
        if(totalNone == 0):
            macroF1 = (supportF1+attackF1)/2
        else:
            macroF1 = (supportF1+attackF1+NoneF1)/3
       
        result1 = {
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
        print(result1,"\n")
        return result1

# end


