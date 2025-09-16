from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import torch, math

class centroid_calculate_function(torch.nn.Module):
    def __init__(self, size):
        super(centroid_calculate_function, self).__init__()
        self.param    = torch.nn.Parameter(torch.randn(size))
        self.centroid = torch.randn(size)
        self.l1loss   = torch.nn.L1Loss()

    def forward(self, input):
        optimizer = torch.optim.LBFGS([self.param], lr=0.01, max_iter= 1000)
        mask      = ~torch.eye(input.size(0)).bool()
        
        def trainer():
            optimizer.zero_grad()
            radius = torch.sqrt((self.param ** 2).sum()).unsqueeze(0)
            sim_matrix = torch.nn.functional.linear(input, self.param/radius)
            
            l1 = sim_matrix.unsqueeze(0).repeat(input.size(0),1)[mask]
            l2 = sim_matrix.unsqueeze(1).repeat(1,input.size(0))[mask]
            loss = self.l1loss(l1,l2)/2 + self.l1loss(radius, torch.ones(1).cuda())

            if loss > 100:
                return 0
            else:
                self.centroid = self.param.data
                loss.backward()
                return loss

        optimizer.step(trainer)
        radius = torch.sqrt((self.centroid ** 2).sum())
        sign   = 1 if torch.nn.functional.linear(input, self.centroid).mean() > 0 else -1
        return sign * self.centroid/radius

class ExpectedCaibrationError(torch.nn.Module):
    def __init__(self, n_bins=15):
        super(ExpectedCaibrationError, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        confidences, predictions = torch.max(logits, dim=1)
        accuracies = predictions.eq(labels)
        
        ece = torch.zeros(1, device=logits.device)
        acc = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                acc += accuracies[in_bin].float().mean() * prop_in_bin
                
        return ece * 100, acc * 100

def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    
    return err, best_th, right_index

def get_threshold(probs, grid_density):
    Min, Max = min(probs), max(probs)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(0.0 + i * 1.0 / float(grid_density))
    thresholds.append(1.1)
    return thresholds

def get_HTER_at_thr(probs, labels, thr):
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if FN + TP == 0:
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif FP + TN == 0:
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    return HTER

def calculate_threshold(probs, labels, threshold):
    TN, FN, FP, TP = eval_state(probs, labels, threshold)
    ACC = (TP + TN) / labels.shape[0]
    return ACC


def performances_val(lines, fpr_rate=0.05):
    ece = ExpectedCaibrationError()

    val_scores = []
    val_probs = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0

    for line in lines:
        try:
            count += 1
            tokens = line.split()
            score = float(tokens[0])
            label = float(tokens[1])  # int(tokens[1])
            val_scores.append(score)
            val_probs.append([1-score, score])
            val_labels.append(label)
            data.append({'map_score': score, 'label': label})
            if label==1:
                num_real += 1
            else:
                num_fake += 1
        except:
            continue
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1, drop_intermediate=False)
    auc_test = auc(fpr, tpr)

    tpr_at_fpr = tpr[np.argmin(abs(fpr - fpr_rate))]

    val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] < val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    
    FRR = 1- tpr
    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate
    val_ece, val_acc = ece(torch.tensor(np.array(val_probs)), torch.tensor(np.array(val_labels)))
    
    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index], auc_test, val_threshold, val_ece.item(), val_acc.item()

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def get_EER_states(probs, labels, grid_density=10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if FN + TP == 0:
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif FP + TN == 0:
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list

def eval(lines, p_threshold=0.5):
    ece = ExpectedCaibrationError()
    val_scores = []
    val_probs = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0

    for line in lines:
        try:
            count += 1
            tokens = line.split()
            score = float(tokens[0])
            label = float(tokens[1])  # int(tokens[1])
            val_scores.append(score)
            val_probs.append([1-score, score])
            val_labels.append(label)
            data.append({'map_score': score, 'label': label})
            if label==1:
                num_real += 1
            else:
                num_fake += 1
        except:
            continue
    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)
    
    auc_score = roc_auc_score(val_labels, val_scores)
    cur_EER_valid, threshold, _, _ = get_EER_states(val_scores, val_labels)
    ACC_threshold = calculate_threshold(val_scores, val_labels, threshold)
    cur_HTER_valid = get_HTER_at_thr(val_scores, val_labels, threshold)

    ACC_05 = calculate_threshold(val_scores, val_labels, 0.5)
    HTER_05 = get_HTER_at_thr(val_scores, val_labels, 0.5)
    if p_threshold == 0.5:
        ACC_p = ACC_05
        HTER_p = HTER_05
    
    else:
        ACC_p = calculate_threshold(val_scores, val_labels, p_threshold)
        HTER_p = get_HTER_at_thr(val_scores, val_labels, p_threshold)

    val_ece, val_acc = ece(torch.tensor(np.array(val_probs)), torch.tensor(val_labels))

    fpr, tpr, thr = roc_curve(val_labels, val_scores)
    tpr_filtered_1p = tpr[fpr <= 1 / 100]
    if len(tpr_filtered_1p) == 0:
        rate1p = 0
    else:
        rate1p = tpr_filtered_1p[-1]

    return [ACC_threshold, ACC_05, ACC_p], rate1p, [cur_HTER_valid, HTER_05, HTER_p], auc_score, threshold, val_ece.item(), val_acc.item(), val_scores, val_labels