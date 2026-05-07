import math


"""
Visualiser class handles dynamic axis scaling for data plots and supports computing misclassified samples per iteration. Can compute internally with optional print-only mode to reduce overhead. Also tracks training/validation error convergence and updates decision boundary across iterations.
"""
class Visualiser:
    def __init__(self, N, N_val, T):
        self.N = N
        self.N_val = N_val
        self.T = T
        self.temp_sum = np.zeros(N, dtype=np.float64)
        self.final_label = np.zeros(N, dtype=np.float64)
        self.misshits = np.array([None] * (T+1), dtype=float)
        self.misshits[0] = 1.0
        
        self.temp_sum_val = np.zeros(N_val, dtype=np.float64)
        self.final_label_val = np.zeros(N_val, dtype=np.float64)
        self.misshits_val = np.array([None] * (T+1), dtype=float)
        self.misshits_val[0] = 1.0
        
        self.classification = None

        self.need_to_compute = True

       
        self.classification_boundaries = []

    def compute_dynamic_axis(self, x):
        self.x_min, self.x_max = x[:, 0].min(), x[:, 0].max()
        self.y_min, self.y_max = x[:, 1].min(), x[:, 1].max()

        margin = 0.1 * max(self.x_max - self.x_min, self.y_max - self.y_min)
        self.x_min -= margin
        self.y_min -= margin
        self.x_max += margin
        self.y_max += margin

    def compute_misshits(self, x, x_val, h, t, alpha, label, label_val):
        x = x if isinstance(x, np.ndarray) else x.to_numpy()
        x_val = x_val if isinstance(x_val, np.ndarray) else x_val.to_numpy()
        
        self.temp = h[t][2] * np.sign(x[:, int(h[t][1]) ] - h[t][0])
        self.temp_sum = np.float64(self.temp_sum + alpha[t] * self.temp)
        self.final_label = np.sign(self.temp_sum)
        self.misshits[t+1] = np.sum(np.float64(self.final_label != label)) / self.N

        self.temp_val = h[t][2] * np.sign(x_val[:, int(h[t][1]) ] - h[t][0])
        self.temp_sum_val = np.float64(self.temp_sum_val + alpha[t] * self.temp_val)
        self.final_label_val = np.sign(self.temp_sum_val)
        self.misshits_val[t+1] = np.sum(np.float64(self.final_label_val != label_val)) / self.N_val


    def print_info(self, h, t, err):
        print('BOOSTING ROUND: ' + str(t+1))
        print('WEIGHTED ERROR OF DECISION BOUNDARY: ' + str(err[t]), '--->> FEATURE, THRESHOLD, DIRECTION: ' + str( (int(h[t][1]), h[t][0], h[t][2]) ) )
        print('CURRENT ERROR: ' + str(self.misshits[t+1]) )
        print('CURRENT VAL ERROR: ' + str(self.misshits_val[t+1]) )
    
    def compute_and_print_info(self, x, x_val, h, t, err, alpha, label, label_val):
        self.compute_misshits(x, x_val, h, t, alpha, label, label_val)
        self.print_info(h, t, err)
    
    def print_info_no_compute(self, t, weighted_error, train_err, val_err):
        print(f"BOOSTING ROUND: {t + 1}")
        print(f"WEIGHTED ERROR OF DECISION BOUNDARY: {weighted_error:.4f}")
        print(f"CURRENT TRAIN ERROR: {train_err:.4f}")
        print(f"CURRENT VAL ERROR: {val_err:.4f}")


    def set_computations(self, classification, final_label, final_label_val, misshits, misshits_val):
        self.final_label = final_label
        self.final_label_val = final_label_val
        self.misshits = misshits
        self.misshits_val = misshits_val
        self.classification = classification
        self.need_to_compute = False


    def get_train_val_error(self):
        return self.misshits, self.misshits_val

  
    def visualise(self, x, x_val, h, t, err, alpha, label, label_val, weight):
        x = x if isinstance(x, np.ndarray) else x.to_numpy()
        x_val = x_val if isinstance(x_val, np.ndarray) else x_val.to_numpy()

        self.compute_dynamic_axis(x)
        
        rcParams['figure.figsize'] = 40, 6
        rcParams['font.size'] = 15
        
        f, axarr = plt.subplots(1, 6)

        # 1 Dataset Weighting Update and Decision Boundary
        
        if self.need_to_compute:
            self.compute_misshits(x, x_val, h, t, alpha, label, label_val)
            self.classification = np.sign(np.float64(alpha[t] * self.temp))
        

        scale_markers = [w / (np.max(weight) - np.min(weight)) for w in weight]
        
        # plot weighted samples
        for l in range(len(x)):
            if label[l] == 1:
                axarr[0].plot([x[l,0]],[x[l,1]], 'bo', markersize=scale_markers[l] * 10.0)
            else:
                axarr[0].plot(x[l,0],x[l,1], 'ro', markersize=scale_markers[l]* 10.0)
    
    
        if h[t][1]  == 0 :
            # Add dynamic axis limits
            axarr[0].plot((h[t][0], h[t][0]), (self.y_min, self.y_max), 'k-')
            self.classification_boundaries.append(((h[t][0], h[t][0]), (self.y_min, self.y_max)))
        else:
            axarr[0].plot((self.x_min, self.x_max), (h[t][0], h[t][0]), 'k-')
            self.classification_boundaries.append( ((self.x_min, self.x_max), (h[t][0], h[t][0])))
        
        axarr[0].set_title("Dataset Weighting Update and Decision Boundary \n", fontsize=15, alpha=0.75)
    
        # 2 Dataset Classification for Weak Classifier t

        for l in range(len(x)):
            if self.classification[l] == 1:
                axarr[1].plot([x[l,0]],[x[l,1]], 'bo')
    
            else:
                axarr[1].plot(x[l,0],x[l,1], 'ro')
    
        if h[t][1]  == 0 :
            axarr[1].plot((h[t][0], h[t][0]), (self.y_min, self.y_max), 'k-')
        else: 
            axarr[1].plot((self.x_min, self.x_max), (h[t][0], h[t][0]), 'k-')
    
        axarr[1].set_title("Dataset Classification for Weak Classifier t \n", fontsize=15, alpha=0.75)

        # 3 Final Classification of Samples from Current Strong Classifier H
        
        pos1 = np.where(self.final_label == 1)
        pos2 = np.where(self.final_label == -1)
    
        axarr[2].plot(x[pos1[0], 0], x[pos1[0], 1], 'bo')
        axarr[2].plot(x[pos2[0], 0], x[pos2[0], 1], 'ro')

        axarr[2].axis([self.x_min, self.x_max, self.y_min, self.y_max])
        
        for p1, p2 in self.classification_boundaries:
            axarr[2].plot(p1, p2, 'k-')
            
        axarr[2].set_title("Final Classification of Samples from \n Current Strong Classifier H", fontsize=15, alpha=0.75)

        # 4 Missclassified Samples from Current Strong Classifier H
        correct = np.where(self.final_label == label)
        wrong = np.where(self.final_label != label)
        
        axarr[3].plot(x[correct, 0], x[correct, 1], 'ko')
        axarr[3].plot(x[wrong, 0], x[wrong, 1], 'rv')
        axarr[3].axis([self.x_min, self.x_max, self.y_min, self.y_max])
        axarr[3].set_title("Missclassified Samples from \nCurrent Strong Classifier H", fontsize=15, alpha=0.75)
        
        # 5 Convergence Rate of the Total Error
        axarr[4].plot(self.misshits, alpha=0.75, label='Training Error')
        axarr[4].plot(self.misshits_val, color='orange', alpha=0.75, label='Validation Error')
        
        axarr[4].axis([ 0, self.T, 0, 1])
        axarr[4].set_ylabel("Error Rate", fontsize=15, alpha=0.75)
        axarr[4].set_xlabel('Boosting Rounds', fontsize=15, alpha=0.75)
        axarr[4].legend(loc='upper right', fontsize=15)
        axarr[4].grid(True)
        axarr[4].set_title("Convergence Rate of the Total Error\nFor Training and Validation Sets", fontsize=15, alpha=0.75)
        
        # 6. Decision Boundary
        dimension = 100
        f1 = np.linspace(self.x_min, self.x_max, num=dimension)
        f2 = np.linspace(self.y_min, self.y_max, num=dimension)
        f1 = np.reshape(f1, (dimension, 1))
        f2 = np.reshape(f2, (dimension, 1))
        decision_boundary_data = np.dstack(np.meshgrid(f1, f2)).reshape(-1, 2)
        axarr[5].scatter(decision_boundary_data[:, 0], decision_boundary_data[:, 1])
        
        decision_values = np.zeros(len(decision_boundary_data))
        for i in range(t + 1):
            feature_index = int(h[i][1])
            threshold = h[i][0]
            direction = h[i][2]
            decision_values += alpha[i] * direction * np.sign(decision_boundary_data[:, feature_index] - threshold)

        pos = np.where(np.sign(decision_values) == 1)
        neg = np.where(np.sign(decision_values) == -1)
        
        axarr[5].plot(decision_boundary_data[pos, 0], decision_boundary_data[pos, 1], 'bo', markersize=5, alpha=0.4)
        axarr[5].plot(decision_boundary_data[neg, 0], decision_boundary_data[neg, 1], 'ro', markersize=5, alpha=0.4)
        
        pos1 = np.where(self.final_label == 1)
        pos2 = np.where(self.final_label == -1)
        axarr[5].plot(x[pos1[0], 0], x[pos1[0], 1], 'w1', markersize=8)
        axarr[5].plot(x[pos2[0], 0], x[pos2[0], 1], 'k+', markersize=8)
        
        axarr[5].set_xlim(self.x_min, self.x_max)
        axarr[5].set_ylim(self.y_min, self.y_max)
        axarr[5].set_title("Decision Boundary", size=20)
       
        plt.subplots_adjust(hspace=0.1, wspace=0.2)

        self.print_info(h, t, err)
        plt.show()


def calculate_conf_mat(pred_prob, true_label, threshold=0.5):
    true_label = true_label.map({0:-1,1:1,-1:-1})
    true_positives = sum((pred_prob >= threshold) & (true_label == 1))
    false_positives = sum((pred_prob >= threshold) & (true_label == -1))
    true_negatives = sum((pred_prob < threshold) & (true_label == -1))
    false_negatives = sum((pred_prob < threshold) & (true_label == 1))
    return true_positives, false_positives, true_negatives, false_negatives


def calculate_tpr_fpr(pred_prob, true_label, threshold=0.5):
    tp, fp, tn, fn = calculate_conf_mat(pred_prob, true_label, threshold)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0   
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

    
def calculate_precision(pred_prob, true_label, threshold=0.5):
    tp, fp, tn, fn = calculate_conf_mat(pred_prob, true_label, threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision

    
def calculate_recall(pred_prob, true_label, threshold=0.5):
    tp, fp, tn, fn = calculate_conf_mat(pred_prob, true_label, threshold)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall

    
def calculate_f1(pred_prob, true_label, threshold=0.5):
    precision = calculate_precision(pred_prob, true_label, threshold)
    recall = calculate_recall(pred_prob, true_label, threshold)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def calculate_accuracy(pred_prob, true_label, threshold=0.5):
    tp, fp, tn, fn = calculate_conf_mat(pred_prob, true_label, threshold)
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    return accuracy


def roc_auc(pred_prob, true_label, plot=True): 
    thresholds = np.linspace(0, 1, 100)
    
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        tpr, fpr = calculate_tpr_fpr(pred_prob, true_label, threshold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
            
    auc = np.abs(np.trapezoid(tpr_list, fpr_list)).item()
    if not plot: return auc
    
    print(f"AUC: {auc}")
            
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list)
    plt.plot([0, 1], [0, 1], linestyle='--')  #default
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')   
    plt.show()


def plot_multiple_roc_curves(pred_probs, true_label):
    plt.figure(figsize=(12, 10))
    
    for key, pred_prob in pred_probs.items():
        # Generate thresholds
        thresholds = np.linspace(0, 1, 100)
        
        # Calculate TPR and FPR for each threshold
        tpr_list = []
        fpr_list = []
        for threshold in thresholds:
            tpr, fpr = calculate_tpr_fpr(pred_prob, true_label, threshold)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        auc = np.abs(np.trapezoid(tpr_list, fpr_list)).item()
        plt.plot(fpr_list, tpr_list, label=f'{key} (AUC = {auc:.3f})', alpha=0.8)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


"""
Matthew's correlation coefficient - a binary classification metric that
considers true positives, true negatives, false positives, and false negatives.
It measures the correlation between the predicted and actual labels and is
useful in reducing the entire confusion matrix to a single number.
An MCC score of 1 implies perfect classification, 0 is random, and -1 indicates complete disagreement. MCC is less sensitive to class imbalance than accuracy.
"""
def calculate_mcc(pred_prob, true_label, threshold=0.5):
    tp, fp, tn, fn = calculate_conf_mat(pred_prob, true_label, threshold)

    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0 if numerator == 0 else np.sign(numerator)

    mcc = numerator / denominator
    
    return mcc


"""
Cohen's Kappa - measures the amount of agreement between two raters.
It is more robust than a simple percent agreement, as it considers the possibility of chance agreement.
A Kappa value of 1 implies perfect agreement, 0 means the agreement is as would be expected by chance,
and nagative values imply less-than-chance agreement.
Here, Cohen's kappa is repurposed to measure the agreement between predicted and actual labels,
so the classifier acts as one rater while the ground truths act as the other.
This let's us determine how aligned the predictions are with the actual labels while considering chance.
Cohen's kappa is insensitive (albiet not as much so as MCC) to class imbalance.
"""
def calculate_cohens_kappa(pred_prob, true_labels, threshold=0.5):
    TP, FP, TN, FN = calculate_conf_mat(pred_prob, true_labels, threshold)    
    N = TP + TN + FP + FN
    Po = (TP + TN) / N
    P_yes = ((TP + FP) / N) * ((TP + FN) / N)
    P_no = ((TN + FN) / N) * ((TN + FP) / N)
    Pe = P_yes + P_no
    kappa = (Po - Pe) / (1 - Pe)
    return kappa

"""
Lift - evaluates how much better the model predicts positive instances compared to chance.
Lift is calculated as the ratio of precision to the baseline rate of positive instances, 
where precision is the proportion of true positives to all positive predictions, and the baseline precision is
the proportion of positive instances in the dataset as a whole.
A lift value above 1 implies that the model correctly classifies positive samples more often than with random guessing, while
1 is equivalent to chance, and below 1 is worse than random guessing.
"""
def calculate_lift(pred_prob, true_labels, threshold=0.5, silent=True):
    true_labels = true_labels.map({-1:0,1:1,0:0})
    # pred_labels = (pred_prob >= threshold).astype(int)
    
    # Calculate Precision
    model_precision = calculate_precision(pred_prob, true_labels, threshold)
    
    # Calculate Baseline Precision
    baseline_precision = true_labels.mean()
    
    # Calculate Lift
    lift = model_precision / baseline_precision

    if not silent:
        print(f'Model Precision: {model_precision}')
        print(f'Baseline Precision: {baseline_precision}')
        print(f'Lift: {lift}')
    
    return lift.item()


def calculate_probabilities(pred_df):
    prob_df = pred_df.copy()
        
    prob_df['margins'] = prob_df[['sum_alpha','pos_votes','neg_votes']].apply(
        lambda x: margin_calculation(*x), axis=1)
    prob_df['probabilities'] = prob_df.apply(
        lambda x: 1.0 - x.margins if x.classifications == -1 else x.margins, axis=1 )

    return prob_df


def calculate_metrics(probabilities, true_labels, threshold=0.5):
    metrics = [f(probabilities, true_labels, threshold)
               for f in (calculate_precision, calculate_recall, calculate_f1,
                         calculate_accuracy, calculate_mcc, calculate_cohens_kappa, 
                         calculate_lift)]
    metrics_dict = dict(zip(['precision', 'recall', 'f1', 'accuracy', 'mcc', 'kappa', 'lift'], metrics))
    metrics_dict['roc_auc'] = roc_auc(probabilities, true_labels, plot=False) 

    return metrics_dict
    

def get_metrics(classifier_df, X, y, threshold=0.5, dp=4, tree=False):
    if tree:
        pred = classify_and_sum_votes_tree(X, classifier_df)
    else:
        pred = classify_and_sum_votes(X, classifier_df)
    
    prob = calculate_probabilities(pred).probabilities
    metrics = calculate_metrics(prob, y, threshold=threshold)
    return {k: round(v, dp) for k,v in metrics.items()}


def calculate_metric(prob, label, metric='roc_auc', threshold=0.5):
    if metric == 'roc_auc':
        return roc_auc(prob, label, plot=False) 
    
    metrics_f = [calculate_precision, calculate_recall, calculate_f1,
                  calculate_accuracy, calculate_mcc, calculate_cohens_kappa, 
                  calculate_lift]
    metrics_dict = dict(zip(['precision', 'recall', 'f1', 'accuracy', 'mcc', 'kappa', 'lift'], metrics_f))
    
    return metrics_dict[metric](prob, label, threshold)
    


def get_metric(classifier_df, X, y, metric='roc_auc', threshold=0.5, tree=False):
    if tree:
        pred = classify_and_sum_votes_tree(X, classifier_df)
    else:
        pred = classify_and_sum_votes(X, classifier_df)
    prob = calculate_probabilities(pred).probabilities

    return calculate_metric(prob, y, metric, threshold)
