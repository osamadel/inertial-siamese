def ROC(classifier, x, y):
    import numpy as np
    pred = classifier.predict(x)
    b_labels_test = np.array(y)
    FAR = []
    FRR = []
    for th in range(0, 10001, 1):
        ths = th/10000.0
        Y = pred[b_labels_test == 0]
        FAR.append(np.sum(Y >= ths) / len(Y))
        X = pred[b_labels_test == 1]
        FRR.append(np.sum(X < ths) / len(X))
    FARs = np.array(FAR)
    FRRs = np.array(FRR)
    EER_index = np.argmin(np.abs(FARs - FRRs))
    EER = min(FAR[EER_index],FRR[EER_index])
    EER_th = EER_index/10000
    
    return FRR, FAR, EER, EER_th