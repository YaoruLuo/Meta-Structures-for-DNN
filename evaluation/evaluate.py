import os
import sys
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, confusion_matrix

currentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(currentdir)

if __name__ == "__main__":
    order = False

    test_data_dir = currentdir + "/dataset/txt/mito/test.txt"
    prd_dir = currentdir + "/train_log/er/unet_bce_sgd_norm_20201123/results/results_epoch_100/pred"
    with open(test_data_dir, "r") as fid:
        lines = fid.readlines()

    masks_list = []
    for line in lines:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split(" ")
        masks_list.append(words[1])

    if order == True:
        masks_list = sorted(masks_list, key=lambda s: int(os.path.split(s)[1][0:-4]))

    img_num = len(masks_list)

    y_true = []
    y_scores = []

    for mask_path in masks_list:
        mask_name = os.path.splitext(os.path.split(mask_path)[1])[0]
        score_path = os.path.join(prd_dir, mask_name+".tif")
        score = cv2.imread(score_path, -1)

        # invert score if necessary
        # score = 1 - score

        print("==> Read score map: %s." % (score_path))
        label = cv2.imread(mask_path, -1) / 255
        label = label.astype(np.uint8)
        y_true.append(label.flatten())
        y_scores.append(score.flatten())


    thresholds = np.arange(0.24, 0.26, 0.01)[:]

    y_true, y_scores = np.concatenate(y_true, axis=0), np.concatenate(y_scores, axis=0)

    acc = np.zeros(len(thresholds))
    specificity = np.zeros(len(thresholds))
    sensitivity = np.zeros(len(thresholds))
    precision = np.zeros(len(thresholds))
    iou = np.zeros(len(thresholds))
    dice = np.zeros(len(thresholds))

    for indy in range(len(thresholds)):
        threshold = thresholds[indy]
        y_pred = (y_scores > threshold).astype(np.uint8)
        confusion = confusion_matrix(y_true, y_pred)
        tp = float(confusion[1, 1])
        fn = float(confusion[1, 0])
        fp = float(confusion[0, 1])
        tn = float(confusion[0, 0])

        acc[indy] = (tp + tn) / (tp + fn + fp + tn)
        sensitivity[indy] = tp / (tp + fn)
        specificity[indy] = tn / (tn + fp)
        precision[indy] = tp / (tp + fp)
        dice[indy] = 2 * sensitivity[indy] * precision[indy] / (sensitivity[indy] + precision[indy])
        sum_area = (y_pred + y_true)
        union = np.sum(sum_area == 1)
        iou[indy] = tp / float(union + tp)

        print('threshold {:.10f} ==> iou: {:.4f}, dice score: {:.4f}, acc: {:.4f},'.format(threshold,  iou[indy], dice[indy], acc[indy]))

    thred_indx = np.argmax(iou)
    m_iou = iou[thred_indx]
    m_dice = dice[thred_indx]
    m_acc = acc[thred_indx]
    m_auc = roc_auc_score(y_true, y_scores)
    print("==> Threshold: %.9f." % (thresholds[thred_indx]))
    print("==> IoU: %.4f." % (m_iou))
    print("==> dice: %.4f." % (m_dice))
    print("==> AUC: %.4f." % (m_auc))
    print("==> ACC: %.4f." % (m_acc))
    print(prd_dir)
    print(img_num)



