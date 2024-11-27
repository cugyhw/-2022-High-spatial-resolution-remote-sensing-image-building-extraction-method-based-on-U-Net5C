import numpy as np

def calc_confusion_matrix(pred, gt, cmat):
    '''
    Add confusion matrix to cmat.
    '''
    assert pred.shape == gt.shape

    cl = np.arange(cmat.shape[0])
    n_cl = len(cl)
    pred_mask = extract_masks(pred, cl, n_cl)
    gt_mask = extract_masks(gt, cl, n_cl)
    
    for ig in range(n_cl):
        gm = gt_mask[ig, :, :]
        if np.sum(gm) == 0:
            continue

        for ip in range(n_cl):
            pm = pred_mask[ip, :, :]
            if np.sum(pm) == 0:
                continue

            cmat[ig, ip] += np.sum(np.logical_and(pm, gm))

    return cmat

def calc_pixel_accuracy(cmat, ignore=None):
    _cmat = cmat.copy()
    if ignore:
        _cmat = np.delete(_cmat, ignore, axis=0)
        _cmat = np.delete(_cmat, ignore, axis=1)
    
    raw, col = cmat.shape
    count = 0

    for i in range(raw):
        count += cmat[i, i]
    
    pixel_accuracy = count / np.sum(_cmat)
    return pixel_accuracy

def calc_mean_precision(cmat, ignore=None):
    _cmat = cmat.copy()
    if ignore:
        _cmat = np.delete(_cmat, ignore, axis=0)
        _cmat = np.delete(_cmat, ignore, axis=1)

    raw, col = _cmat.shape
    class_precision = np.asarray([np.nan] * raw)

    for i in range(raw):
        n_iou = _cmat[i, i]
        n_pred = np.sum(_cmat[:, i])
        if n_pred != 0:
            class_precision[i] = n_iou / n_pred

    mean_precision = np.average(class_precision[~np.isnan(class_precision)])
    return mean_precision, class_precision

def calc_mean_recall(cmat, ignore=None):
    _cmat = cmat.copy()
    if ignore:
        _cmat = np.delete(_cmat, ignore, axis=0)
        _cmat = np.delete(_cmat, ignore, axis=1)

    raw, col = _cmat.shape
    class_recall = np.asarray([np.nan] * raw)

    for i in range(raw):
        n_iou = _cmat[i, i]
        n_gt = np.sum(_cmat[i])
        if n_gt != 0:
            class_recall[i] = n_iou / n_gt

    mean_recall = np.average(class_recall[~np.isnan(class_recall)])
    return mean_recall, class_recall

def calc_mean_IoU(cmat, ignore=None):
    _cmat = cmat.copy()
    if ignore:
        _cmat = np.delete(_cmat, ignore, axis=0)
        _cmat = np.delete(_cmat, ignore, axis=1)

    raw, col = _cmat.shape
    class_mIoU = np.asarray([np.nan] * raw)

    for i in range(raw):
        '''
        TODO 
        need to confirm which is the right calc 
        1. ignore mispredition for ignore area (current)
        2. just ignore the IoU of ignored labels
        '''
        # if i in ignore:
        #     continue
        n_iou = _cmat[i, i]
        n_union = np.sum(_cmat[i])
        if n_union > 0:
            n_union += np.sum(_cmat[:, i]) - _cmat[i, i]
            class_mIoU[i] = n_iou / n_union
            # print('\t', np.sum(_cmat[i]), np.sum(_cmat[:,i]), n_iou, class_mIoU[i])

    mean_IoU = np.average(class_mIoU[~np.isnan(class_mIoU)])
    return mean_IoU, class_mIoU

def extract_masks(seg, cl, n_cl):
    h, w  = seg.shape
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = seg == c

    return masks

def delete_ignore_label(cl, ignore):
    for c in cl:
        if c in ignore:
            cl = np.delete(cl, c)
    return cl
