from medpy.metric.binary import dc, hd95


def compute_dice_coefficient(truth_array, prediction_array, label):
    '''

    Args:
        truth_array:
        prediction_array:[[[0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  ...,  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.]],, [[0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  ...,  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.]],, [[0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  ...,  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.]],, ...,, [[0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  ...,  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.]],, [[0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  ...,  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.]],, [[0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  ...,  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.],  [0. 0. 0. ... 0. 0. 0.]]]
        label:[[[0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  ...,  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0]],, [[0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  ...,  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0]],, [[0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  ...,  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0]],, ...,, [[0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  ...,  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0]],, [[0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  ...,  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0]],, [[0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  ...,  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0]]]

    Returns:

    '''

    # 提取特定标签的二值掩码
    truth_mask = (truth_array == label)                 #[[[False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False],  ...,  [False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False]],, [[False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False],  ...,  [False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False]],, [[False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False],  ...,  [False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False]],, ...,, [[False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False],  ...,  [False False False ... False False Fal...
    prediction_mask = (prediction_array == label)       #[[[False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False],  ...,  [False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False]],, [[False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False],  ...,  [False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False]],, [[False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False],  ...,  [False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False]],, ...,, [[False False False ... False False False],  [False False False ... False False False],  [False False False ... False False False],  ...,  [False False False ... False False Fal...

    # 计算 DSC
    dice = dc(prediction_mask, truth_mask)
    return dice


def compute_hd95(truth_array, prediction_array, label):
    # 提取特定标签的二值掩码
    truth_mask = (truth_array == label)
    prediction_mask = (prediction_array == label)

    # 计算 HD95
    hd95_value = hd95(prediction_mask, truth_mask)
    return hd95_value