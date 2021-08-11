import os
import shutil
import itertools

import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

IMG_OUT_ROOT_DIR = '/home/ruiming/workspace/pro/facialExpression/data/error_analysis/affectnet_mobilenet_cleanaffectnetval_heavyonly_result'
if not os.path.exists(IMG_OUT_ROOT_DIR):
    os.makedirs(IMG_OUT_ROOT_DIR)
RESULT_FILE_PATH = '/home/ruiming/workspace/pro/facialExpression/data/error_analysis/affectnet_mobilenet_cleanaffectnetval_heavyonly_result/affectnet_mobilenet_cleanaffectnetval_heavyonly_result.txt'
CONFUSION_MATRIX_IMG_OUT_PATH = os.path.join(IMG_OUT_ROOT_DIR, 'confusion_matrix_affectnet_mobilenet_cleanaffectnetval_heavyonly_result.png')

label_map = {0:'Angry', 1:'Happy', 2:'Neutral', 3:'Sad'}
labels_name = ['Angry', 'Happy', 'Neutral', 'Sad']

def get_info_from_resultfile(file_path):
    img_paths = []
    true_labels = []
    predict_labels = []
    scores = []
    with open(file_path, 'r') as fp:
        for l in fp:
            # line format: image_path true_label predict_label Anger_predict_scores Happy_predict_scores Neutral_predict_scores Sad_predict_scores
            infos = l.strip().split()

            img_paths.append(infos[0])
            true_labels.append(int(infos[1]))
            predict_labels.append(int(infos[2]))
            scores.append([float(x) for x in infos[3:]])

    return img_paths, true_labels, predict_labels, scores

def get_accuracy(labels:list, predicts:list):
    # check if valid labels and predicts
    assert (len(labels)==len(predicts), 'label length {} != predict length {}'.format(len(labels), len(predicts)))

    # cal total accuracy
    labels = np.array(labels)
    predicts = np.array(predicts)
    accuracy = np.sum(labels == predicts) / len(labels)

    # cal category accuracy
    cat_accuracy = {}
    for cat_id in range(4):
        cat_labels = labels[labels==cat_id]
        cat_predicts = predicts[labels==cat_id]
        cat_accuracy[cat_id] = np.sum(cat_labels==cat_predicts) / len(cat_labels)
    return accuracy,cat_accuracy

def error_analysis(img_paths, true_labels, predict_labels, out_root_dir, label_map):
    """
    Copy false predicted images to out_dir
    :param img_paths:
    :param true_labels:
    :param predict_labels:
    :param out_root_dir:
    :param label_map:
    :return:
    """
    for img_path,true_label,predict_label in zip(img_paths,true_labels,predict_labels):
        if true_label==predict_label:
            continue
        # check and create dir
        cat_name = label_map[true_label]
        out_cat_dir = os.path.join(out_root_dir, cat_name)
        if not os.path.exists(out_cat_dir):
            os.makedirs(out_cat_dir)
        # copy image
        img_name = os.path.basename(img_path)
        out_img_path = os.path.join(out_cat_dir, label_map[predict_label]+'_'+img_name)
        shutil.copy(img_path, out_img_path)

def plot_confusion_matrix(cm,
                          target_names,
                          img_out_path,
                          title='Facial expression Confusion matrix',
                          cmap='Blues',
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 4))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)  # 画出x轴各点的相应标签，rotation表示标签逆时针旋转45度
        plt.yticks(tick_marks, target_names)  # 画出y轴各点的相应标签

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化,左上角特异度，右下角灵敏度

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2  # 设定矩阵颜色阈值
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")  # 将cm数值填入到混淆矩阵中

    plt.tight_layout()
    plt.ylabel('True label', size=15)  # 设置Y轴名称及字体尺寸
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), size=15)  # 设置X轴名称及字体尺寸
    plt.savefig(img_out_path, format='png', bbox_inches='tight')  # 将混淆矩阵保存为图片
    plt.show()

# generate confusion matrix
def gen_confusion_matrix(true_labels, predict_labels, labels_name, img_out_path):
    cm = confusion_matrix(true_labels, predict_labels)
    plot_confusion_matrix(cm, labels_name, img_out_path)

# get predict infos
img_paths, true_labels, predict_labels, scores = get_info_from_resultfile(RESULT_FILE_PATH)

# get accuracy
accuracy,cat_accuracy = get_accuracy(true_labels, predict_labels)
print("Total Accuracy : {}\n{} : {}\n{} : {}\n{} : {}\n{} : {}".format(accuracy,
                                                                       label_map[0], cat_accuracy[0],
                                                                       label_map[1], cat_accuracy[1],
                                                                       label_map[2], cat_accuracy[2],
                                                                       label_map[3], cat_accuracy[3]))

# error analysis
error_analysis(img_paths, true_labels, predict_labels, IMG_OUT_ROOT_DIR, label_map)

# confusion matrix
gen_confusion_matrix(true_labels, predict_labels, labels_name, CONFUSION_MATRIX_IMG_OUT_PATH)


