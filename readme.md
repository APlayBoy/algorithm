# Algorithm

自己参考别人的博客和github, 动手实现了一些常用的机器学习算法和深度学习算法，仅供学习讨论使用，如有错误和不足指出，还望指出！


其中机器学习模型有以下模型，基于numpy实现。
```
├── ml
│   ├── boosting
│   │   ├── adaboost.py
│   │   ├── gbdt.py
│   │   ├── xgboost.py
│   ├── classfication
│   │   ├── nvtive_bayes.py
│   │   ├── svm.py
│   ├── cluster
│   │   ├── gmm.py
│   │   ├── lda.py
│   ├── ensemble
│   │   ├── random_forest.py
│   ├── hmm
│   │   ├── base_hmm.py
│   │   ├── log_hmm.py
│   ├── linear_model
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── softmax_regression.py
│   ├── cluster
│   │   ├── cart.py
│   │   ├── decision_tree.py
```
其中深度学习模型有以下模型,基于pytorch实现
```
├── dl
│   ├── fasttext
│   │   ├── __init__.py
│   │   ├── fasttext.py
│   │   ├── huffman.py
│   ├── image_classfication
│   │   ├── alex_net.py
│   │   ├── darknet.py
│   │   ├── densenet.py
│   │   ├── googlenet.py
│   │   ├── inceptionv3.py
│   │   ├── lenet.py
│   │   ├── nin.py
│   │   ├── resnet.py
│   │   ├── vgg.py
│   │   ├── zfnet.py
│   ├── rcnn
│   │   ├── _utils.py
│   │   ├── backbone_utils.py
│   │   ├── faster_rcnn.py
│   │   ├── generalized_recnn.py
│   │   ├── image_list.py
│   │   ├── keypoint_rcnn.py
│   │   ├── mask_rcnn.py
│   │   ├── roi_heads.py
│   │   ├── transform.py
│   ├── segmentation
│   │   ├── models
│   │   │   ├── bisenet.py
│   │   │   ├── ccnet.py
│   │   │   ├── cgnet.py
│   │   │   ├── danet.py
│   │   │   ├── deeplabv3.py
│   │   │   ├── deeplaabv3_plus.py
│   │   │   ├── denseaspp.py
│   │   │   ├── dfanet.py
│   │   │   ├── dunet.py
│   │   │   ├── encnet.py
│   │   │   ├── enet.py
│   │   │   ├── espnet.py
│   │   │   ├── fcn.py
│   │   │   ├── fcnv2.py
│   │   │   ├── hrnet.py
│   │   │   ├── icnet.py
│   │   │   ├── lednet.py
│   │   │   ├── model_store.py
│   │   │   ├── model_zoo.py
│   │   │   ├── ocnet.py
│   │   │   ├── psanet.py
│   │   │   ├── psanet_old.py
│   │   │   ├── pspnet.py
│   │   │   ├── segbase.py
│   ├── yolo
│   │   ├── slim_yolo_v2.py
│   │   ├── tiny_yolo_v3.py
│   │   ├── tools.py
│   │   ├── yolo_v2.py
│   │   ├── yolo_v3.py

```
