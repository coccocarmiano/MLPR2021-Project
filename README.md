## Folders Structure

```
|-|
  ├──|code 
  |  |
  |  ├── calibration.py             (Library Files)
  |  ├── classifiers.py             (Library Files)
  |  ├── dist.py                    (Generates Feature Distribution Plot)
  |  ├── FinalEvaluation.py         (Generates Evaluation Data for GMMs, Tied Covariance, RBF SVM, ...)
  |  ├── GMMOptimization.py         (Generates Optimized Parameters for GMMs)
  |  ├── KernelSVM_Normalized.py    (Generates Optimized Parameters for Normalized RBF SVM)    
  |  ├── KernelSVM.py               (Generates Optimized Parameters for RBF SVM)
  |  ├── LinearSVM.py               (Generates Optimized Parameters for Linear SVM)
  |  ├── LogReg.py                  (Generates Optimzed Parameters for Logistic Regression)  
  |  ├── LogRegQuad.py              (Generates Optimzed Parameters for Quadratic Logistic Regression)
  |  ├── model_evaluation.py        (Generates evalation data for logistic regressions and linear SVM)
  |  ├── model_validation.py        (Process generated data for optimized parameters)
  |  ├── MVG-FC-Optimization.py     (Generates Optimized Parameters for MVG Classifier)
  |  ├── MVG-NB-Optimization.py     (Generates Optimized Parameters for MVG Classifier)
  |  ├── MVG.py                     (Generates Optimized PCA for MVG Classifier)
  |  ├── MVG-TC-Optimization.py     (Generates Optimized Parameters for MVG Classifier)
  |  ├── PCA_Tests.py               (Tests to select good PCA values)
  |  ├── PolySVM_Normalized.py      (Generates Optimzed Parameters for Normalized Polynomial SVM)
  |  ├── PolySVM.py                 (Generates Optimzed Parameters for Polynomial SVM)
  |  ├── PolySVM_Whitened.py        (Generates Optimzed Parameters for WhitenedPolynomial SVM)
  |  ├── scatter.py                 (Generates scatter plots)
  |  └── utils.py                   (Library files)
  |
  ├── data                          (Datasets and scores computed by validating for GMMs, Kernel SVMs, Gaussian Classifiers)
  ├── img                           (Images for report)
  ├── report                        (Report source folder)
  └── trained                       (Scores computed by validating for linear SVMs and Logistic Regressions)
```