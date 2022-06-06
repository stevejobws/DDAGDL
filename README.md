# DDAGDL
## paper "A geometric deep learning framework for drug repositioning over heterogeneous information networks"

### 'data' directory
Contain B-Dataset, C-dataset and F-dataset.

### main.py
To obtain train and test data, run
  - python main.py 
  - -d is dataset selection, including B-dataset, C-dataset and F-dataset.
  - -n is the times of cross-validation

###
To predict drug-disease associations by DDAGDL, run
  - python main.py

### Options
See help for the other available options to use with *DDAGDL*
  - python main.py --help

### Requirements
DDAGDL is tested to work under Python 3.6.0+  
The required dependencies for DDAGDL are Keras, PyTorch, TensorFlow, numpy, pandas, scipy, and scikit-learn.

### Contacts
If you have any questions or comments, please feel free to email BoWei Zhao (stevejobwes@gmail.com).
