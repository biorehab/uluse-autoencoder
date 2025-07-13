# Upper limb use detection using autoencoders

This repository contains code and data for analyzing upper limb use in healthy individuals and stroke patients using 3 different autoencoder-based approaches. 

### Repository structure

```bash
├── data/                       # Contains four CSV files (Healthy (right and left) & Stroke data (affected and unaffected))
├── src/                        # Contains Python scripts for autoencoder models
│   ├── autoencoder.py          # Contains the autoencoder architecture (encoder, decoder, latent space, compile model)
│   ├── utils.py                # Implements nested cross-validation for 3 approaches
│   ├── autoencoder_funcs.py    # Contains callbacks (earlystopping, LR reduction, error computation, fitting model)
│   ├── metrics.py              # Contains youden index, sensitivity, specificity calculation 
│   └── plot.py                 # Contains functions for plotting errors
    
├── approach1/                  # Contains first approach
│   ├── approach1_aff.ipynb     # Notebook containing code for running limb-specific model for stroke affected limb 
│   ├── approach1_unaff.ipynb   # Notebook containing code for running limb-specific model for stroke unaffected limb 
│   ├── approach1_right.ipynb   # Notebook containing code for running limb-specific model for healthy right limb
│   ├── approach1_left.ipynb    # Notebook containing code for running limb-specific model for healthy left limb  
│   └── approach1_generic.ipynb # Notebook containing code for running generic model containing both stroke and healthy limb data 
   
├── approach2/                  # Second approach (same structure as in approach1)
├── approach3a/                 # Third approach (variant a) (same structure as in approach1)
├── approach3b/                 # Third approach (variant b) (same structure as in approach1)
├── results/                    # Contains Youden index values from all schemes, and stats results.
└── README.md                   # Documentation
```
### Usage

To run the models, open the corresponding notebook(1, 2, 3a, 3b) for each approach and each method(limb-specific/generic). 
Each notebook contains the code needed to run a specific model. 



