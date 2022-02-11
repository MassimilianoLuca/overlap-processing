# Code for: Trajectory Test-Train Overlap in Next-Location Prediction Datasets

To successfully run the code, you need the following dependencies:

- pandas 1.1.4
- numpy 1.19.4
- scikit-mobility
- tqdm
- textdistance

The data folder contains the dataset of Foursquare NYC already processed. Due to the significant weights of the files and the limits for the supplementary material, we only provide one of the datasets presented in the main paper. However, in the code folder, we also provide two notebooks, namely `data_downloader.ipynb` and `data_preparation.ipynb`, to let you download and preprocess the other datasets.

The script `overlap.py` contains the logic to compute the overlap of the test trajectories and produce the train-test stratification. The new computed files are automatically saved in the `data` folder with the names `dataset_metric_overlap.pkl`. 
Please note that running `overlap.py` may take a while.

The files are in a convenient format and can be directly used as input to well-known mobility models like Deep Move [https://github.com/vonfeng/DeepMove](https://github.com/vonfeng/DeepMove)
