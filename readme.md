# Readme

Run: main.py

Before running, please:
1. Download and move to folder __data__ our depth videos:
https://edysk.zut.edu.pl/index.php/s/yLBiJPH6WGAG2t6
2. Install dependencies - __pip install -r requirements.txt__. Convenient is using PyCharm IDE and creating a virtual environment.
3. If you would like to use __nn__ predictor, please compile and install __spconv 1.2.1__ library (readme and library included in folder spconv, works only on Linux).


## Configuration (in file config.ini)

1. main (__the most important part of config__)
	* Predictor - set __graphclust__ for graph optimization or __nn__ for convolutional neural network
	* Camera - left/right
	* VideoType - train/test
2. visualization
	* Open3dEnabled - __True__ / __False__ - enable / disable 3D visualization of K-Means and graph optimization result
	* PlotBackgroundExtractor - __True__ / __False__ - enable / disable drawing positive and negative images for background extractor
	* Show4Images - __True__ / __False__ - enable / disable visualization of processing steps 
3. graph
	* Intermediates - number of intermediates points in graph fitting procedure
	* Epochs - number of optimization algorithm epochs
	* Kmeans - number of K-Means centers
4. nn
    * TorchNumThreads - number of threads for neural network
    * SklearnNumThreads - number of threads for graph optimization