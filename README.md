# Forced Autoencoder with Siamese Network for Improved Novelty Estimation
Forced Autoencoder utilizes training the encoder mapping to predetermined random latent space based on label.
Label can either be determined from the dataset, or from KNN Classification of Images in dataset.
The Google Colab can be found [here](https://colab.research.google.com/drive/1deKzMnPDUC2omfoXGCSdUPVwVJ-93Byc?usp=sharing)

## To install dependencies
There are some dependencies that is required to run the code, such as torch.
```
pip install -r requirements.txt
```
<br/>

However, if you still face issue importing torch, you might need to install it manually using this command
```
1. pip uninstall torch
2. pip uninstall torchvision
3. conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
<br />

## Download Pretrained Models
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1MDSUt3_YH17-a_XwR5D0NO4MIUCKr-EV?usp=sharing)
<br />

Once the model is downloaded, add the pretrained models to "Data", such that the model in "Data"
```
Data/autoencoderVariant4KNN.pkl
Data/autoencoderVariant5KNN.pkl
Data/decoderTrainedKNN.pth
Data/encoderTrainedKNN.pth
Data/normalDecoder.pth
Data/normalEncoder.pth
Data/retrain_autoencoderVariant4KNN.pkl
Data/retrain_autoencoderVariant5KNN.pkl
Data/retrain_decoderTrainedKNN.pth
Data/retrain_encoderTrainedKNN.pth
Data/retrain_normalDecoder.pth
Data/retrain_normalEncoder.pth
Data/retrain_variant2Encoder.pth
Data/retrain_variant3Encoder.pth
Data/retrain_variant4Decoder.pth
Data/retrain_variant4Encoder.pth
Data/retrain_variant5Encoder.pth
Data/siameseVariant1.pth
Data/siameseVariant2.pth
Data/siameseVariant2KNN.pkl
Data/variant2Decoder.pth
Data/variant2Encoder.pth
Data/variant3Decoder.pth
Data/variant3Encoder.pth
Data/variant4Decoder.pth
Data/variant4Encoder.pth
Data/variant5Decoder.pth
Data/variant5Encoder.pth
```

<br />


## Python Version
The Python version required for the code to run is Python 3.7.10<br />
To install this Python version, you will first need Anaconda to make swithing between different environment easier.
You can get Anaconda [here](https://www.anaconda.com/products/distribution)
<br /><br />
To check for current Python version, type this:
```
python --version
```

To install different version of Python in Anaconda:
In the terminal, 
```
conda create --name py3 python=3.7.10
```

To activate the environment:
```
conda activate py3
```
Note: if you already have Python 3.7.10 installed, and for the subsequent running of the program, you can just run this command.
<br />

## To run the code
```
python main.py
```

