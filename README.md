# mp3-codec

### Simplified MPEG-1-Layer-III Coding/Decoding

This project is a simplified implementation of the MPEG-1-Layer-III standard for coding and decoding audio data. It was created as part of the Multimedia Systems Class of 2023. The process involves the following stages:

1. Dividing the signal into frequency subbands
2. Applying the Discrete Cosine Transform (DCT) algorithm
3. Applying the psychoacoustic model to determine which frequency components can be safely removed
4. Quantization and dequantization of the DCT coefficients
5. Run Length Encoding and Decoding to reduce the data size
6. Huffman Encoding and Decoding for further data compression

This process is shown in the diagram below:

![Screenshot_14](https://user-images.githubusercontent.com/47897459/213874673-391f1d0b-4f91-4a6e-ba82-d9b2d5820945.png)


### Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Before you can run this project, you will need to install the following dependencies:

- Python 3.8
- numpy
- scipy
- matplotlib

### Installing
Clone the repository to your local machine:

git clone https://github.com/dmylon/mp3-codec.git

Then, install the required dependencies by running the following command(assuming python is already installed on your local machine):

```pip install -r requirements.txt```



### Usage
To run the code, open a terminal, navigate to the local folder the project is saved and run the following command:

- python main.py

### Results
After running this command, you should be able to view some basic diagrams containg the subband analysis in both the f(Hz) and the z frequency(barks). In addition, the  "output.wav" audio file is a representation of the audio data after it has been encoded and decoded from the original version in the "myfile.wav" file.
