# keras_link
良さげなKerasのリンクをまとめます．良さそうなまとめなどがありましたらPRして下さい．
## Official starter resources

- [keras.io](http://keras.io/) - Keras documentation
- [Getting started with the Sequential model](http://keras.io/getting-started/sequential-model-guide/)
- [Getting started with the functional API](http://keras.io/getting-started/functional-api-guide/)
- [Keras FAQ](http://keras.io/getting-started/faq/)

## Tutorials

- [Quick start: the Iris dataset in Keras and scikit-learn](https://github.com/fastforwardlabs/keras-hello-world/blob/master/kerashelloworld.ipynb)
- [Using pre-trained word embeddings in a Keras model](http://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
- [Building powerful image classification models using very little data](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [Building Autoencoders in Keras](http://blog.keras.io/building-autoencoders-in-keras.html)
- [A complete guide to using Keras as part of a TensorFlow workflow](http://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html)
- Introduction to Keras, from University of Waterloo: [video](https://www.youtube.com/watch?v=Tp3SaRbql4k) - [slides](https://uwaterloo.ca/data-science/sites/ca.data-science/files/uploads/files/keras_tutorial.pdf)
- Introduction to Deep Learning with Keras, from CERN: [video](http://cds.cern.ch/record/2157570?ln=en) - [slides](https://indico.cern.ch/event/506145/contributions/2132944/attachments/1258124/1858154/NNinKeras_MPaganini.pdf)
- [Installing Keras for deep learning](http://www.pyimagesearch.com/2016/07/18/installing-keras-for-deep-learning/)
- [Develop Your First Neural Network in Python With Keras Step-By-Step](http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
- [Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras](http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)
- [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
- [Keras video tutorials from Dan Van Boxel](https://www.youtube.com/playlist?list=PLFxrZqbLojdKuK7Lm6uamegEFGW2wki6P)
- [Keras Deep Learning Tutorial for Kaggle 2nd Annual Data Science Bowl](https://github.com/jocicmarko/kaggle-dsb2-keras/)
- [Collection of tutorials setting up DNNs with Keras](http://ml4a.github.io/guides/)
- [Fast.AI - Practical Deep Learning For Coders, Part 1](http://course.fast.ai/) (great information on deep learning in general, heavily uses Keras for the labs)

## Code examples

### Working with text

- [Reuters topic classification](https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py)
- [LSTM on the IMDB dataset (text sentiment classification)](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py)
- [Bidirectional LSTM on the IMDB dataset](https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py)
- [1D CNN on the IMDB dataset](https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py)
- [1D CNN-LSTM on the IMDB dataset](https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py)
- [LSTM-based network on the bAbI dataset](https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py)
- [Memory network on the bAbI dataset (reading comprehension question answering)](https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py)
- [Sequence to sequence learning for performing additions of strings of digits](https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py)
- [LSTM text generation](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)
- [Using pre-trained word embeddings](https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py)
- [Monolingual and Multilingual Image Captioning](https://github.com/elliottd/GroundedTranslation)
- [FastText on the IMDB dataset](https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py)
- [Structurally constrained recurrent nets text generation](https://github.com/nzw0301/keras-examples/blob/master/SCRNLM.ipynb)
- [Character-level convolutional neural nets for text classification](https://github.com/johnb30/py_crepe)
- [LSTM to predict gender of a name](https://github.com/divamgupta/lstm-gender-predictor)
- [Language/dialect identification with multiple character-level CNNs](https://github.com/boknilev/dsl-char-cnn)

### Working with images

- [Simple CNN on MNIST](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)
- [Simple CNN on CIFAR10 with data augmentation](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py)
- [Inception v3](https://github.com/fchollet/keras/blob/master/examples/inception_v3.py)
- [VGG 16 (with pre-trained weights)](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
- [VGG 19 (with pre-trained weights)](https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d)
- ResNet 50 (with pre-trained weights): [1](https://github.com/fchollet/keras/pull/3266/files) - [2](https://github.com/raghakot/keras-resnet)
- [FractalNet](https://github.com/snf/keras-fractalnet)
- [AlexNet, VGG 16, VGG 19, and class heatmap visualization](https://github.com/heuritech/convnets-keras)
- [Visual-Semantic Embedding](https://github.com/awentzonline/keras-visual-semantic-embedding)
- Variational Autoencoder: [with deconvolutions](https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder_deconv.py) - [with upsampling](https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py)
- [Visual question answering](https://github.com/avisingh599/visual-qa)
- [Deep Networks with Stochastic Depth](https://github.com/dblN/stochastic_depth_keras)
- [Smile detection with a CNN](https://github.com/kylemcdonald/SmileCNN)
- [VGG-CAM](https://github.com/tdeboissiere/VGG16CAM-keras)
- [t-SNE of image CNN fc7 activations](https://github.com/ml4a/ml4a-guides/blob/master/notebooks/tsne-images.ipynb)
- [VGG16 Deconvolution network](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DeconvNet)
- Wide Residual Networks (with pre-trained weights): [1](https://github.com/asmith26/wide_resnets_keras) - [2](https://github.com/titu1994/Wide-Residual-Networks)
- Ultrasound nerve segmentation: [1](https://github.com/jocicmarko/ultrasound-nerve-segmentation) - [2](https://github.com/raghakot/ultrasound-nerve-segmentation)
- [DeepMask object segmentation](https://github.com/abbypa/NNProject_DeepMask)
- Densely Connected Convolutional Networks: [1](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet) - [2](https://github.com/titu1994/DenseNet)
- [Snapshot Ensembles: Train 1, Get M for Free](https://github.com/titu1994/Snapshot-Ensembles)
- [Single Shot MultiBox Detector](https://github.com/rykov8/ssd_keras)

### Creative visual applications

- [Real-time style transfer](https://github.com/awentzonline/keras-rtst)
- Style transfer: [1](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py) - [2](https://github.com/titu1994/Neural-Style-Transfer)
- [Image analogies](https://github.com/awentzonline/image-analogies): Generate image analogies using neural matching and blending.
- [Visualizing the filters learned by a CNN](https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep dreams](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py)
- GAN / DCGAN: [1](https://github.com/phreeza/keras-GAN) - [2](https://github.com/jacobgil/keras-dcgan) - [3](https://github.com/osh/KerasGAN) - [4](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/GAN)
- [InfoGAN](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/InfoGAN)
- [pix2pix](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)
- [DFI](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DFI): Deep Feature Interpolation
- [Colorful Image colorization](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/Colorful): B&W to color

## その他
- [Machine Learning in a Year](https://medium.com/learning-new-stuff/machine-learning-in-a-year-cdb0b0ebd29c)  
機械学習速習コース紹介．ML系の知識がない場合は一通り目を通しておくのが良いです．

- [Classifying White Blood Cells With Deep Learning (Code and data included!)](https://blog.athelas.com/classifying-white-blood-cells-with-convolutional-neural-networks-2ca6da239331)

### Chainer
-  [Chainerビギナーズハンズオン資料](https://github.com/mitmul/chainer-handson)  
Trainerを使わない書き方は非常に参考に．学習済みモデルでの評価方法が参考になる

## 参考文献
- [keras-resouces](https://github.com/fchollet/keras-resources/blob/master/README.md)
