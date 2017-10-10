# Deep Learning Project

## Network Architecture

#### FCN Overview 

FCN is short for Fully Convolutional Network. This type of network is especially suited for semantic segmentation tasks like the one in this project. Here we are given images taken by a drone which can contain 3 classes of objects. These 3 classes are 'background', 'other people' and 'hero'.

In the image below we can see: (left) an image from the sample evaluation data collected from the drone flying in a simulator containing the 'hero'; (middle) the ground truth showing the 3 classes of objects; (right) the 3 classes predicted by the trained network.

![image with hero](/images/img_hero.png)

A FCN consists of 2 main parts: an **encoder** and a **decoder**. Those two parts are connected by a **1x1 convolutional layer**.

#### Encoder

The *encoder* applies a number of convolutional layers to an input image, with the result that the image is down-sampled. In this implementation the encoder consists of 2 convolutional layers. 

```python
def encoder_block(input_layer, filters, strides):
    
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

#### 1x1 Convolutional layer 

A *1x1 convolutional layer* avoids losing spatial information unlike a fully connected layer.

```python
 # Add 1x1 Convolution layer using conv2d_batchnorm().
 conv_1_1_layer = conv2d_batchnorm(encoder_layer_2, 128, kernel_size=1, strides=1)
```

#### Decoder with skip connections

The *decoder* consists of the exact same number of layers, but it up-samples the down-sampled image such that the output again has the same size as the input. The operation which these layers perform can be referred to as transposed convolutions or de-convolutions. In this implementation the decoder also makes use of so-called *skip connections*. These connections help the network in keeping an overview of 'the big picture' which would be lost otherwise.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([upsampled_layer, large_ip_layer])
    
    # Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concat_layer, filters)
    
    return output_layer
```
#### FCN implementation

Combining all the above pieces into an FCN, gives the following code:

```python
def fcn_model(inputs, num_classes):
    
    # Add Encoder Blocks. 
    encoder_layer_1 = encoder_block(inputs, 32, 2)
    encoder_layer_2 = encoder_block(encoder_layer_1, 64, 2)

    # Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_1_1_layer = conv2d_batchnorm(encoder_layer_2, 128, kernel_size=1, strides=1)
    
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_layer_1 = decoder_block(conv_1_1_layer, encoder_layer_1, 64)
    decoder_layer_2 = decoder_block(decoder_layer_1, inputs, 32)
    
    x = decoder_layer_2
    
    # The function returns the output layer of your model. "x" is the final layer obtained 		from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```



## Parameter Tuning

#### Overview

Manually tuning the various hyper parameters of the FCN is a pain (a solution to this is suggested later on). 

I limited the manual tuning to 3 parameters: **learning rate**, **batch size** and **number of epochs**. The remaining 3 parameters were left as they were found in the Segmentation Lab notebook. This was a measure to make the tuning easier, simply by limiting the possibilities that needed to be tried out.

#### Learning rate

I tried out various learning rates from 0.1 to 0.001. If the rate is too high the loss will be too noisy. On the other hand, if the rate is too low, learning will not progress very quickly.

The optimal rate I found was `learning_rate = 0.005`. 

#### Batch size
Here I tried out batch sizes using the power of 2, from 2 to 128.

I settled at a batch size of `batch_size = 64`

#### Number of epochs

At first I tried to train the network at incremental steps of 10 epochs. By doing this repeatedly, I came to the conclusion that 50 epochs would be a good total number. Hence, I restarted the training and let it run for 50 epochs and again reached the same result.

`num_epochs = 50`

#### Other parameters

I left the following parameters as they were on the original *segmentation lab* notebook. At some point I did try to change the number of workers, but it seemed to make no difference in training speed.

`steps_per_epoch = 500`

`validation_steps = 50`

`workers = 2`

## Network Results

The training and validation data was supplied by Udacity. I did not collect any additional data.

Training was done using the Adam optimizer.

By constructing the FCN and training it using the parameter settings mentioned above, I was able to make the network achieve a final score of 40%.

The evolution of the loss rates for the training data and the validation data can be seen in the graph below:

![loss graph](/images/training.png)

The weights from this training run have been saved in the `model_weights_VI_01` file. It is in *h5* format.

## Limitations

In the training set there is a class called 'hero', this only applies to human characters with a very specific look. If the 'hero' were to change looks then the network would not recognize her anymore.

In order to recognize other objects like for example cats, dogs, cars, trucks, etc. new training data needs to be collected which contains these classes.

In addition, the training images were collected by the drone flying at a specific altitude.  If the drone were to fly at a different altitude, the network would probably have some difficulty recognizing the 'hero' and the 'other people'.

## Potential Improvements

#### Automated tuning of the hyper parameters

As was mentioned previously, the manual tuning process of the hyper parameters is quite a pain. A possible solution is to use a scikit-learn http://scikit-learn.org/stable/ 'grid search' or 'randomized search' to find good parameters in an automated fashion.

#### Generating additional data

For this project I only used the data which was made available by Udacity. I did not generate any additional data myself. Increasing the amount of training data should improve the result though.

#### Data augmentation

Data augmentation. mirroring images, zooming in, adding noise

#### Transfer learning for the encoder

Transfer learning using for example a pre-trained VGG network for the encoder part, thus leaving only the 1x1 convolution and the decoder to be trained. This can drastically speed up the learning process and hence significantly lower the time needed for this.