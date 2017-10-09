# Deep Learning Project

## Network Architecture

#### FCN Overview 

FCN is short for Fully Convolutional Network. This type of network is especially suited for semantic segmentation tasks like the one in this project. Here we are given images taken by a drone which can contain 3 classes of objects. These 3 classes are 'background', 'other people' and 'hero'.

![image with hero](/images/img_hero.png)

A FCN consists of 2 main parts: 'an encoder' and 'a decoder'. Those two parts are connected by a 1x1 convolutional layer.

The *encoder* applies a number of convolutional layers to an input image, with the result that the image is downsampled. In this implementation the encoder consists of 2 convolutional layers. 

The 1x1 

The *decoder* consists of the exact same number of layers, but it upsamples the downsampled image such that the output again has the same size as the input. The operation which these layers perform can be referred to as transposed convolutions or de-convolutions. In this implementation the decoder also makes use of so-called *skip connections*. These connections help the network in keeping an overview of 'the big picture' which would be lost otherwise.



#### Encoder

```python
def encoder_block(input_layer, filters, strides):
    
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

#### 1x1 Convolution 

A 1x1 convolutional layer avoids losing spatial information like fully connected layers would.

```python
 # Add 1x1 Convolution layer using conv2d_batchnorm().
 conv_1_1_layer = conv2d_batchnorm(encoder_layer_2, 128, kernel_size=1, strides=1)
```

#### Decoder w skip connections

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
Here I tried out batch sizes from 2 to 128.

`batch_size = 64`

#### Number of epochs
`num_epochs = 50`

#### Other parameters
`steps_per_epoch = 500`

`validation_steps = 50`

`workers = 2`

## Network Results and Limitations

drone flying at specific height

only applies to human characters with a specific hero



## Potential Improvements

#### Automated tuning of the hyper parameters

As was mentioned previously, the manual tuning process of the hyper parameters is quite a pain. A possible solution is to use a scikit-learn http://scikit-learn.org/stable/ grid search or randomized search to find good parameters in an automated fashion.

#### Generating additional data

For this project I only used the data which was made available by Udacity. I did not generate any additional data myself. Theoretically speaking, increasing the amount of training data should improve the result though.

Judging from the conversations I followed on the Slack channel of this project I made the following conclusions about the additional data:

* Just blindly generating extra data, would not help.
* Adding data with 'other people' and/or 'hero' visible and thus reducing the relative amount of data only containing the background, would help.

#### Data augmentation

Data augmentation... mirroring images, zooming in, adding noise

#### Transfer learning for the encoder

Transfer learning using for example a pre-trained VGG network for the encoder part, thus leaving only the 1x1 convolution and the decoder to be trained. This can drastically speed up the learning process and hence significantly lower the time needed for this.