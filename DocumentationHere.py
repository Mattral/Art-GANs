'''
Test Run
Only aims to read through commments
'''

import tensorflow.keras as keras
import numpy as np
# np.random.seed(1337)

#This imports the deque class from the collections module,
#which is used to create a double-ended queue data structure.
from collections import deque

import time

# Imports Python Imaging Library (PIL)
import PIL
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#This imports the garbage collector module,
#which is used to deallocate memory that is no longer being used
import gc



'''
________________________________________________End of Imports_________________________________________________
'''



def normImage(img):
    '''
   This defines a function called normImage
   that takes in an image and normalizes its pixel values to the range [-1, 1].
   '''
    img = (img / 127.5) - 1
    return img


def denormImage(img):
    '''
   This defines a function called denormImage that takes in a normalized image and
   denormalizes its pixel values back to the range [0, 255]
   '''
    img = (img + 1) * 127.5
    return img.astype(np.uint8)


def wassersteinLoss(y_true, y_pred):
     '''
     This defines a custom loss function called wassersteinLoss that takes in two inputs: y_true and y_pred.
     This function calculates the Wasserstein loss,
     which is a differentiable distance metric used to measure the distance between two probability distributions.
     '''
     return keras.backend.mean(y_true * y_pred)

'''
___________________________________________________________________________________________________________
'''


class nftGAN():
    '''
    noiseShape specifies the shape of the noise vector that is used as input to the generator,
    imageShape specifies the shape of the output images.
    '''
    def __init__(self, noiseShape, imageShape):

        # no. of classes for GAN
        self.nClasses = 3

        #asign arguments
        self.noiseShape = noiseShape
        self.imageShape = imageShape

        
        self.generator = self.generateGenerator()  #method that generates the generator neural network.
        self.criticer = self.generateCriticer()    #method that generates the critic (or discriminator) neural network.
        self.adversial = self.generateAdversial()  #method that combines the generator and critic networks into a single adversarial network.

        # set Dir
    self.imageSaveDir = "generatedImages"
    self.datasetDir = "Dir"

        

'''
___________________________________________________________________________________________________________
'''




# The following defines a method called generateGenerator in the nftGAN class that generates a generator neural network.

    def generateGenerator(self):
     
      '''
      This creates a sequential model called cnn that consists of several layers. The first layer is a dense layer with 1024 neurons, followed by a leaky ReLU activation function.
      The second layer is another dense layer with 16 * 16 * 32 neurons, followed by a leaky ReLU activation function.
      The third layer reshapes the output of the second layer into a 16 x 16 x 32 tensor.
       The fourth layer upsamples the tensor by a factor of 2 using a 2D upsampling layer.
       The fifth layer is a 2D convolutional layer with 256 filters, a filter size of 5, a "same" padding scheme,
      and a glorot_uniform kernel initializer. It is followed by a leaky ReLU activation function.
      The sixth layer upsamples the tensor by another factor of 2.
      The seventh layer is another 2D convolutional layer with 128 filters, a filter size of 5,
      a "same" padding scheme, and a glorot_uniform kernel initializer. It is followed by a leaky ReLU activation function.
      Finally, the eighth layer is a 2D convolutional layer with 3 filters, a filter size of 2,
      a "same" padding scheme, a tanh activation function, and a glorot_uniform kernel initializer.
      '''


        cnn = keras.Sequential([
            keras.layers.Dense(1024, input_dim=self.noiseShape),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(16 * 16 * 32),
            keras.layers.LeakyReLU(),
            keras.layers.Reshape((16, 16, 32)),
            # (32, 32, 256)
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(256, 5, padding="same", kernel_initializer="glorot_uniform"),
            keras.layers.LeakyReLU(),
            # (64, 64, 128)
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(128, 5, padding="same", kernel_initializer="glorot_uniform"),
            keras.layers.LeakyReLU(),
            # (64, 64, 3)
            keras.layers.Conv2D(3, 2, padding="same", activation="tanh", kernel_initializer="glorot_uniform")
        ])


        #This creates an input layer for the noise vector
        latent = keras.layers.Input(shape=(self.noiseShape,))
        
        #This creates an input layer for the image class, with a shape of (1,) and a data type of int32.
        image_class = keras.layers.Input(shape=(1,), dtype="int32")

        #This creates an embedding layer for the image class, with self.nClasses
        #as the number of classes and self.noiseShape as the size of the embedding vector.
        #The layer is then flattened into a 1D tensor.
        cls = keras.layers.Flatten()(keras.layers.Embedding(self.nClasses, self.noiseShape)(image_class))

        #This multiplies the noise vector with the embedding tensor element-wise.
        h = keras.layers.Multiply()([latent, cls])

        #This passes the element-wise product of the noise vector and the embedding tensor through the CNN defined earlier to generate a fake image.
        fake_image = cnn(h)


        #This creates a Keras model that takes in the noise vector and the image class as inputs and outputs the fake image.
        model = keras.Model(inputs=[latent, image_class], outputs=fake_image)

        #This compiles the Keras model with an Adam optimizer with a learning rate of 0.0002 and a beta1 value of 0.5, and a binary cross-entropy loss function.
        model.compile(optimizer=keras.optimizers.Adam(lr=.0002, beta_1=.5), loss="binary_crossentropy")

    return model



'''
___________________________________________________________________________________________________________
'''


    '''
    The generateCriticer() method defines the critic (or discriminator) model of the GAN.
    This model takes an image as input and produces two outputs:
    one that is a real/fake decision and one that is a classification decision.

     series of convolutional layers that extract features from the input image.
     The extracted features are then flattened and passed through two dense layers to produce the two output predictions.

     The Conv2D layers have a 3x3 kernel size with padding set to "same" and an activation function of "relu".
     The first layer has 32 filters, the second layer has 64 filters, the third layer has 128 filters,
     and the fourth layer has 256 filters. Each convolutional layer
     is followed by a LeakyReLU() activation function with a negative slope of 0.2 to prevent the gradient from vanishing.
     After each convolutional layer, there is a Dropout() layer with a dropout rate of 0.3 to prevent overfitting.
     '''  
    
    def generateCriticer(self):       
        cnn = keras.Sequential([
            keras.layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=self.imageShape),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(.3),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(.3),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(.3),
            keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(.3),
            keras.layers.Flatten()
        ])

        #The image input is a tensor with shape self.imageShape, which is the shape of the input images.
        #The features tensor is obtained by passing the image tensor through the convolutional layers.
        
        image = keras.layers.Input(shape=self.imageShape)
        features = cnn(image)


        # The fake output is produced by a dense layer with one output unit and a linear activation function.
        # This output represents the critic's decision on whether the input image is real or fake.


        fake = keras.layers.Dense(1, activation="linear", name="generation")(features)

        # The aux output is produced by a dense layer with self.nClasses output units and a softmax activation function.
        # This output represents the critic's classification decision,
        # i.e., which class the input image belongs to.
        
        aux = keras.layers.Dense(self.nClasses, activation="softmax", name="auxiliary")(features)

        # The model is compiled using the wassersteinLoss function (which implements the Wasserstein loss)
        # as the loss for the fake output, and "sparse_categorical_crossentropy" as the loss for the aux output.
        
        model = keras.Model(inputs=image, outputs=[fake, aux])
        # model.compile(optimizer=keras.optimizers.SGD(clipvalue=.01), loss=[wassersteinLoss, "sparse_categorical_crossentropy"])


        # The optimizer used is the Adam optimizer with a learning rate of 0.0002 and a beta_1 value of 0.5.
        model.compile(optimizer=keras.optimizers.Adam(lr=.0002, beta_1=.5), loss=[wassersteinLoss, "sparse_categorical_crossentropy"])
    return model




'''
___________________________________________________________________________________________________________
'''





    '''
    This function generates the adversarial model which consists of a combination of the generator and the critic.
    
    The fake images generated by the generator are passed through the critic to get the critic score and the auxiliary classifier score.
    The critic is not updated with these fake images as it is not trainable.
    '''

    
    def generateAdversial(self):

        #First, it defines the input layers for the latent vector and image class.
        latent = keras.layers.Input(shape=(self.noiseShape,))
        image_class = keras.layers.Input(shape=(1,), dtype="int32")

        # Then, it generates fake images using the generator by passing the latent vector and image class as input.
        fake = self.generator([latent, image_class])
        
        # The critic is set to be non-trainable as we are only interested in training the generator in this case.
        self.criticer.trainable = False

        # The fake images generated by the generator are passed through the critic to get the critic score
        # and the auxiliary classifier score.
        # The critic is not updated with these fake images as it is not trainable.
    
        fake, aux = self.criticer(fake)
        combined = keras.Model(inputs=[latent, image_class], outputs=[fake, aux])
        combined.compile(optimizer="RMSprop", loss=[wassersteinLoss, "sparse_categorical_crossentropy"])
    return combined



#___________________________________________________________________________________________________________



    '''
    This code implements the training loop of a GAN model.
    The loop iterates through epochs and batches of data.
    In each batch, it loads real images and generates fake images with the generator model.
    Then, it prepares the data for training by concatenating real and fake images,
    and generating labels for the discriminator and the adversarial models.
    
    After that, it trains the critic model on the real and fake images with their respective labels
    and calculates its loss. Next, it trains the adversarial model on the generated noise and labels,
    also calculating its loss
    
    The average loss of the discriminator and the adversarial models are stored
    in two deque objects for logging purposes.
    At the end of each epoch, the average losses are printed.
    '''

    
    def fit(self, epochs, batchSize):
        averageDiscriminatorLoss = deque([0], maxlen=250)
        averageGanLoss = deque([0], maxlen=250)
        
        print("Images loaded.")
        
        for epoch in range(epochs):
            print("Epoch:", epoch)
            
            startTime = time.time()
            
            # Loop over dataset
            for iBatch in range(0, 1110, batchSize):
                
                # load real images
                realImagesX, labels = self.getSamplesFromDataset(iBatch, iBatch + batchSize)
                if len(realImagesX) == 0: break
                
                # generate fake images with generator
                noise = self.generateNoise(len(realImagesX))
                fakeImagesX = self.generator.predict([noise, labels])
                
                # save generator samples
                if epoch % 2 == 0 and iBatch == 0 and epoch != 0:
                    stepNum = str(epoch).zfill(len(str(epochs)))
                    self.saveImageBatch(fakeImagesX, str(stepNum) + "_image.png")
                    
                # prepare data for training
                sampledLabels = np.random.randint(0, self.nClasses, len(realImagesX))
                yRealness = np.concatenate((- np.ones(len(realImagesX)), np.ones(len(realImagesX))), axis=0)
                yLabel = np.concatenate((labels, sampledLabels), axis=0)
                x = np.concatenate((realImagesX, fakeImagesX))
                
                # train criticer
                discriminatorMetrics = self.criticer.train_on_batch(x, [yRealness, yLabel])
                print("Discriminator: loss: %f" % (discriminatorMetrics[0]))
                averageDiscriminatorLoss.append(discriminatorMetrics[0])
                
                # train adversial model
                ganX = self.generateNoise(len(realImagesX) * 2)
                sampledLabels = np.random.randint(0, self.nClasses, len(realImagesX) * 2)
                ganY = - np.ones(len(realImagesX) * 2)
                ganMetrics = self.adversial.train_on_batch([ganX, sampledLabels], [ganY, sampledLabels])
                print("GAN loss: %f" % (ganMetrics[0]))
                averageGanLoss.append(ganMetrics[0])
                gc.collect()
                
            # finish epoch and log results
            diffTime = int(time.time() - startTime)
            print("Epoch %d completed. Time took: %s secs." % (epoch, diffTime))
            if (epoch + 1) % 500 == 0:
                print("-----------------------------------------------------------------")
                print("Average Disc loss: %f" % (np.mean(averageDiscriminatorLoss)))
                print("Average GAN loss: %f" % (np.mean(averageGanLoss)))
                print("-----------------------------------------------------------------")
    return {"Discriminator": averageDiscriminatorLoss, "Adversial": averageGanLoss}
    
    #The function returns a dictionary containing the average losses of both models.





#___________________________________________________________________________________________________________



'''
This function generates random noise for the generator network.
It takes in the batch size and returns an array of shape (batchSize, self.noiseShape),
where each element is a random number generated from a normal distribution
with mean 0 and standard deviation 1.
'''
    

def generateNoise(self, batchSize):
    return np.random.normal(0, 1, size=(batchSize,self.noiseShape))




#___________________________________________________________________________________________________________


    '''
    This method saves a batch of images generated by the generator to a file
    imageBatch is a numpy array of shape (batchSize, height, width, channels) representing the generated images.
    fileName is the name of the file to save the images in.
    The method uses matplotlib to create a 4x4 grid of 16 images randomly selected from imageBatch
    and saves the resulting plot to the file specified in fileName.
    The images are denormalized before being displayed and saved using the denormImage function.
    '''


    def saveImageBatch(self, imageBatch, fileName):
        plt.figure(figsize=(4,4))
        gs1 = gridspec.GridSpec(4, 4)
        gs1.update(wspace=0, hspace=0)
        rand_indices = np.random.choice(imageBatch.shape[0], 16, replace=True)
        for i in range(16):
            ax1 = plt.subplot(gs1[i])
            ax1.set_aspect("equal")
            rand_index = rand_indices[i]
            image = imageBatch[rand_index, :,:,:]
            fig = plt.imshow(denormImage(image))
            plt.axis("off")
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(self.imageSaveDir + "/" + fileName, bbox_inches="tight", pad_inches=0)
    plt.close()


    
    
#___________________________________________________________________________________________________________



# This function loads an image from the dataset directory and returns it as a normalized numpy array
# with dimensions specified by the imageShape attribute of the GAN class.
# The image is loaded using the PIL library and is opened as an RGB image.
# It is then resized to the dimensions specified by imageShape and converted to a numpy array.
# Finally, it is normalized using the normImage function.
# The fileName parameter specifies the name of the file to be loaded and should include the file extension.


    def loadImage(self, fileName):
        image = PIL.Image.open(self.datasetDir + "/images/" + fileName)
        image = image.resize(self.imageShape[:-1])
        image = image.convert("RGB")
        image = np.array(image)
        image = normImage(image)
    return image



#___________________________________________________________________________________________________________



    '''
    This function getSamplesFromDataset loads a batch of samples from the dataset.
    It takes two arguments: countStart and countEnd which define the range of samples to be loaded.
    It first reads the file names of the images in the given range from the images directory of the dataset
    and then loads the images by calling the loadImage function.
    Then, it reads the corresponding labels from the labels.txt file and converts them to one-hot encoded format.
        
    
    In this implementation, the labels are converted to integers where the original labels are 1, 4, and 5,
    and they are mapped to 0, 1, and 2, respectively.
    This is done by iterating over the labels and checking each label
    to determine the corresponding integer value.
    '''


    def getSamplesFromDataset(self, countStart, countEnd):
        images, labels = [], []
        fileNames = os.listdir(self.datasetDir + "/images")
        fileNames = [file for file in fileNames if len(file.split(".")) == 2 and file.split(".")[1] == "png"][countStart : countEnd]
        images = [self.loadImage(file) for file in fileNames]# if len(file.split(".")) == 2 and file.split(".")[1] == "jpg"]        
        with open(self.datasetDir + "/labels.txt") as file: labels = file.readlines()[countStart : countEnd]
        labels_ = []
        for label_ in labels:
            label = int(label_)
            if label == 1:
                labels_.append(0)
            elif label == 4:
                labels_.append(1)
            elif label == 5:
                labels_.append(2)
            else:
                assert 1==2, "impossible case: " + str(label) + str(type(label))
        labels = labels_
    return np.array(images), np.array(labels)

    #Finally, it returns the loaded images and labels as numpy arrays.






#___________________________________________________________________________________________________________


    '''
    This is a function that can be used to plot loss values during training.
    It takes a dictionary of losses as input, where the keys are the names of the loss functions
    and the values are lists of loss values for each epoch.
    
    This would generate two figures, one for the "generator" loss and one for the "discriminator" loss,
    each with four data points plotted.
    '''


    def plotLosses(losses:dict):
        for key, value in losses.items():
            plt.figure()
            plt.plot(value, label=key)
            plt.ylabel("loss")
            plt.legend()
        plt.show()



'''___________________________________________________________________________________________________________
'''


NOISE_SHAPE = 200#100
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SHAPE = (64,64,3)

if __name__ == "__main__":
    gan = nftGAN(NOISE_SHAPE, IMAGE_SHAPE)
    losses = gan.fit(EPOCHS, BATCH_SIZE)
print("Training finished.")



'''
# We usually use rule of thumb thing in adjusting Batch Size. Determine a suitable batch according to your dataset's total images.
# you can save weighs and load then again using keras
'''    
    
