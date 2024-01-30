# Art-GANs
Art work generation with generative adversarial networks. Make sure to read external documentations about GANs before testing.
Will Update further progress.

Magic will not happen just by runnning this code, you need to adjust according to data and shape of image. Read comments in DocumentationHere.py before editing or modifying anything.


# Art-GANs

Art-GANs is a project for generating art using Generative Adversarial Networks (GANs). It leverages the power of deep learning to create stunning and imaginative pieces of art. Please take the time to read through the documentation before running or modifying the code.


## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Code](#running-the-code)
- [Customization](#customization)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About

Art-GANs is an art generation project that uses Generative Adversarial Networks (GANs) to produce unique and creative artworks. GANs are a class of deep learning models that are capable of generating data that is similar to a given dataset. In this project, we leverage the capabilities of GANs to create art with a touch of magic.

## Getting Started

To get started with Art-GANs, follow the steps below.

### Prerequisites

Before running the code, make sure you have the following prerequisites:

- Python 3.x
- TensorFlow and Keras
- Numpy
- PIL (Python Imaging Library)
- Matplotlib

You can install the necessary Python libraries using pip:

```bash
pip install tensorflow numpy pillow matplotlib
```

## Installation

1. Clone the Art-GANs repository to your local machine:

   ```bash
   git clone (this repo)
   ```

2. Change your working directory to the project folder:
   ```
   cd art-gans
   ```
   
3. You are now ready to use the Art-GANs project.


## Usage

### Running the Code

1. **Navigate to the project directory in your terminal.**

2. **Make sure to read the comments in the `DocumentationHere.py` file.** These comments provide important information about the code structure and adjustments you might need to make according to your dataset and desired image shape.

3. **Execute the following command to start the art generation process (you need dataset):**

   ```bash
   python (FileName).py
   ```

## Customization

The Art-GANs project can be customized to create art tailored to your specific requirements. Here are a few areas you can customize:

- **Noise Shape**: Adjust the `NOISE_SHAPE` variable in the code to change the shape of the noise vector used as input to the generator.

- **Image Shape**: Modify the `IMAGE_SHAPE` variable to set the desired shape of the output images.

- **Batch Size and Epochs**: You can customize the batch size and the number of training epochs in the code to suit your dataset and time constraints.

## Testing

The project includes unit tests to ensure that the code is functioning correctly. You can run the tests using the following command:

```bash
python test.py
```

## Contributing
- I welcome contributions to the Art-GANs project. If you have ideas for improvements or would like to report issues, please feel free to submit pull requests or bug reports. Make sure to follow coding standards and conventions when contributing.

