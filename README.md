### Setup

Run the script `setup.sh`, as follows:
```
$ ./setup.sh
```
This script basically does the following:
  1. Creates a virtualenv.
  2. Installs Torch and Torchvision, choosing the appropriate version depending on the availability of a GPU on your system and the version of CUDA installed if you have a GPU. (This is done using an [external script](https://gist.github.com/codeandfire/5b98dac9a5453e765f1c212625b118b2).)
  3. Installs other packages (specified in `requirements.txt`), and also the [official LIME package](https://github.com/marcotcr/lime).
  4. Collects a dataset of images from ImageNet, consisting of around 150 images for each of the classes 'goldfish', 'house finch', 'bulbul', 'balloon', 'bathing cap', 'analog clock', 'digital clock', 'muzzle' and 'geyser'. (Again, this is done using an [external script](https://github.com/codeandfire/imagenet-scraper).)
  The entire dataset is stored within a directory named `dataset/` in the current directory.
  5. Obtains a file containing the list of the 1000 ImageNet classes along with their WordNet IDs. (This is the `wnids_1000.txt` file of the `imagenet-scraper` repository, if you followed the previous link.)
  6. Downloads a pretrained version of AlexNet from PyTorch.

### Usage

Run the script `run_lime.py`, passing to it the path to an image.
For example:
```
$ python3 run_lime.py dataset/n01532829/51694033_d5a7fd6f12.jpg'
```

The output is
```
Segmentation plot saved to segmentation_51694033_d5a7fd6f12.png
Generating perturbations ...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
[Parallel(n_jobs=-1)]: Done  18 tasks      | elapsed:    0.2s
[Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:    0.6s finished
Sample perturbation saved to sample_perturbation_51694033_d5a7fd6f12.png
Loading AlexNet ...
Passing image to AlexNet ...
class        wordnet id      probability
-----------  ------------  -------------
house_finch  n01532829          0.997684
Passing perturbations to AlexNet ...
100%|█████████████████████████████████████████████| 4/4 [00:00<00:00,  6.83it/s]
Generating explanations ...
Explanation plot saved to explanation_house_finch_51694033_d5a7fd6f12.png
```
As you can see, this script runs the LIME algorithm on the given image, and saves plots of the segmentation of the image, a sample from the generated perturbations, and the final explanation plot.

Additional configuration options are displayed if you pass the `--help` flag.
```
$ python3 run_lime.py --help
usage: run_lime.py [-h] [--classes CLASSES] [--perturbations PERTURBATIONS]
                   [--batch-size BATCH_SIZE] [--segments SEGMENTS]
                   [--hide-colour {black,white,mean}]
                   image_path

A re-implementation of LIME for explaining images.

positional arguments:
  image_path            image on which to carry out inference and prediction

options:
  -h, --help            show this help message and exit
  --classes CLASSES     provide explanations for the top n classes predicted
                        for the given image (default: 1)
  --perturbations PERTURBATIONS
                        number of perturbations to generate for the given
                        image (default: 1000)
  --batch-size BATCH_SIZE
                        batch size to be used while passing perturbations to
                        the AlexNet model (default: 256)
  --segments SEGMENTS   number of segments to be highlighted in the
                        explanation (default: 5)
  --hide-colour {black,white,mean}
                        colour used for hiding segments while generating
                        perturbations (default: mean)
```

### Idea

The script `run_lime.py` is a simple re-implementation of the LIME algorithm for images.
I have tried to make this script as readable as possible, so you can go ahead and read the script in order to understand the various steps involved in the LIME algorithm.

The idea behind writing this script is that I wanted to get a clear understanding of the entire algorithm, and I found this rather difficult to do with the official LIME package.
To the best of my knowledge, this script matches the original algorithm exactly, except for the following points:
  - Before training the explanation model, the LIME package includes a step of feature selection, which we have skipped in order to keep the code brief.
  Nevertheless, by default, the feature selection step in the LIME package is not triggered because the default number of features to select is set to a very high value, so all features are selected unless you change this number of features yourself.
  - The LIME package provides several options to control the underlying model.
  For example, it allows you to control the segmentation algorithm used, the metric used for computing distances between the perturbations and the original image, the kernel used for converting these distance values into proximity values, and the explanation model used.
  In our case, we keep these options fixed to their defaults as defined in the package: the segmentation algorithm used is Quickshift, the distance metric is cosine distance, the kernel is an exponential kernel, and the explanation model is a ridge regression model.
  All of these models (except the cosine distance metric) also have hyperparameters: again, the LIME package allows you to control these hyperparameters, but we just stick with the defaults that they have defined. 
  - Just as the LIME package provides options to control the underlying model, they also provide a similar variety of options to control the resulting explanation image.
  For example, it allows you to control whether only positively-contributing segments or negatively-contributing segments are shown or both, whether to hide insignificant segments or not, the colour used for hiding such segments, and the minimum weight value for a segment to be considered as significant.
  In our case, we stick with a configuration wherein both positively- and negatively-contributing segments are shown, with the former highlighted in green and the latter in red, and no segments are hidden.
  The minimum weight value is fixed to the default of zero as defined in the package.
  - One option that the LIME package provides and that we have retained is the option of choosing the colour used to hide segments while generating perturbations. The reason for this will be discussed very soon.
  - I have attempted to parallelize the generation of perturbations using Joblib.
  Recall that LIME works by generating a large number (in the range of thousands) of perturbations of a single given image, hence we may benefit if these perturbations are generated in parallel over multiple CPU cores.

The output from this script should typically exactly match the output from the LIME package.
You can verify this as follows.
The script `verify_lime.py` is a script that uses the official LIME package and follows their official tutorial notebook for image explanations (using PyTorch), [found here](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb).
You can check that the script is exactly the same as the notebook, with only a few minor changes.
Then, you can run `verify_lime.py` with the same image and the same configuration options as `run_lime.py`, and check that the resulting explanation image is very, very similar.
