"""A re-implementation of LIME for explaining images."""

import argparse
from collections import namedtuple
from pathlib import Path
import random

from joblib import delayed, Parallel
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.segmentation import mark_boundaries, quickshift
from sklearn.linear_model import Ridge
from tabulate import tabulate
import torch
from torch.nn.functional import softmax
from torchvision import models, transforms
from tqdm import tqdm

# command-line arguments.
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'image_path', type=str,
    help='image on which to carry out inference and prediction')
parser.add_argument(
    '--classes', type=int, default=1,
    help='provide explanations for the top n classes predicted for the given image')
parser.add_argument(
    '--perturbations', type=int, default=1000,
    help='number of perturbations to generate for the given image')
parser.add_argument(
    '--batch-size', type=int, default=256,
    help='batch size to be used while passing perturbations to the AlexNet model')
parser.add_argument(
    '--segments', type=int, default=5,
    help='number of segments to be highlighted in the explanation')
parser.add_argument(
    '--hide-colour', type=str, choices=['black', 'white', 'mean'], default='mean',
    help='colour used for hiding segments while generating perturbations')
args = parser.parse_args()

# load the 1000 ImageNet classes with their name and WordNet ID.
image_class = namedtuple('ImageClass', ['name', 'wordnet_id', 'index'])
classes = []
with open('wnids_1000.txt', 'r') as f:
    for index, line in enumerate(f.readlines()):
        name, wordnet_id = line.strip().split(',')
        # we don't want spaces in any of the class names.
        name = name.replace(' ', '_')
        classes.append(image_class(name, wordnet_id, index))

image_path = Path(args.image_path)

# open the image, transform it to the standard size of 224 x 224 pixels, and extract its
# pixel values into a Numpy array.
image = Image.open(image_path)
image = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
image = np.array(image)

# find the segments (superpixels) of the image using the QuickShift algorithm.
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2, random_seed=123)

# create an image marking the segments found.
segmented_image = mark_boundaries(image, segments)
segmented_image = Image.fromarray((segmented_image * 255).astype('uint8'), mode='RGB')
file = 'segmentation_{}.png'.format(image_path.stem)
segmented_image.save(file)
print('Segmentation plot saved to', file)

# the total number of segments.
num_segments = len(np.unique(segments))

# the colour used for hiding a segment while generating perturbations.
# this is a mapping from segment numbers to their respective colours.
hide_colour = {}
for seg in range(num_segments):
    if args.hide_colour == 'black':
        # use black for all segments.
        hide_colour[seg] = (0, 0, 0)
    elif args.hide_colour == 'white':
        # use white for all segments.
        hide_colour[seg] = (255, 255, 255)
    else:
        # find all the pixels that fall within the area of this segment, and set
        # the colour to the mean of these pixel values across all three colour channels.
        idxs = (segments == seg).nonzero()
        hide_colour[seg] = (
            np.mean(image[idxs, 0]),
            np.mean(image[idxs, 1]),
            np.mean(image[idxs, 2]),
        )

perturbation = namedtuple('Perturbation', ['image', 'segments_on_off'])


def perturb_image(seed=None):
    """Compute a perturbation of the given image, by turning "off" certain segments and
    retaining others as "on".
    """
    if seed is not None:
        np.random.seed(seed)

    perturbed_image = np.copy(image)

    # randomly turn certain segments "off" and leave others "on".
    # an "off" segment corresponds to an assignment of 0 to the segment, while an "on"
    # segment corresponds to an assignment of 1.
    segments_on_off = np.random.choice([0, 1], (num_segments,), replace=True)

    # the segments that have been turned "off".
    segments_off = (segments_on_off == 0).nonzero()[0]

    for seg in segments_off:
        idxs = (segments == seg).nonzero()
        perturbed_image[idxs] = hide_colour[seg]

    return perturbation(image=perturbed_image, segments_on_off=segments_on_off)


# carry out the given number of perturbations in parallel over all available cores.
print('Generating perturbations ...')
image_perturbations = Parallel(n_jobs=-1, verbose=1)(
    [delayed(perturb_image)(seed=10*i) for i in range(args.perturbations)])

# add the original image too, as a perturbation, with all segments turned "on".
image_perturbations.append(
    perturbation(image=image, segments_on_off=np.ones((num_segments,))))

sample_perturbation = random.choice(image_perturbations)
sample_perturbation = mark_boundaries(sample_perturbation.image, segments)
sample_perturbation = Image.fromarray((sample_perturbation * 255).astype('uint8'), mode='RGB')
file = 'sample_perturbation_{}.png'.format(image_path.stem)
sample_perturbation.save(file)
print('Sample perturbation saved to', file)


print('Loading AlexNet ...')
model = models.alexnet(pretrained=True)
# put the model in inference mode.
model.eval()

# transfer the model to a GPU if one is available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# transform to normalize pixel values.
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print('Passing image to AlexNet ...')
x = normalize(image)
x = x.unsqueeze(0).to(device)
with torch.no_grad():
    # the predicted probability values.
    y = softmax(model(x), dim=1)

# extract the probabilities of the top n classes, where n = args.classes, and print them.
probs, top_classes = y.topk(args.classes)
probs, top_classes = probs.squeeze(0), top_classes.squeeze(0)
top_classes = [classes[class_idx] for class_idx in top_classes]
print(tabulate(
    [(class_.name, class_.wordnet_id, prob) for class_, prob in zip(top_classes, probs)],
    headers=['class', 'wordnet id', 'probability']))

print('Passing perturbations to AlexNet ...')

# we maintain a separate output list for each class in the list top_classes.
outputs = {class_: [] for class_ in top_classes}

num_batches = (len(image_perturbations) // args.batch_size
               + min(len(image_perturbations) % args.batch_size, 1))

for i in tqdm(range(num_batches), total=num_batches):

    # extract a batch of perturbations.
    x = image_perturbations[(args.batch_size * i): (args.batch_size * (i+1))]
    x = [perturbation.image for perturbation in x]

    # pass the batch to the model and get the predicted probability values.
    x = torch.stack([normalize(image) for image in x]).to(device)
    with torch.no_grad():
        y = softmax(model(x), dim=1)

    # bring the output to the CPU (if it was on a GPU).
    y = y.cpu()

    for class_ in top_classes:
        # extract the portion of the output corresponding to the class class_.
        outputs[class_].extend(y[:, class_.index].tolist())

print('Generating explanations ...')

# extract the features of each perturbation, which are none other than the segments turned
# "on" or "off" in the respective perturbation.
features = [perturbation.segments_on_off for perturbation in image_perturbations]

# each perturbation is weighted by its proximity to the original image.
# calculating this proximity is a two stage process: the first stage computes the cosine
# distance between the perturbations and the original image.
distances = cdist(
    [perturbation.image.ravel() for perturbation in image_perturbations], [image.ravel()],
    metric='cosine')

# the second stage uses a "kernel" function to convert the distances into proximity
# values.
def kernel_fn(dist, kernel_width=0.25):
    return np.sqrt(np.exp(-(dist ** 2) / kernel_width ** 2))

proximities = kernel_fn(distances).ravel()

# finally we generate explanations for each class in top_classes, the top n classes
# predicted for the original image (n = args.classes).
for class_ in top_classes:

    # the explainer model is a simple Ridge regression model.
    explainer_model = Ridge(alpha=1.0, fit_intercept=True, random_state=101)

    # the model takes as input the features of the perturbations.
    # it is trained to predict the output of the AlexNet model.
    # the proximities of the perturbations to the original image are also taken into
    # account.
    explainer_model.fit(features, outputs[class_], sample_weight=proximities)

    # TODO: what to do of this score?
    score = explainer_model.score(features, outputs[class_], sample_weight=proximities)

    # the coefficients of the regression model are a measure of the "importance" or
    # "saliency" of their corresponding segments.
    segment_vals = explainer_model.coef_

    # this is now a list of two-tuples, where the first element of each two-tuple is a
    # segment number and the second element is its corresponding "importance" value.
    segment_vals = list(enumerate(segment_vals))

    # sort this list in descending order of the magnitude of "importance" values.
    segment_vals.sort(key=lambda v: abs(v[1]), reverse=True)

    # finally we generate the explanation.
    # the explanation image is a copy of the original image, in which we highlight the
    # segments with positive and negative "importance" values.
    exp_image = np.copy(image)

    # we start with only one segment in the explanation image, i.e. the segment numbered
    # zero, for which we use np.zeros_like.
    exp_segments = np.zeros_like(segments)

    # segments with positive "importance" value are highlighted green, while those with
    # negative "importance" value are highlighted red.
    # the pixel value corresponding to this highlight can be chosen as the maximum value
    # of all pixels in the original image.
    mask_pixel_value = np.max(image)

    # segment_vals[:args.segments] chooses the top n segments according to magnitude of
    # "importance" value, where n = args.segments.
    for seg, val in segment_vals[:args.segments]:
        if val > 0:
            # the segment seg in the original image corresponds to a segment numbered as
            # 1 in the explanation image.
            exp_segments[segments == seg] = 1

            # the green channel of this segment is highlighted (in (R, G, B), G occurs at
            # index 1).
            exp_image[segments == seg, 1] = mask_pixel_value

        else:
            # the segment seg in the original image corresponds to a segment numbered as
            # -1 in the explanation image.
            exp_segments[segments == seg] = -1

            # the red channel of this segment is highlighted (R occurs at index 0).
            exp_image[segments == seg, 0] = mask_pixel_value

    # create the explanation image.
    segmented_exp_image = mark_boundaries(exp_image, exp_segments)
    segmented_exp_image = Image.fromarray((segmented_exp_image * 255).astype('uint8'), mode='RGB')
    file = 'explanation_{}_{}.png'.format(class_.name, image_path.stem)
    segmented_exp_image.save(file)
    print('Explanation plot saved to', file)
