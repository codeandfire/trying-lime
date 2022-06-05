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
    '--hide-colour', type=str, choices=['black', 'white', 'mean_gray', 'mean_rgb'],
    default='mean_gray',
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

assert len(classes) == 1000, f'{len(classes)} classes found instead of 1000'
assert classes[0].name == 'tench' and classes[0].wordnet_id == 'n01440764', \
        (f'name of first class is {classes[0].name} instead of tench and wordnet ID is '
         f'{classes[0].wordnet_id} instead of n01440764')

image_path = Path(args.image_path)

# open the image, transform it to the standard size of 224 x 224 pixels, and extract its
# pixel values into a Numpy array.
image = Image.open(image_path)
image = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
image = np.array(image)

assert image.shape == (224, 224, 3), \
        f'image has shape {image.shape} instead of (224, 224, 3)'
assert (0 <= image).all() and (image <= 255).all(), \
        'image does not have pixel values in the range of 0 to 255'

# find the segments (superpixels) of the image using the QuickShift algorithm.
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2, random_seed=123)

assert segments.shape == image.shape[:2], \
        f'segments array has shape {segments.shape} while image has shape {image.shape}'

# create an image marking the segments found.
segmented_image = mark_boundaries(image, segments)

assert segmented_image.shape == image.shape, \
        f'segmented image has shape {segmented_image.shape} while image has shape {image.shape}'
assert (0 <= segmented_image).all() and (segmented_image <= 1).all(), \
        'segmented image does not have normalized pixel values in the range of 0 to 1'

segmented_image = Image.fromarray((segmented_image * 255).astype('uint8'), mode='RGB')
file = 'segmentation_{}.png'.format(image_path.stem)
segmented_image.save(file)
print('Segmentation plot saved to', file)

# the total number of segments.
num_segments = len(np.unique(segments))

assert num_segments > 1, 'only one segment found'

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

    elif args.hide_colour == 'mean_gray':
        # find all the pixels that fall within the area of this segment, and find the mean
        # of their values.
        hide_colour[seg] = np.mean(image[segments == seg])

        assert 0 <= hide_colour[seg] <= 255, \
                f'invalid hide_colour pixel value calculated: {hide_colour[seg]}'

        # duplicate this mean value across all the RGB channels of hide_colour.
        hide_colour[seg] = (hide_colour[seg], hide_colour[seg], hide_colour[seg])

    else:
        # find all the pixels that fall within the area of this segment, and find the mean
        # of their values, but separately across the three RGB colour channels.
        hide_colour[seg] = tuple(np.mean(image[segments == seg], axis=0))

        assert len(hide_colour[seg]) == 3, \
                f'invalid hide_colour with {len(hide_colour[seg])} instead of three colour channels'

assert (hide_colour[0] != hide_colour[1]
        if args.hide_colour in ['mean_gray', 'mean_rgb']
        else True), \
            (f"with 'mean' hide_colour option, hide_colour for two adjacent segments is "
             f'the same: {hide_colour[0]}')

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
        perturbed_image[segments == seg] = hide_colour[seg]

    assert perturbed_image.shape == image.shape, \
            f'perturbed image has shape {perturbed_image.shape} while image has shape {image.shape}'
    assert not (perturbed_image == image).all() or 0 not in segments_on_off, \
            'perturbed image is the same as image despite some segments turned off'
    assert (perturbed_image == image).any() or 1 not in segments_on_off, \
            'perturbed image does not share any pixels with image despite some segments turned on'

    return perturbation(image=perturbed_image, segments_on_off=segments_on_off)


# carry out the given number of perturbations in parallel over all available cores.
print('Generating perturbations ...')
image_perturbations = Parallel(n_jobs=-1, verbose=1)(
    [delayed(perturb_image)(seed=10*i) for i in range(args.perturbations)])

# add the original image too, as a perturbation, with all segments turned "on".
image_perturbations.append(
    perturbation(image=image, segments_on_off=np.ones((num_segments,))))

assert len(image_perturbations) == args.perturbations + 1, \
        (f'number of perturbations generated (including input image) is '
         f'{len(image_perturbations)} instead of {args.perturbations + 1}')

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

assert x.shape == (1, 3, 224, 224), \
        f'input tensor has shape {x.shape} instead of (1, 3, 224, 224)'
assert (-5 <= x).all() and (x <= 5).all(), 'input tensor does not contain normalized pixel values'

with torch.no_grad():
    # the predicted probability values.
    y = softmax(model(x), dim=1)

assert y.shape == (1, 1000), \
        f'output tensor has shape {y.shape} instead of (1, 1000)'
assert torch.allclose(torch.sum(y), torch.tensor(1.0)), \
        'output probability values do not sum to one'

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

# calculate number of batches.
num_batches = (len(image_perturbations) // args.batch_size
               + min(len(image_perturbations) % args.batch_size, 1))

assert num_batches * args.batch_size >= len(image_perturbations), \
        f'number of batches calculated is {num_batches} which is too less'
assert (num_batches - 1) * args.batch_size < len(image_perturbations), \
        f'number of batches calculated is {num_batches} which is too large'

for i in tqdm(range(num_batches), total=num_batches):

    # extract a batch of perturbations.
    x = image_perturbations[(args.batch_size * i): (args.batch_size * (i+1))]
    x = [perturbation.image for perturbation in x]
    # this would be expected to work, but it doesn't: x = normalize(x).to(device)
    # hence we have to resort to the following line.
    x = torch.stack([normalize(image) for image in x]).to(device)

    assert x.shape == (args.batch_size, 3, 224, 224) if i < num_batches - 1 else True, \
            f'input tensor has shape {x.shape} instead of ({args.batch_size}, 3, 224, 224)'

    # pass the batch to the model and get the predicted probability values.
    with torch.no_grad():
        y = softmax(model(x), dim=1)

    assert y.shape == (args.batch_size, 1000) if i < num_batches - 1 else True, \
            f'output tensor has shape {y.shape} instead of ({args.batch_size}, 1000)'
    assert (torch.allclose(torch.sum(y, axis=1).cpu(), torch.ones(args.batch_size))
            if i < num_batches - 1
            else True), \
            'output probability values do not sum to one'

    # bring the output to the CPU (if it was on a GPU).
    y = y.cpu()

    for class_ in top_classes:
        # extract the portion of the output corresponding to the class class_.
        y_class = y[:, class_.index]

        assert y_class.shape == (args.batch_size,) if i < num_batches - 1 else True, \
                ('output tensor corresponding to a certain class has shape '
                 f'{y_class.shape} instead of ({args.batch_size},)')

        outputs[class_].extend(y_class.tolist())

assert len(outputs[top_classes[0]]) == len(image_perturbations), \
        (f'output probability values for a certain class are {len(outputs[top_classes[0]])}'
         f' in number instead of {len(image_perturbations)}')

print('Generating explanations ...')

# extract the features of each perturbation, which are none other than the segments turned
# "on" or "off" in the respective perturbation.
features = np.array([perturbation.segments_on_off for perturbation in image_perturbations])

assert features.shape == (len(image_perturbations), num_segments), \
        (f'array of features has shape {features.shape} instead of ({len(image_perturbations)}, '
         f'{num_segments})')

# each perturbation is weighted by its proximity to the original image.
# calculating this proximity is a two stage process: the first stage computes the cosine
# distance between the perturbations and the original image.
distances = cdist(
    [perturbation.image.ravel() for perturbation in image_perturbations], [image.ravel()],
    metric='cosine')
distances = distances.ravel()

assert distances.shape == (len(image_perturbations),), \
        (f'array of distances between perturbations and the image has shape '
         f'{distances.shape} instead of ({len(image_perturbations)},)')
assert distances[-1] == 0.0, \
        ('the last perturbation is identical to the image and yet has a nonzero distance '
         f'{distances[-1]} from the image')

# the second stage uses a "kernel" function to convert the distances into proximity
# values.
def kernel_fn(dist, kernel_width=0.25):
    return np.sqrt(np.exp(-(dist ** 2) / kernel_width ** 2))


proximities = kernel_fn(distances)

assert proximities.shape == (len(image_perturbations),), \
        (f'array of proximities between perturbations and the image has shape '
         f'{proximities.shape} instead of ({len(image_perturbations)},)')
assert proximities[-1] == 1.0, \
        ('the last perturbation is identical to the image and yet has a proximity '
         f'{proximities[-1]} less than 1.0 from the image')
assert np.argmax(distances) == np.argmin(proximities), \
        ('the index of the perturbation with the largest distance from the image is '
         f'{np.argmax(distances)} while the index of the perturbation with the smallest '
         f'proximity to the image is {np.argmin(proximities)}: both are different whereas'
         'they should be the same')

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

    assert segment_vals.shape == (num_segments,), \
            (f'array of coefficients of the explainer model has shape {segment_vals.shape} '
             f'instead of ({num_segments},)')

    # this is now a list of two-tuples, where the first element of each two-tuple is a
    # segment number and the second element is its corresponding "importance" value.
    segment_vals = list(enumerate(segment_vals))

    # sort this list in descending order of the magnitude of "importance" values.
    segment_vals.sort(key=lambda v: abs(v[1]), reverse=True)

    assert np.argmax(np.abs(explainer_model.coef_)) == segment_vals[0][0], \
        ('segment importance values are not ordered correctly: segment with the highest '
         f'importance value is not segment {np.argmax(np.abs(explainer_model.coef_))}')

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

    # counter which holds segment numbers.
    # during the initialization of exp_segments, we started with segment number zero.
    # the next segment will have segment number one, and this counter will be incremented
    # as more and more segments are added.
    exp_seg = 1

    # segment_vals[:args.segments] chooses the top n segments according to magnitude of
    # "importance" value, where n = args.segments.
    for seg, val in segment_vals[:args.segments]:
        if val > 0:
            exp_segments[segments == seg] = exp_seg

            # the green channel of this segment is highlighted (in (R, G, B), G occurs at
            # index 1).
            exp_image[segments == seg, 1] = mask_pixel_value

            assert (exp_image[segments == seg, 1] >= image[segments == seg, 1]).all(), \
                    f'segment {seg} is not highlighted as green in the explanation'
            assert (exp_image[segments == seg][:, [0, 2]] == image[segments == seg][:, [0, 2]]).all(), \
                    f'segment {seg} is incorrectly highlighted: red and blue pixel values are altered'

        else:
            exp_segments[segments == seg] = exp_seg

            # the red channel of this segment is highlighted (R occurs at index 0).
            exp_image[segments == seg, 0] = mask_pixel_value

            assert (exp_image[segments == seg, 0] >= image[segments == seg, 0]).all(), \
                    f'segment {seg} is not highlighted as red in the explanation'
            assert (exp_image[segments == seg][:, [1, 2]] == image[segments == seg][:, [1, 2]]).all(), \
                    f'segment {seg} is incorrectly highlighted: green and blue pixel values are altered'

        exp_seg += 1

    assert len(np.unique(exp_segments)) in [args.segments + 1, args.segments], \
            f'explanation image has an invalid number of segments: {len(np.unique(exp_segments))}'

    # create the explanation image.
    segmented_exp_image = mark_boundaries(exp_image, exp_segments)
    segmented_exp_image = Image.fromarray((segmented_exp_image * 255).astype('uint8'), mode='RGB')
    file = 'explanation_{}_{}.png'.format(class_.name, image_path.stem)
    segmented_exp_image.save(file)
    print('Explanation plot saved to', file)
