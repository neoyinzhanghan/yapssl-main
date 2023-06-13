from PIL import Image
from yapssl_models.mae_lightning import MAE
from lightly.models import utils
import torchvision
import os
from PIL import Image, ImageDraw, ImageFont
from lightly.transforms.mae_transform import MAETransform
from lightly.data import LightlyDataset
from torchvision.utils import make_grid
import torch
import PIL
import numpy as np


images_dir = '/Users/neo/Documents/Research/CP/SSL/yapssl-main/examples/test_patches'
model_path = '/Users/neo/Documents/Research/CP/MyCheckpoints/MAE-1/epoch=799-step=16000.ckpt'
save_dir = '/Users/neo/Documents/Research/CP/SSL/yapssl-main/examples/mae_reconstructions'


# get a list of filepaths to .jpg or .png images in images_dir
image_paths = []
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_paths.append(os.path.join(images_dir, filename))

print('Loading image...')
image_path = image_paths[0]
image = Image.open(image_path)

# terminate the program if the dataset is empty

# load the model
print('Loading model...')
model = MAE.load_from_checkpoint(model_path, lr=0, batch_size=1)


from torchvision.utils import make_grid
from PIL import Image

def tensor_to_img(patches: torch.Tensor):
    """
    Converts a batch of image patches from a tensor to a PIL image.

    Args:
        patches:
            Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2)

    Returns:
        The PIL image.
    """
    # Unnormalize the patches
    # patches = patches * 0.5 + 0.5

    # N, P, S = (batch_size, num_patches, patch_size**2 * channels)
    N, P, S = patches.shape
    patch_size = int((S / 3) ** 0.5)  # Assuming 3 color channels
    patch_h = patch_w = int(P ** 0.5)

    # Reconstruct the original image from patches
    patches = patches.view(N, patch_h, patch_w, 3, patch_size, patch_size)
    images = patches.permute(0, 3, 1, 4, 2, 5)
    images = images.reshape(N, 3, patch_h * patch_size, patch_w * patch_size)

    # Convert the first image in the batch to a PIL image
    img_np = images[0].numpy().transpose(1, 2, 0)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

    return img_pil


def mae_reconstruct_image(image, model, mask_ratio=0.75, present_original=True, present_masked=False, present_reconstructed=True, present_interlaced=False, device='cpu'):
    """ Reconstruct a PIL image using a trained MAE model.
    """

    assert model.ssl_arch == 'mae', 'This function only works with MAE models!'
    assert present_original or present_masked or present_reconstructed, 'At least one of present_original, present_masked, or present_reconstructed must be True!'

    # duplicate the image so that we can present the original, masked, and reconstructed versions
    image_original = image.copy()

    # feed the image into the model to get the reconstructed image
    model.eval()
    batch_size = 1

    # convert the PIL image to a tensor
    image_tensor = torchvision.transforms.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0) # add a batch dimension
    image_tensor = image_tensor.to(device)

    idx_keep, idx_mask = utils.random_token_mask(size=(batch_size, model.sequence_length),
                                                 mask_ratio=model.mask_ratio,
                                                 device=image_tensor.device)
                                                 

    # get the encoding of the image
    print('Encoding image...')
    x_encoded = model.forward_encoder(image_tensor, idx_keep)

    # get the masked image
    print('Masking image...')
    x_decode = model.decoder.embed(x_encoded)
    x_masked = utils.repeat_token(model.mask_token, (batch_size, model.sequence_length))
    x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

    image_masked = torchvision.transforms.ToPILImage()(x_masked.squeeze(0))

    # get the reconstructed image
    print('Reconstructing image...')
    x_pred = model.forward_decoder(x_encoded, idx_keep, idx_mask)
    image_reconstructed = torchvision.transforms.ToPILImage()(x_pred.squeeze(0))

    # get the image interlacing non-masked patches and the reconstructed masked patches
    print('Interlacing image...')
    x_interlaced = utils.set_at_index(x_masked, idx_mask, x_pred)

    image_interlaced = torchvision.transforms.ToPILImage()(x_interlaced.squeeze(0))

    # save the concatenated image
    print('Saving concatenated image...')
    image_concatenated = Image.new('RGB', (image_original.width * 4, image_original.height))

    if present_original:
        image_concatenated.paste(image_original, (0, 0))
        # add a vertical line to separate the original image from the masked image
        draw = ImageDraw.Draw(image_concatenated)
        draw.line((image_original.width, 0, image_original.width, image_original.height), fill=128)

        # caption the original image
        draw.text((0, 0), 'Original', fill=(255, 255, 255))
    
    if present_masked:
        image_concatenated.paste(image_masked, (image_original.width, 0))

        # caption the masked image
        draw.text((image_original.width, 0), 'Masked', fill=(255, 255, 255))

    if present_reconstructed:
        image_concatenated.paste(image_reconstructed, (image_original.width * 2, 0))

        # caption the reconstructed image
        draw.text((image_original.width * 2, 0), 'Reconstructed', fill=(255, 255, 255))
    
    if present_interlaced:
        image_concatenated.paste(image_interlaced, (image_original.width *3, 0))

        # caption the interlaced image
        draw.text((image_original.width * 3, 0), 'Interlaced', fill=(255, 255, 255))

    # generate filename of the concatenated image
    root = os.path.splitext(os.path.basename(image_path))[0]
    new_filename = root + '_reconstructed.png'
    new_path = os.path.join(save_dir, new_filename)

    # save the concatenated image at new_path
    print('Saving concatenated image...')

    image_concatenated.save(new_path)


if __name__ == '__main__':
    mae_reconstruct_image(image, model, mask_ratio=0.75, present_original=True, present_masked=True, present_reconstructed=True, present_interlaced=True, device='cpu')