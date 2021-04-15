from damri.batch_iter import sample_center_uniformly, extract_patch, get_random_slice  # augm_spatial
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, multiply
from dpipe.im.shape_utils import prepend_dims
from dpipe.im.utils import identity


def get_random_patch_2d(image_slc, segm_slc, x_patch_size, y_patch_size):
    sp_dims_2d = (-2, -1)
    center = sample_center_uniformly(segm_slc.shape, y_patch_size, sp_dims_2d)
    x, y = extract_patch((image_slc, segm_slc, center), x_patch_size, y_patch_size)
    return x, y


ids_sampling_weights = None
slice_sampling_interval = 1

augm_fn = identity
# augm_fn = partial(augm_spatial, dims_flip=(-2, -1), dims_rot=(-2, -1))

batch_iter = Infinite\
(
    load_by_random_id(dataset.load_image, dataset.load_segm, ids=train_ids, #source
                      weights=ids_sampling_weights, random_state=seed),
    unpack_args(get_random_slice, interval=slice_sampling_interval),
    unpack_args(get_random_patch_2d, x_patch_size=x_patch_size, y_patch_size=y_patch_size),
    multiply(prepend_dims),
    augm_fn,
    batch_size=batch_size, batches_per_epoch=batches_per_epoch
)
