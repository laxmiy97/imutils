import os
import re

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from natsort import natsorted

from imutils import MicroscopeDataReader
import dask.array as da
from skimage.morphology import binary_erosion
from scipy.ndimage import label, sum as ndi_sum, center_of_mass
from scipy.ndimage import gaussian_filter
import warnings
import traceback


def tiff2avi(tiff_path, avi_path, fourcc, fps):
    """
    Convert tiff file into avi file with the specified fourcc codec and fps
    The isColor parameter of the writer is harcoded set to False.

    Update: Lukas Reader: Function can now also receive a source folder path with Lukas reader and directly convert
    NDTif to Avi

    Parameters:
    -----------
    tiff_path: str,
        Path to the tiff file
    avi_path: str
        Path to the output file
    fourcc: str, fourcc code
        0 means no coompression, other codecs will have some compression
        To learn more visit: https://www.fourcc.org/
    fps: float (should it be int?)
        Number of frames per second at which the recording was acquired

    To improve:
    ----------
    Write Multifile as option, so it can be set to True

    """

    # corrects fourcc nomenclature
    if fourcc == '0':
        fourcc = 0
    else:
        fourcc = cv2.VideoWriter_fourcc(*fourcc)

    # make fps a float
    fps = float(fps)
    print('Path:', tiff_path)

    tiff_path = os.path.normpath(tiff_path)
    print('Path resolved:', tiff_path)

    tiff_path = os.path.abspath(tiff_path)
    print('Path absolute:', tiff_path)

    if os.path.isdir(tiff_path):
        # Initialize for directory
        reader_obj = MicroscopeDataReader(tiff_path)
    elif os.path.isfile(tiff_path):
        # Initialize for a file (any file, not limited to .btf)
        reader_obj = MicroscopeDataReader(tiff_path, as_raw_tiff=True, raw_tiff_num_slices=1)
    else:
        raise ValueError("Invalid input file path. Please provide a valid directory or file.")

    tif = da.squeeze(reader_obj.dask_array)
    frame_size_unknown_len = tif[0].shape

    print(frame_size_unknown_len)
    # if image has channels get height and width (ignore 3rd output)

    print(f"Opening video writer with frame size {frame_size_unknown_len}")
    if len(frame_size_unknown_len) == 3:
        # Time should be first
        _, frame_height, frame_width = frame_size_unknown_len
    elif len(frame_size_unknown_len) == 2:
        frame_height, frame_width = frame_size_unknown_len
    else:
        raise ValueError(f"Invalid frame size, expected 2 or 3 dimensions, got {frame_size_unknown_len}")

    video_out = cv2.VideoWriter(avi_path, apiPreference=0, fourcc=fourcc, fps=fps,
                                frameSize=(frame_width, frame_height), isColor=False)

    for i, img in enumerate(tif):
        # img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        video_out.write(cv2.convertScaleAbs(np.array(img)))  # if img is uint16 it can't save it
        # if i>20: break
    video_out.release()
    print(f'Finished! Video saved at: {avi_path}')


def ometiff2bigtiff(path, output_filename=None):
    """
    List all ome.tiff in a directory and make them one bigtiff
    Somehow it gives an error for the last ome tiff, but resulting .btf is fine.

    IMPORTANT: This ometiff2big tiff removes the Z-Stack information in a recording with Z stacks!
    At least if the number of Z Stacks is inconsistent, which is the case for the current writer in ome.tiff. While recording, the microscope saves the ome.tiff file even before the z-stack is finished.

    Parameters:
    -----------
    path: str,
        Path to the directory containing the several ome tiff files.
    output_filename: str,
    if not defined it will be generated based on the path name.
    """
    print(path)
    # Create 'output' folder if it doesn't exist
    output_folder = os.path.join(path, "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # find number of files in path that end with "NDTiffStack.tif"
    tiff_files = [name for name in os.listdir(path) if name.endswith(".tif")]
    # Sort files (assuming the suffixes are numerical like _1, _2, etc., so they are sorted correctly)
    tiff_files = natsorted(tiff_files)
    print(f"Found TIFF files: {tiff_files}")

    if len(tiff_files) == 0:
        print("Aborted because no nd.tiff files found in path, therefore no bigtiff was created.")
        return
    if output_filename is None:
        if path.endswith('/'):
            output_filename = os.path.join(output_folder, 'raw_stack.btf')
        else:
            output_filename = os.path.join(output_folder, 'raw_stack.btf')
    #else:
        # Ensure the custom output file is in the output folder
     #   output_filename = os.path.join(output_folder, os.path.basename(output_filename))

    print(f"Output file will be saved to: {output_filename}")

    with tiff.TiffWriter(output_filename, bigtiff=True) as output_tif:
        # print(f'list of files is {os.listdir(path)}')
        for file in tiff_files:
            # print(os.path.join(path, file))
            # if file.endswith('.tif'):
                # print(os.path.join(path, file))
                with tiff.TiffFile(os.path.join(path, file)) as tif:
                    # print('length of pages is: ', len(tif.pages))
                    # print('length of series is: ', len(tif.series))
                    for idx, page in enumerate(tif.pages):
                        # print(idx)
                        img = page.asarray()
                        output_tif.write(img, photometric='minisblack',
                                         contiguous=True)  # , description=omexmlMetadataString)
# path='/Users/ulises.rey/local_data/2022-02-23_11-28_immobilised_1_Ch0'
# ometiff2bigtiff(path)

def ometiff2bigtiffZ(path, output_dir=None, actually_write=True, num_slices=None):
    """
    This function was copied from video_conversions/Python/bigtiff/
    """
    if output_dir is None:
        output_dir = path
    if path.endswith('/'):
        output_filename = output_dir + re.split('/', path)[-2] + 'bigtiff.btf'
    else:
        output_filename = output_dir + '/' + re.split('/', path)[-1] + 'bigtiff.btf'

    print(f"File will be written that is divisible by {num_slices}")
    print(f"And written to filename {output_filename}")
    total_num_frames = 0
    buffer = []

    reader_obj = MicroscopeDataReader(path, as_raw_tiff=True, raw_tiff_num_slices=1)
    tif = da.squeeze(reader_obj.dask_array)
    with tiff.TiffWriter(output_filename, bigtiff=True) as output_tif:
        '''
        for i_file, file in enumerate(natsorted(os.listdir(path))):
            if not file.endswith('ome.tif') or 'bg' in file:
                continue
            this_ome_tiff = os.path.join(path, file)
            print("Currently reading: ")
            print(this_ome_tiff)
            with tiff.TiffFile(this_ome_tiff) as tif:
        '''
        for i, page in enumerate(tif):
            #print(f'Page {i}/{len(tif.pages)} in file {i_file}')
            # Bottleneck line
            #img = page.asarray()
            img = np.array(page)
            # Convert to proper format, and write single frame
            # img = (alpha*img).astype('uint8')
            total_num_frames += 1
            if num_slices is None:
                if actually_write:
                    output_tif.write(img, photometric='minisblack')
            else:
                buffer.append(img)
                if len(buffer) >= num_slices:
                    print(f"Writing {num_slices} frames from buffer...")
                    for img in buffer:
                        if actually_write:
                            output_tif.write(img, photometric='minisblack', contiguous=True)
                    buffer = []
            if len(buffer) > 0:
                print(f"{len(buffer)} frames not written")

                # if num_frames is not None and i > num_frames: break


def max_projection_3d(input_filepath, output_filepath, fold_increase=3, nplanes=24, flip=False):
    """
    Create a visualization image of a volume, with the 3 max projections possible.
    To improve: Make another function that does the same but in a figure. Then the fold increase would not have to be int.
    Parameters:
    ------------
    input_filepath: str,

    output_filepath: str,

    fold_increase: int, (it can't be float, so the ratio dimensions xyz are not exactly real)
        Expands the z dimension so that the image has crrect dimensions. Depends on the ratio between xy pixel size and z-step size.
    nplanes: int,

    """
    with tiff.TiffWriter(output_filepath, bigtiff=True) as output_tif:
        with tiff.TiffFile(input_filepath) as tif:
            for idx, page in enumerate(tif.pages):
                img = page.asarray()
                # if it is the first plane of the volume create an empty img_stack shape=(w, h, z)
                if idx % nplanes == 0:
                    # print('index is ',idx)
                    img_stack = np.full(shape=(img.shape[0], img.shape[1], nplanes), fill_value=np.nan, dtype=np.uint16)

                # fill the idx plane with the img
                img_stack[:, :, idx % nplanes] = img

                # if it is the last plane on the volume do max projection on the three axis and concatenate them
                if idx % nplanes == nplanes - 1:
                    max0 = np.max(img_stack, axis=0)
                    max1 = np.max(img_stack, axis=1)
                    max2 = np.max(img_stack, axis=2)

                    # extends the YZ and XZ max projection for better visualization based on input fold_increase variable
                    max0 = np.repeat(max0, fold_increase, axis=1)
                    # rotates max0
                    max0 = np.transpose(max0)
                    max1 = np.repeat(max1, fold_increase, axis=1)

                    # defines corner array dimensions and fill value of the corner matrix
                    fill_value = 100
                    corner_matrix = np.full((fold_increase * img_stack.shape[2], fold_increase * img_stack.shape[2]),
                                            fill_value, dtype='uint16')

                    # flip if needed (for Green Channel, since the image is mirrored compared to red channel)
                    if flip is True:
                        max2 = cv2.flip(max2, 1)
                        max0 = cv2.flip(max0, 1)
                    # concatenate the different max projections into one image

                    vert_conc_1 = cv2.hconcat([max2, max1])
                    vert_conc2 = cv2.hconcat([max0, corner_matrix])
                    final_img = cv2.vconcat([vert_conc_1, vert_conc2])

                    # save the 3 max projection image
                    output_tif.write(final_img, photometric='minisblack', contiguous=True)


def stack_subtract_background(input_filepath, output_filepath, background_img_filepath, invert=True):
    """
    Subtract the background image from a btf stack
    its parser in imutils_parser.py might not work due to being it boolean
    Parameters:
    ----------
    input_filepath, str
    input path to the tiff file
    output_filepath, str
    path to where the file will be written
    background_img_filepath, str
    path to the background image
    inverse, bool
    It is default True because the function before did not have this parameter and was doing the inverse by default
    Returns:
    ----------
    """
    print("Do not use this function with a parser unless you are sure it works (See docstring)")

    # load background image
    reader_obj_background = MicroscopeDataReader(background_img_filepath, as_raw_tiff=True, raw_tiff_is_2d=True)
    bg_img = np.array(da.squeeze(reader_obj_background.dask_array))
    assert bg_img.dtype == np.uint8
    try:
        # Try to read the input file as a .btf
        reader_obj_video = MicroscopeDataReader(input_filepath, as_raw_tiff=True, raw_tiff_num_slices=1)
    except TypeError:
        # Try to read as an ndtiff (input_filepath should be a folder)
        reader_obj_video = MicroscopeDataReader(input_filepath, as_raw_tiff=False)

    tif = da.squeeze(reader_obj_video.dask_array)
    assert tif.dtype == np.uint8

    if invert:
        bg_img = cv2.bitwise_not(bg_img) # .astype(dtype=np.uint8)
        print("inverting background image")
    else:
        print("using background as it is")
    mean_bg = 2*(np.max(bg_img)-np.min(bg_img))
    bg_img = bg_img.astype(np.float64)
    print(mean_bg)
    with tiff.TiffWriter(output_filepath, bigtiff=True) as tif_writer:
        for i, img in enumerate(tif):
            img = np.array(img)
            if invert:
                img = cv2.bitwise_not(img)
            img = img.astype(np.float64)
            #print(np.mean(img))
            new_img = cv2.subtract(img, bg_img)
            #print(f'max: {np.max(new_img)}, min: {np.min(new_img)}')
            new_img += mean_bg
            #print(f'max+: {np.max(new_img)}, min+: {np.min(new_img)}')
            #print(f'dtype after addition: {new_img.dtype}')
            new_img = new_img.astype(np.uint8)
            #print(new_img.dtype)
            tif_writer.write(new_img, photometric='minisblack',  contiguous=True)
            


def stack_make_binary(stack_input_filepath: str, stack_output_filepath: str, threshold: float,
                      max_value: float = 65535.0, min_component_size: int = 3):
    """
    write a binary stack based on lower and higher threshold
    Parameters:
    -------------
    stack_input_filepath, str
    stack_output_filepath, str
    lower_threshold, float
    max_val, float
    Returns:
    -------------
    None
    """
    print(f"--- Starting Binary Filtering (Output: uint16) ---")
    print(f"Input: {stack_input_filepath}, Output: {stack_output_filepath}")
    print(f"Threshold: {threshold}, Max Value: {max_value}, Min Component Size > {min_component_size}")
    
    output_dtype = np.uint16
    max_uint16 = np.iinfo(output_dtype).max # 65535

    # --- Validate and Prepare max_value ---
    try:
        max_value_int = int(round(max_value))
        # Clamp to valid uint16 range
        max_value_int = max(0, min(max_value_int, max_uint16))
        max_value_out = output_dtype(max_value_int)
    except (ValueError, TypeError):
        print(f"Warning: Invalid max_value ({max_value}). Using {max_uint16}.")
        max_value_out = output_dtype(max_uint16)
    print(f"Using output 'on' value: {max_value_out}")

    # --- Load Input Data ---
    try:
        # Use MicroscopeDataReader or replace with tifffile.imread or TiffFile
        reader_obj = MicroscopeDataReader(stack_input_filepath, as_raw_tiff=True, raw_tiff_num_slices=1)
        tif = da.squeeze(reader_obj.dask_array)
        print(f"Input shape (Dask): {tif.shape}, dtype: {tif.dtype}")
    except NameError:
        print("Warning: MicroscopeDataReader not found. Using tifffile.")
        return

    # --- Process Frames and Write Output ---
    with tiff.TiffWriter(stack_output_filepath, bigtiff=True) as tif_writer:
        # Iterate through each frame
        for i, img_frame in enumerate(tif):
            # Ensure frame is a NumPy array (computes if using Dask)
            img = np.array(img_frame)

            # 1. Apply Intensity Threshold
            # Result is float64, needs conversion later
            ret, initial_binary_img_float = cv2.threshold(img, threshold, float(max_value_out), cv2.THRESH_BINARY)

            # 2. Create Boolean Mask (directly from float is fine)
            initial_mask = initial_binary_img_float > 0

            # 3. Label Connected Components
            labeled_mask, num_features = label(initial_mask, structure=np.ones((3,3)))

            # Initialize final frame as zeros with the correct uint16 type
            final_frame = np.zeros_like(img, dtype=output_dtype)

            # 4. Filter Components by Size
            if num_features > 0:
                component_sizes = ndi_sum(initial_mask, labels=labeled_mask, index=np.arange(1, num_features + 1))
                # Note: min_component_size is now correctly defined as a function parameter
                large_enough_labels = [label_index for label_index, size in enumerate(component_sizes, start=1)
                                       if size > min_component_size]

                if large_enough_labels:
                    # Create mask of only large components
                    size_filtered_mask = np.isin(labeled_mask, large_enough_labels)
                    # Generate final frame using the correct output type
                    final_frame = np.where(size_filtered_mask, max_value_out, output_dtype(0))
            # If no features or no large features, final_frame remains zeros

            # 5. Write the Processed Frame (as uint16)
            tif_writer.write(final_frame.astype(output_dtype), contiguous=True, photometric='minisblack')

    print(f"--- Finished writing filtered binary stack ({output_dtype}) to {stack_output_filepath} ---")
   

def stack_normalise(stack_input_filepath: str, stack_output_filepath: str, alpha: float,
                      beta: float):
    """
    Normalise the stack
    Parameters:
    -------------
    stack_input_filepath, str
    stack_output_filepath, str
    alpha, float
    beta, float
    Returns:
    -------------
    None
    """
    reader_obj = MicroscopeDataReader(stack_input_filepath, as_raw_tiff=True, raw_tiff_num_slices=1)
    tif = da.squeeze(reader_obj.dask_array)
    with tiff.TiffWriter(stack_output_filepath, bigtiff=True) as tif_writer:
        for i, img in enumerate(tif):
            normalised_img = cv2.normalize(np.array(img), None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
            tif_writer.write(normalised_img, contiguous=True)


def stack_subsample(stack_input_filepath, stack_output_filepath, range):
    """Subsample the stack based on the given range

    """
    with tiff.TiffWriter(stack_output_filepath, bigtiff=True) as tif_writer:
        with tiff.TiffFile(stack_input_filepath) as tif:
            for i, page in enumerate(tif.pages[range]):
                # loads the first frame
                img = page.asarray()
                tif_writer.write(img, contiguous=True)


def make_contour_based_binary(stack_input_filepath, stack_output_filepath, median_blur, threshold,
                              max_value, contour_size, tolerance, inner_contour_area_to_fill, gaussian_blur=0, substract_background=1):

    """
    Produce a binary image based on contour and inner contour sizes, by calling draw_some_contours()
    better than the make_binary before which was on centerline package
    TODO: Split into several functions. stack_binary already exists. From that oen could have the fill inner contours
    Parameters:
    -----------
    :param stack_input_filepath:
    :param stack_output_filepath:
    :param median_blur: (needs an odd number)
    :param threshold:
    :param max_value:
    :param contour_size:
    :param tolerance: contour sizes will be considered between contour_size*-tolerance and cotnour_size*tolerance
    :param inner_contour_area_to_fill: all inner contours below this value will be filled (will be part of the worm)
    :return:
    Returns:
    --------
    """
    with tiff.TiffWriter(stack_output_filepath, bigtiff=True) as tif_writer:
        with tiff.TiffFile(stack_input_filepath) as tif:
            for i, page in enumerate(tif.pages):
                # loads the first frame
                img = page.asarray()

                if substract_background != 1:
                    img = 255 - img

                # median Blur
                if gaussian_blur != 0:
                    img = cv2.GaussianBlur(img, (gaussian_blur, gaussian_blur), 0)

                if median_blur != 0:
                    img = cv2.medianBlur(img, median_blur)

                # apply threshold
                ret, new_img = cv2.threshold(img, threshold, max_value, cv2.THRESH_BINARY)
                # draw_some_contours does not need imfunctions.draw_some_contours in here. But outside this file.
                worm_contour_img = draw_some_contours(new_img, contour_size=contour_size, tolerance=tolerance,
                                                      inner_contour_area_to_fill=inner_contour_area_to_fill)

                tif_writer.write(worm_contour_img, contiguous=True)


def unet_segmentation_contours_with_children(binary_input_filepath, raw_input_filepath, output_filepath, weights_path):
    """
    Run through the unet segmentation the contours with children.
    TO DO: It would be more efficient to do a list of frames.
    Parameters:
    -----------
    input_filepath, str
    output_filepath, str
    weights_path, str
    Returns:
    -----------

    """

    from imutils.src.model import unet
    model = unet()
    model.load_weights(weights_path)

    reader_obj_binary = MicroscopeDataReader(binary_input_filepath, as_raw_tiff=True, raw_tiff_num_slices=1)
    reader_obj_raw = MicroscopeDataReader(raw_input_filepath, as_raw_tiff=True, raw_tiff_num_slices=1)
    binary_tif = da.squeeze(reader_obj_binary.dask_array)
    raw_tif = da.squeeze(reader_obj_raw.dask_array)

    with tiff.TiffWriter(output_filepath, bigtiff=True) as tif_writer:

        for i, img in enumerate(binary_tif):

            img = np.array(img)
            # find contours
            cnts, hierarchy = cv2_find_contours_compatibility_mode(img)

            # if there is None or less than 2 contours: write the binary and continue
            if cnts is None or len(cnts) < 2:
                tif_writer.write(img, contiguous=True)
                continue

            # find contours with children
            contours_with_children = extract_contours_with_children(img)

            # If there are no contours_with_children (empty list), write binary too
            if contours_with_children == []:
                tif_writer.write(img, contiguous=True)
                continue

            # make a copy of the original image here in order to paste more than one contour with children
            #TODO: Is this copy needed?
            # new_img = raw_tif.pages[i].asarray()
            new_img = raw_tif[i].compute().copy()
            for cnt_idx, cnt in enumerate(contours_with_children):
                x, y, w, h = cv2.boundingRect(cnt)
                # make the crop
                cnt_img = new_img[y:y + h, x:x + w]

                #TODO: can the unet_functions.unet_segmentation() be used here instead of all this, to produce results_reshaped?
                #TODO: Be careful, because that function does NOT normalize to 255!
                # run U-Net network:
                cnt_img = cv2.resize(cnt_img, (256, 256))
                cnt_img = np.reshape(cnt_img, cnt_img.shape + (1,))
                cnt_img = np.reshape(cnt_img, (1,) + cnt_img.shape)

                # normalize to 1 by dividing by 255
                cnt_img = cnt_img / 255
                results = model.predict(cnt_img)
                # reshape results
                results_reshaped = results.reshape(256, 256)
                # resize results
                results_reshaped = cv2.resize(results_reshaped, (w, h))
                # multiply it by 255
                results_reshaped = results_reshaped * 255

                # paste it into the binary image
                img[y:y + h, x:x + w] = results_reshaped

            tif_writer.write(img, contiguous=True)


def erode(binary_input_filepath, output_filepath):
    """
    erode all the frames of a stack file
    Paramereters:
    -------------
    input_filepath, str
    Binary file
    output_filepath, str
    """
    with tiff.TiffFile(binary_input_filepath) as tif, tiff.TiffWriter(output_filepath,
                                                                                       bigtiff=True) as tif_writer:
        for i, page in enumerate(tif.pages):
            img = page.asarray()
            eroded_img = binary_erosion(img)
            # convery to image with values form 0 to 255
            eroded_img = eroded_img.astype(np.uint8)  # convert to an unsigned byte
            eroded_img *= 255
            tif_writer.write(eroded_img, contiguous=True)


def make_hyperstack_from_ometif(input_path, output_filepath, shape, dtype, imagej=True, metadata={'axes': 'TZYX'}):
    """
    Creates a hyperstack from ome.tiff files path
    Parameters:
    --------------
    input_path,
    output_filepath,
    shape, tuple
    Dimensions of the stack. Prefered format TZYX. Example: (100,30,600,600)
    dtype, str
    data type. Example: 'uint16'
    imagej=True,
    metadata, dict
    Any metadata that has to be in the hyperstack
    """

    # create the hyperstack
    hyperstack = tiff.memmap(
        output_filepath,
        shape=shape,
        dtype=dtype,
        imagej=True,
        metadata={'axes': 'TZYX'},
    )

    # loop through it to fill it:
    c = 0
    z_index = 0
    t_index = 0
    for file in natsorted(os.listdir(input_path)):
        if file.endswith('ome.tif'):
            # print(os.path.join(path,file))
            with tiff.TiffFile(os.path.join(input_path, file)) as tif:
                for idx, page in enumerate(tif.pages):
                    img = page.asarray()
                    hyperstack[t_index, z_index] = img
                    hyperstack.flush()
                    c = c + 1
                    z_index = z_index + 1
                    # if z index is equal to the planes per volume (shape[1]), reset z and start new t_index
                    if z_index == shape[1]:
                        z_index = 0
                        t_index = t_index + 1


####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:


def extract_frames(input_image, output_folder, frames_list):
    """
    Extract the frames from an stack (image) and save them as single images given a list.
    #TODO name of the output file has the basename of the original image

    Parameters:
    -----------
    input_image: str,
        Path to the input_image (stack)
    output_folder: str
        Path to the folder where the images will be saved (It will be created if it does not exist)
    frames_list: list, numpy array or int
        list of integers, frames that will be extracted
    """
    # create output_folder if it does not exist
    if not os.path.exists(output_folder):
        print('making ', output_folder, ' directory')
        os.makedirs(output_folder)
    else:
        print(output_folder, 'already exists')

    with tiff.TiffFile(input_image) as tif:
        # iterate over the frames in the list
        # for i, page in enumerate(tif.pages[frames_list]):
        for frame in frames_list:
            img = tif.pages[frame].asarray()
            # print(os.path.join(output_folder,'img'+str(i)+'.'+str(file_format)))
            tiff.imwrite(os.path.join(output_folder, 'img' + str(frame) + '.tif'), img)


def add_zeros_to_filename(path, len_max_number=6):
    """
    Change the filename of images inside the path from img235.png to img00235.png depending on len_max_number
    It has a sister function: add_zeros_to_csv
    # TODO: make it less specific, so it does not required the 'img' string
    Parameters:
    -----------
    path: str,
        Path to the directory with the images
    len_max_number: int
        number of digits the number should have, default is 6
    """
    # creates numberic regular expression
    regex_num = re.compile(r'\d+')

    files = os.listdir(path)
    # this could be improved with a regular expression to catch the numbers.
    for filename in files:
        if 'img' not in filename: continue
        # print(filename)
        new_filename = filename

        number = regex_num.search(filename).group(0)

        # get the file exntesion without the do (e.g. 'png')
        file_extension = re.split('\.', filename)[-1]

        # while the number is smaller than the len_max_number
        while len(number) < len_max_number:
            # print(re.split('img', filename)[1])
            number = '0' + number
        # new filename= img string+number+ dot + extension
        new_filename = 'img' + number + '.' + file_extension
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))


def images2stack_RAM(path, output_filename):
    """
    Convert the images in one folder into a stack, keeping their filenames in the metadata.
    It is not an object to it needs to allocate all the memory for the stack
    Parameters:
    -------------
    path: str, path to the input_folder
    output_filename: str, name of the output stack
    """
    # Function in construction
    # files = os.listdir(path)#tifffile.natural_sorted(output_path)
    # image = tifffile.imread(os.path.join(path,files))


def images2stack(path, output_filename):
    """
    Convert the images in one folder into a stack, keeping their filenames
    Images have to be .tif, can't be PNG.
    Parameters:
    -------------
    path: str, path to the input_folder
    output_filename: str, name of the output stack
    """
    files = natsorted(os.listdir(path))
    metadata = {'Info': '\n'.join(files)}

    with tiff.TiffWriter(output_filename, imagej=True) as tif:
        for filename in files:
            # if the images are tiff
            print(filename)
            #skip files that start with '.'
            if filename.startswith('.'): continue

            if filename.endswith(('.tif', '.tiff')):
                image = tiff.imread(os.path.join(path, filename))
            # if the images are png
            if filename.endswith('.png'):
                image = cv2.imread(os.path.join(path, filename))

            tif.write(image, contiguous=True, photometric='minisblack', metadata=metadata)


def rgbimages2stacks(list_of_images, output_path):
    """
    Converts a list of PNG images into three stacks, one for each channel. It assumes the default is BGR.
    Parameters:
    -------------
    list_of_images: list, list of png images
    output_path: str, name of the directory where the stack will be written
    """
    r_output = os.path.join(output_path, 'Stack_red.tiff')
    g_output = os.path.join(output_path, 'Stack_green.tiff')
    b_output = os.path.join(output_path, 'Stack_blue.tiff')

    with tiff.TiffWriter(r_output, imagej=True) as tif_r, tiff.TiffWriter(g_output,
                                                                          imagej=True) as tif_g, tiff.TiffWriter(
        b_output, imagej=True) as tif_b:
        for image_path in list_of_images:
            image = cv2.imread(image_path)
            # somehow the default from PreSens is BGR so we need to convert it
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            r_image = rgb_image[:, :, 0]
            g_image = rgb_image[:, :, 1]
            b_image = rgb_image[:, :, 2]

            tif_r.write(r_image, contiguous=True, photometric='minisblack')
            tif_g.write(g_image, contiguous=True, photometric='minisblack')
            tif_b.write(b_image, contiguous=True, photometric='minisblack')


def stack2images(input_filename, output_path):
    """
    Convert a stack into a folder with all the images, saving each image with its original name
    Parameters:
    -------------
    input_filename:str, name of the input stack
    output_path: str, path to the directory where it will be saved

    If it does not work check here: https://forum.image.sc/t/keep-image-description-metadata-in-a-stack-after-modifying-it/50625/4
    """
    try:
        os.mkdir(output_path)  # creates the subdirectory where it should be stored
    except:
        print('Output Directory already exists, might overwrite')
    with tiff.TiffFile(input_filename) as tif:
        # try to get metadata from imageJ
        try:
            files = tif.imagej_metadata['Info'].split('\n')
            metadata = True
        except:
            metadata = False
        metadata = False
        for idx, page in enumerate(tif.pages):
            img = page.asarray()
            # if metadata is True: name according to metadata, else name image1.tif,etc.
            if metadata == True:
                filename = files[idx]
            else:
                filename = 'image' + str(idx) + '.tif'
            tiff.imwrite(os.path.join(output_path, filename), img)


def tiff2png_list(tiff_img_list):
    """
    Convert images in the list from tiff to png. There is compression happening.
    :param tiff_img_list:
    :return:
    """
    for img_path in tiff_img_list:
        img = tiff.imread(img_path)
        new_filename = os.path.splitext(img_path)[0]+".png"
        io.imsave(new_filename, img)
    return None

def contours_length(img):
    """
    Return length and perimeter of the contours in an image.
    Length is assumed to be half of the perimeter. Only valid for elongated contours.

    Parameters
    -------------
    img: numpy_array,
    name of the input stack

    Returns
    ------------
    contours_len: numpy array,
    contains the length of the contours
    contours_peri: numpy array,
    contains the perimeter of the contours

    """
    cnts, hierarchy = cv2_find_contours_compatibility_mode(img)
    contours_peri = []

    for cnt in cnts:
        contours_peri.append(cv2.arcLength(cnt, True))

    contours_peri = np.array(contours_peri)
    contours_len = contours_peri / 2

    return contours_len, contours_peri


def z_projection(img, projection_type, axis=0):
    """
    Careful: mean projection might change dtype to float32
    Parameters:
    ------------
    img, 3-D numpy array
    Image stack that needs to be projected across the z dimension

    projection type, str
    String containing one of the 4 projections options: max, min, mean or median.

    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.
    Returns:
    ------------
    projected_img, numpy array
    Contains the projected img
    """
    if projection_type == 'max':
        projected_img = np.max(img, axis=axis)
    if projection_type == 'min':
        projected_img = np.min(img, axis=axis)
    if projection_type == 'mean':
        projected_img = np.mean(img, axis=axis)
    if projection_type == 'median':
        projected_img = np.median(img, axis=axis)

    return projected_img


def stack_z_projection(input_path, output_path, projection_type, dtype='uint16', axis=0):
    """

    Parameters:
    :param input_path:
    :param output_path:
    :param projection_type:
    :param dtype:
    :param axis:
    :return:
    """
    try:
        reader_obj = MicroscopeDataReader(input_path, as_raw_tiff=True, raw_tiff_num_slices=1)
    except TypeError:
        reader_obj = MicroscopeDataReader(input_path, as_raw_tiff=False)
    stack = da.squeeze(reader_obj.dask_array)
    projected_img = z_projection(stack, projection_type, axis)
    projected_img = projected_img.astype(dtype)
    tiff.imwrite(output_path, projected_img)
    return None


def z_projection_parser(hyperstack_filepath, output_filepath, projection_type, axis):
    """
    parser do run the z_projection function

    Warning: Write permission is required for this function

    Parameters:
    ----------
    img_path, str
    output_path, str
    projection_type, str
    axis : None or int or tuple of ints, optional
    Axis or axes along which to operate.  By default, flattened input is
    used.
    Returns:
    ----------
    Writes the projection. Function itself returns None
    """
    # load hyperstack in memory map
    hyperstack = tiff.memmap(hyperstack_filepath, dtype='uint16')
    # crate writer object
    with tiff.TiffWriter(output_filepath, bigtiff=True) as tif_writer:
        # iterate for each volume of the hyperstack
        for volume in hyperstack:
            projected_img = z_projection(volume, projection_type, axis)
            tif_writer.write(projected_img, contiguous=True)


def draw_some_contours(img, contour_size, tolerance, inner_contour_area_to_fill):
    """
    Return img with drawn contours based on size, filling contours below inner_contour_area_to_fill
    Parameters:
    -----------
    img, numpy array
    image from where the contours will be taken
    contour_size, float
        expected area of the contour to be extracted
    tolerance, float
        tolerance around which other contours will be accepted. e.g. contour_size 100 and tolerance 0.1 will include contours from 90 to 110.
    inner_contour_area_to_fill, float
        area of inner contours that will be filled

    Returns:
    -----------
    img_contours, numpy array
        image with drawn contours
    """

    # convert image dtype if not uint8
    # image has to be transformed to uint8 for the findContours
    img = img.astype(np.uint8)
    # get contours
    cnts, hierarchy = cv2_find_contours_compatibility_mode(img)

    # good contours index
    cnts_idx = []  # np.array([])
    # create empty image
    img_contours = np.zeros(img.shape)

    for cnt_idx, cnt in enumerate(cnts):
        cnt_area = cv2.contourArea(cnt)
        # if the contour area is between the expected values with tolerance, save contour in cnts_idx and draw it
        if (contour_size * (1 - tolerance) < cnt_area < contour_size * (1 + tolerance)):
            cnts_idx.append(np.array(cnt_idx))
            cv2.drawContours(img_contours, cnts, cnt_idx, color=255, thickness=-1, hierarchy=hierarchy, maxLevel=1)

        # if the current cnt_idx has as a parent a contour in good countours (cnts_idx)
        if hierarchy[0][cnt_idx][3] in cnts_idx:
            # (and) if it is smaller than inner contour, draw it
            if cnt_area < inner_contour_area_to_fill:
                # print(cv2.contourArea(contours[j]))
                cv2.drawContours(img_contours, cnts, cnt_idx, color=255, thickness=-1)

    # convert the resulting image into a 8 binary numpy array
    img_contours = np.array(img_contours, dtype=np.uint8)

    return img_contours


def extract_contours_with_children(img):
    """
    Find the contours that have a children in the given image and return them as list

    Parameters:
    -----------
    img, np.array
    Returns:
    -----------
    contours that have a children, list of contours with children
    Important does not return the children, only the contour that has children.
    """

    #important, findCountour() has different outputs depending on CV version! _, cnts, hierarchy or cnts, hierarchy
    cnts, hierarchy = cv2_find_contours_compatibility_mode(img)
    contours_with_children = []
    for cnt_idx, cnt in enumerate(cnts):
        # draw contours with children: last column in the array is -1 if an external contour, column 2 is different than -1 meaning it has children
        if hierarchy[0][cnt_idx][3] == -1 and hierarchy[0][cnt_idx][2] != -1:
            contours_with_children.append(cnt)
            # not needed, do it outside this function
            # get coords of boundingRect
            # x,y,w,h = cv2.boundingRect(cnt)
            # make the crop
            # cnt_img=img[y:y+h,x:x+w]
    return contours_with_children


def cv2_find_contours_compatibility_mode(img):
    output = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(output) == 3:
        # Older versions of cv2 return 3 values
        _, cnts, hierarchy = output
    elif len(output) == 2:
        cnts, hierarchy = output
    else:
        raise ValueError("The output of cv2.findContours is not the expected length "
                         f"(2 or 3), but instead: {len(output)}; Full output: {output}")
    return cnts, hierarchy


def crop_image_from_contour(img, contour):
    """crop the image based on a contour
    Parameters:
    -----------
    img, img
    contour, contour object
    returns:
    cnt_img
    """
    x, y, w, h = cv2.boundingRect(contour)
    # make the crop
    cnt_img = img[y:y + h, x:x + w]

    return cnt_img


def stack_extract_and_save_contours_with_children(binary_input_filepath, raw_input_filepath, output_folder, crop=False,
                                                  subsample=250):
    """

    :param binary_input_filepath: image from where the contours will be extracted
    :param raw_input_filepath: image that will be cropped based on the binary contours (can be the same as binary_input_filepath)
    :param output_folder: folder where to save the images
    :param subsample: subsample rate to not to save all the images.
    :return: None
    """

    file_basename = os.path.splitext(os.path.basename(raw_input_filepath))[0]
    print(file_basename)

    with tiff.TiffFile(binary_input_filepath) as tif_binary, tiff.TiffFile(raw_input_filepath) as tif_raw:
        for i, page in enumerate(tif_binary.pages):
            if i % subsample != 0: continue
            img = page.asarray()

            # extract contours with children
            contours_with_children = extract_contours_with_children(img)

            for cnt_idx, cnt in enumerate(contours_with_children):
                # get the raw image
                raw_img = tif_raw.pages[i].asarray()
                # if crop == True make the crop on the raw image
                if crop==True:
                    raw_img = crop_image_from_contour(raw_img, cnt)
                # add zeros to str(i) so that it can be more easily read by natsort algorithms
                while (len(str(i)))<6:
                    i= "0"+str(i)
                output_filepath = os.path.join(output_folder,
                                               file_basename + '_frame_' + i + '_cnt_' + str(cnt_idx) + '.tiff')
                print(output_filepath)
                tiff.imwrite(output_filepath, raw_img)
    return None


def find_specific_contours_with_specific_children(img, external_contour_area, internal_contour_area):
    """
    Inspired by imutils.src.imfunctions.extract_contours_with_children()
    :return:
    """

    # important, findCountour() has different outputs depending on CV version! _, cnts, hierarchy or cnts, hierarchy
    cnts, hierarchy = cv2_find_contours_compatibility_mode(img)
    specific_contours_with_specific_children = []

    for cnt_idx, cnt in enumerate(cnts):
        # draw contours with children: last column in the array is -1 if an external contour, column 2 is different than -1 meaning it has children
        if hierarchy[0][cnt_idx][3] == -1 and hierarchy[0][cnt_idx][2] != -1:
            area = cv2.contourArea(cnt)

            # check if the contours with children have an area between the external_cnt_area
            if external_contour_area[0] < area < external_contour_area[1]:
                # find the rows of an array where the 3rd column is equal to cnt_idx
                new_array = np.where(hierarchy[0][:, 3] == cnt_idx)
                for child_cnt_idx in new_array:
                    child_cnt_area = cv2.contourArea(cnts[child_cnt_idx[0]])

                    if internal_contour_area[0] < child_cnt_area < internal_contour_area[1]:
                        specific_contours_with_specific_children.append(cnt)

    return specific_contours_with_specific_children

def stack_self_touch(binary_path, external_contour_area, internal_contour_area):
    """
    This function was written to detect self touch in worm, which will form a contour with children.
    :param binary_path: path to the binary stack image
    :param external_contour_area: area range of the external contour (min and max), e.g. [7000, 20000]
    :param internal_contour_area: area range the internal contour (min and max), e.g. [100, 2000]
    """
    df = pd.DataFrame()

    with tiff.TiffFile(binary_path) as tif_binary:
        for i, page in enumerate(tif_binary.pages):
            img = page.asarray()
            contours_with_children = find_specific_contours_with_specific_children(img, external_contour_area, internal_contour_area)
            if contours_with_children:
                #print('self touch')
                df = df.append({'self_touch': 1}, ignore_index=True)
            else:
                #print('no self touch')
                df = df.append({'self_touch': 0}, ignore_index=True)
    return df

def measure_mask(img, threshold):

    """
    Returns the number of values above a threshold and the sum of its values
    """
    roi = img >= threshold
    n_values = np.sum(roi)
    intensity = np.sum(img[roi])

    return n_values, intensity

def distance_to_image_center(image_shape, point):
    """
    Calculate the distance (in px) from the center of the image to the point coords
    :param image_shape: tuple, shape of the image
    :param point: tuple, point coordinates
    :return: np.array containing the x,y distance
    """
    center = np.asarray(image_shape)/2
    result = np.asarray(point) - center
    return result


#FUNCTIONS ADDED BY LAXMI FOR ZIM06 FLUORESCENCE PIPELINE and can be called from imutils parser.
#FUNCTIONS ADDED BY LAXMI FOR ZIM06 FLUORESCENCE PIPELINE


def calculate_roi_bounds(centroid, roi_half_size, img_shape):
    """
    Calculates ROI boundaries based on centroid and half-size,
    clamping the coordinates to stay within image dimensions.
    Ensures the ROI has a minimum size of 1x1.
    Includes basic check for valid centroid input.
    """
    height, width = img_shape
    # Ensure centroid components are valid numbers before proceeding
    if centroid is None or not all(np.isfinite(c) for c in centroid):
        # Raise error here as it's a logic issue if centroid is invalid
        raise ValueError(f"Invalid centroid coordinates for ROI calculation: {centroid}")
    
    cy, cx = centroid # Expects (row, col) format from center_of_mass
    cy_int, cx_int = int(round(cy)), int(round(cx))

    # Calculate initial bounds
    y_start = max(0, cy_int - roi_half_size)
    y_end = min(height, cy_int + roi_half_size + 1) # +1 for exclusive upper bound
    x_start = max(0, cx_int - roi_half_size)
    x_end = min(width, cx_int + roi_half_size + 1) # +1 for exclusive upper bound

    # Ensure minimum size of 1 pixel if bounds collapsed
    if y_start == y_end:
        if y_end < height: y_end += 1
        elif y_start > 0: y_start -= 1
    if x_start == x_end:
        if x_end < width: x_end += 1
        elif x_start > 0: x_start -= 1

    # Final clamp
    y_start = max(0, y_start)
    y_end = min(height, y_end)
    x_start = max(0, x_start)
    x_end = min(width, x_end)

    # Check for zero size ROI after all adjustments
    if y_start >= y_end or x_start >= x_end:
         warnings.warn(f"Resulting ROI may have zero size ({y_start}:{y_end}, {x_start}:{x_end}) for centroid {centroid}. Adjusting to min 1x1.")
         y_end = max(y_end, y_start + 1)
         x_end = max(x_end, x_start + 1)
         y_end = min(height, y_end)
         x_end = min(width, x_end)
         if y_start >= y_end or x_start >= x_end:
              raise ValueError(f"Unable to create valid 1x1 ROI bounds for centroid {centroid}")

    return y_start, y_end, x_start, x_end

def process_frame_globally(img, threshold, max_value_out, min_component_size, output_dtype, roi_half_size):
    """
    Processes a single frame globally: thresholds, finds connected components,
    filters by size, and calculates centroid and ROI for the largest components.
    Returns the resulting mask, centroid, ROI bounds, and a success flag.
    Includes basic error handling for centroid/ROI calculation.
    """
    if img is None or img.size == 0:
        warnings.warn("process_frame_globally received empty image.")
        return np.array([], dtype=bool), None, None, False

    img_shape = img.shape
    current_mask = np.zeros(img_shape, dtype=bool) # Default to empty
    current_centroid = None
    current_roi_bounds = None
    found_components = False
    
    # 1. Thresholding
    ret, initial_binary_img_float = cv2.threshold(img, threshold, float(max_value_out), cv2.THRESH_BINARY)
    initial_mask = initial_binary_img_float > 0
    # 2. Labeling
    labeled_mask, num_features = label(initial_mask, structure=np.ones((3, 3)))
    # 3. Size Filtering
    if num_features > 0:
        component_sizes = ndi_sum(initial_mask, labels=labeled_mask, index=np.arange(1, num_features + 1))
        large_enough_indices = np.where(component_sizes > min_component_size)[0]
        if large_enough_indices.size > 0:
            large_enough_labels = large_enough_indices + 1
            current_mask = np.isin(labeled_mask, large_enough_labels)
            if np.any(current_mask): found_components = True
            else: 
                current_mask = np.zeros(img_shape, dtype=bool)

    # 4. Centroid and ROI Calculation (only if components were found and kept)
    if found_components:
        try:
            current_centroid = center_of_mass(current_mask)
            current_roi_bounds = calculate_roi_bounds(current_centroid, roi_half_size, img_shape)
        except Exception as e:
             warnings.warn(f"Global search: Centroid/ROI calculation failed: {e}. Treating as no components found.")
             current_mask = np.zeros(img_shape, dtype=bool)
             current_centroid, current_roi_bounds = None, None
             found_components = False

    return current_mask, current_centroid, current_roi_bounds, found_components

def stack_make_binary_tracked(stack_input_filepath: str, stack_output_filepath: str, threshold: float,
                        max_value: float = 255.0, min_component_size: int = 3, roi_half_size: int = 10):
    """
    Writes a binary stack (uint8) using a tracked ROI.
    Attempts global search until tracking is initialized.
    Calculates and prints min/max mask pixel counts and frames without masks at the end.
    """
    print(f"--- Starting Tracked Binary Filtering (Output: uint8) ---")
    print(f"Input: {stack_input_filepath}, Output: {stack_output_filepath}")
    print(f"Threshold: {threshold}, Max Value: {max_value}, Min Component Size > {min_component_size}, ROI half size: {roi_half_size}")

    output_dtype = np.uint8
    # Calculate output value, clamped to uint8 range
    max_value_out = output_dtype(min(np.iinfo(output_dtype).max, max(0, round(max_value))))
    print(f"Using output 'on' value: {max_value_out}")
    
    reader_obj = MicroscopeDataReader(stack_input_filepath, as_raw_tiff=True, raw_tiff_num_slices=1)
    tif_stack = da.squeeze(reader_obj.dask_array)
    # --- Determine Image Shape from Dask Array ---
    try:
        if len(tif_stack.shape) >= 2:
            img_shape = tif_stack.shape[-2:] # Assume YX are the last two dimensions
        else:
            raise ValueError(f"Dask array has unexpected shape: {tif_stack.shape}")
        print(f"Determined Image Shape (YX): {img_shape}")
    except Exception as e:
        print(f"ERROR: Could not determine image shape from Dask array: {e}")
        return # Exit if shape cannot be determined
    
    # --- Initialize Tracking State ---
    # These store the state of the LAST SUCCESSFUL frame
    last_successful_mask = None
    last_successful_centroid = None
    last_successful_roi_bounds = None
    
    # --- Initialize Min/Max Pixel Count Tracking ---
    min_mask_pixels = float('inf')
    max_mask_pixels = 0
    masks_found_count = 0 # To check if any masks were found at all
    frames_without_mask = [] # <-- Initialize list to store frames without masks
    # --- NEW LOGIC FOR CONSECUTIVE EMPTY FRAMES START ---
    consecutive_empty_frames = 0
    CONSECUTIVE_EMPTY_THRESHOLD = 5 
    # --- NEW LOGIC FOR CONSECUTIVE EMPTY FRAMES END ---
    num_frames_processed = 0

    # --- Process Frames and Write Output ---
    try:
        with tiff.TiffWriter(stack_output_filepath, bigtiff=True) as tif_writer:
            # Iterating over a Dask array triggers computation chunk by chunk (or frame by frame if chunked that way)
            for i, img_frame in enumerate(tif_stack):
                num_frames_processed += 1
                # img_frame is now computed (a NumPy array)
                img = np.array(img_frame)
                if img.shape[-2:] != img_shape:
                     warnings.warn(f"Frame {i} shape {img.shape[-2:]} differs from expected {img_shape}. Writing empty frame.")
                     tif_writer.write(np.zeros(img_shape, dtype=output_dtype), contiguous=True, photometric='minisblack')
                     frames_without_mask.append(i) # <-- Add frame index if shape mismatch
                     consecutive_empty_frames += 1 
                     continue

                current_mask = np.zeros(img_shape, dtype=bool)
                final_frame = np.zeros(img_shape, dtype=output_dtype)
                found_in_frame = False
                
                # --- NEW LOGIC FOR CONSECUTIVE EMPTY FRAMES START ---
                if consecutive_empty_frames >= CONSECUTIVE_EMPTY_THRESHOLD:
                    warnings.warn(f"Frame {i}: {consecutive_empty_frames} consecutive empty frames. Forcing global search and resetting tracking state.")
                    last_successful_roi_bounds = None 
                    last_successful_centroid = None
                    last_successful_mask = None # Ensure all tracking state is reset
                    consecutive_empty_frames = 0 # Reset counter after forcing
                # --- NEW LOGIC FOR CONSECUTIVE EMPTY FRAMES END ---

                if last_successful_roi_bounds is None:
                    # --- Tracking NOT Initialized: Attempt Global Search ---
                    if i == 0: print(f"Frame 0: Attempting initial global search...")
                    else: warnings.warn(f"Frame {i}: Tracking not initialized. Attempting global search...")

                    mask_found, centroid, roi_bounds, found = process_frame_globally(
                        img, threshold, max_value_out, min_component_size, output_dtype, roi_half_size
                    )
                    if found:
                         print(f"Frame {i}: Global search SUCCEEDED. Initializing tracking.")
                         current_mask = mask_found
                         last_successful_mask = current_mask
                         last_successful_centroid = centroid
                         last_successful_roi_bounds = roi_bounds
                         found_in_frame = True
                    # else: Global search failed, warning printed in helper

                else:
                    # --- Tracking IS Initialized: Attempt ROI-based Tracking ---
                    y_s, y_e, x_s, x_e = last_successful_roi_bounds
                    if not (y_s < y_e and x_s < x_e):
                         warnings.warn(f"Frame {i}: Invalid previous ROI bounds {last_successful_roi_bounds}. Saving empty frame. Reusing previous ROI.")
                    else:
                        img_roi_only = np.zeros_like(img)
                        img_roi_only[y_s:y_e, x_s:x_e] = img[y_s:y_e, x_s:x_e]
                        ret, roi_binary = cv2.threshold(img_roi_only, threshold, float(max_value_out), cv2.THRESH_BINARY)
                        roi_initial_mask = roi_binary > 0
                        labeled_roi, num_features = label(roi_initial_mask, structure=np.ones((3, 3)))

                        mask_in_roi = np.zeros_like(img, dtype=bool)
                        found_in_roi = False
                        if num_features > 0:
                            sizes = ndi_sum(roi_initial_mask, labels=labeled_roi, index=np.arange(1, num_features + 1))
                            indices = np.where(sizes > min_component_size)[0]
                            if indices.size > 0:
                                labels = indices + 1
                                mask_in_roi = np.isin(labeled_roi, labels)
                                if np.any(mask_in_roi): found_in_roi = True

                        if found_in_roi:
                             try: # Keep try-except for math/logic errors here
                                centroid = center_of_mass(mask_in_roi)
                                roi_bounds = calculate_roi_bounds(centroid, roi_half_size, img_shape)
                                current_mask = mask_in_roi
                                last_successful_mask = current_mask
                                last_successful_centroid = centroid
                                last_successful_roi_bounds = roi_bounds
                                found_in_frame = True
                             except Exception as e:
                                 warnings.warn(f"Frame {i}: ROI centroid/bounds calculation failed: {e}. Saving empty frame. Reusing previous ROI.")
                        # else: # Tracking failed within ROI, warning printed in helper
                        
                # --- Update Min/Max Pixel Counts & Record Frames Without Mask ---
                if found_in_frame:
                     num_pixels = np.sum(current_mask) # Count True values in boolean mask
                     min_mask_pixels = min(min_mask_pixels, num_pixels)
                     max_mask_pixels = max(max_mask_pixels, num_pixels)
                     masks_found_count += 1
                     consecutive_empty_frames = 0
                     # print(f"Frame {i}: Mask found with {num_pixels} pixels.") # Optional debug print
                else:
                     # Add frame index if no mask was successfully generated in this frame
                     frames_without_mask.append(i) # <-- Add frame index on failure
                     consecutive_empty_frames += 1

                # --- Generate & Write Final Frame ---
                final_frame = np.zeros(img_shape, dtype=output_dtype) # Initialize clean
                if found_in_frame:
                     final_frame = np.where(current_mask, max_value_out, output_dtype(0))
                tif_writer.write(final_frame.astype(output_dtype), contiguous=True, photometric='minisblack')

        print(f"--- Finished writing tracked binary stack (uint8) to {stack_output_filepath} ---")

    # Keep this outer try-except for errors during the main loop/writing
    except Exception as e:
        print(f"\n--- ERROR during processing/writing loop ---")
        print(f"{type(e).__name__}: {e}")
        print("Traceback:")
        traceback.print_exc()
        print(f"Output file '{stack_output_filepath}' may be incomplete or corrupted.")
        # --- Print Final Statistics Even if Loop Exited Early ---
        print("\n--- Mask Pixel Statistics (Partial results due to error) ---")
        if masks_found_count > 0:
            print(f"Minimum mask pixels (across {masks_found_count} frames with masks): {min_mask_pixels}")
            print(f"Maximum mask pixels (across {masks_found_count} frames with masks): {max_mask_pixels}")
        else:
            print("No masks were successfully generated.")
        # Print frames without masks (even if partial)
        if frames_without_mask:
            print(f"Frames where no mask was generated: {frames_without_mask}")
        else:
             if masks_found_count > 0: # Only print if some masks *were* found
                  print("Mask generated successfully for all processed frames.")
        return # Exit after reporting partial stats if error occurred

    # --- Print Final Statistics (if loop completed successfully) ---
    print("\n--- Mask Pixel Statistics ---")
    if masks_found_count > 0:
        print(f"Minimum mask pixels (across {masks_found_count} frames with masks): {min_mask_pixels}")
        print(f"Maximum mask pixels (across {masks_found_count} frames with masks): {max_mask_pixels}")
    else:
        print("No masks were successfully generated in any frame.")

    # Print frames without masks
    if frames_without_mask:
        print(f"Frames where no mask was generated: {frames_without_mask}")
    else:
        if masks_found_count > 0: # Check if *any* processing happened
             print("Mask generated successfully for all processed frames.")

def calculate_measurements(input_img_path, background_img_path, output_csv_path, threshold, top_n):
    """
    Reads a main image stack and a background stack, calculates various
    intensity and pixel count metrics per frame, and saves them to a CSV file.
    Top N calculations are only performed if the frame has more pixels than top_n.
    Background intensity at top_n locations is calculated if
    the main image and background image frames have matching dimensions.
    Args:
        input_img_path (str): Path to the main input TIFF stack (masked image).
        background_img_path (str): Path to the background TIFF stack.
        output_csv_path (str): Path where the output CSV file will be saved.
        threshold (float): Intensity threshold for counting pixels in the main image.
        top_n (int): The number of top intensity pixels to consider for metrics.
    Returns:
        None: Writes the results directly to the output CSV file.
    """
    
    frame_number_list = []
    n_values_list = []                   # Count of pixels above threshold in main image
    total_intensity_list = []            # Sum of all pixel intensities in main image frame
    top_total_intensity_list = []        # Sum of top_n brightest pixel intensities in main image frame
    background_total_intensity_list = [] # Sum of all pixel intensities in background image frame
    top_background_intensity_list = []   # Sum of background pixel intensities at main image's top_n locations
    
    background_stack = tiff.imread(background_img_path)
    print(f"Background stack loaded with shape: {background_stack.shape}")
    # Basic check for unexpected dimensions (can be refined based on expected inputs)
    if len(background_stack.shape) < 2:
        print(f"ERROR: Background image has unexpected dimensions: {background_stack.shape}")
        sys.exit(1)
    elif len(background_stack.shape) == 2:
        warnings.warn("Background image appears to be 2D. Assuming it applies to all frames.")
        
        # --- Process Main Image Stack Frame by Frame ---
    frame_count = 0
    try:
        with tiff.TiffFile(input_img_path) as tif:
            num_main_frames = len(tif.pages)
            print(f"Processing {num_main_frames} frames from main image...")

            # --- Sanity Check: Compare Frame Counts (if background is a stack) ---
            process_frames = num_main_frames
            if len(background_stack.shape) >= 3: # Only check if background is a stack
                if num_main_frames != len(background_stack):
                    warnings.warn(f"Warning: Main image ({num_main_frames} frames) and background image ({len(background_stack)} frames) have different lengths!")
                    process_frames = min(num_main_frames, len(background_stack))
                    print(f"Processing only the first {process_frames} frames due to length mismatch.")
            elif len(background_stack.shape) == 2 and num_main_frames > 1:
                 # warnings.warn("Main image is a stack, but background is 2D. Applying same background to all frames.") # Warning moved earlier
                 pass # Allow processing
            elif len(background_stack.shape) == 2 and num_main_frames == 1:
                 pass # Both are single frames, proceed
            else: # Mismatch like 2D main and 3D background
                 print(f"ERROR: Incompatible dimensions between main image ({num_main_frames} frames) and background (shape {background_stack.shape}).")
                 sys.exit(1)


            for i in range(process_frames): # Iterate up to the determined frame count
                page = tif.pages[i]
                img = page.asarray()

                # Get corresponding background frame or the single background image
                if len(background_stack.shape) >= 3:
                    bg_img = background_stack[i]
                else: # Background is 2D
                    bg_img = background_stack

                # --- Calculate metrics for the current frame ---

                # 1. Count of pixels above threshold in main image
                pixels_above_threshold = img[img >= threshold]
                count_of_pixels_above_threshold = pixels_above_threshold.size
                n_values_list.append(count_of_pixels_above_threshold)

                # 2. Total intensity of the main image frame
                total_intensity_list.append(np.sum(img))

                # Initialize top_n related metrics to NaN for the current frame
                current_top_total_main_img = np.nan
                current_top_background_intensity = np.nan

                # 3. Top N calculations (if main image frame is not empty and has enough pixels)
                if img.size > 0:
                    flat_img = img.ravel() # Flatten for easier processing
                    if flat_img.size > top_n:
                        # Find indices of the top_n brightest pixels in the main image
                        # np.argpartition is efficient: it puts the k-th smallest elements in their sorted
                        # positions and all other elements are partitioned around them.
                        # To get largest, we partition around -top_n and take elements from -top_n to end.
                        top_n_indices_flat = np.argpartition(flat_img, -top_n)[-top_n:]

                        # Sum of intensities of these top_n pixels in the main image
                        current_top_total_main_img = np.sum(flat_img[top_n_indices_flat])

                        # Sum of intensities of background pixels at the *same locations*
                        flat_bg_img = bg_img.ravel()
                        if flat_img.shape == flat_bg_img.shape: # Critical check for matching dimensions
                            current_top_background_intensity = np.sum(flat_bg_img[top_n_indices_flat])
                        else:
                            warnings.warn(
                                f"Frame {i}: Main image shape {img.shape} and background image shape {bg_img.shape} "
                                f"are different. Skipping 'top_background_intensity' calculation for this frame."
                            )
                            # current_top_background_intensity remains NaN
                    else:
                        # Not enough pixels in the main image for top_n calculation
                        print(
                            f"Frame {i}: Skipping top_n related calculations for main image (found {flat_img.size} "
                            f"pixels, need > {top_n}). 'top_background_intensity' also skipped."
                        )
                        # current_top_total_main_img and current_top_background_intensity remain NaN
                else:
                     # Main image frame is empty
                     print(
                         f"Frame {i}: Skipping top_n related calculations (main image frame is empty). "
                         f"'top_background_intensity' also skipped."
                     )
                     # current_top_total_main_img and current_top_background_intensity remain NaN

                top_total_intensity_list.append(current_top_total_main_img)
                top_background_intensity_list.append(current_top_background_intensity)

                # 4. Total intensity of the corresponding background image frame
                background_total_intensity_list.append(np.sum(bg_img))

                frame_number_list.append(i)
                frame_count += 1

        print(f"Finished processing {frame_count} frames.")

    except FileNotFoundError:
        print(f"ERROR: Main input image file not found: {input_img_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed during processing of main image {input_img_path}: {e}")
        traceback.print_exc()
        sys.exit(1)


    # --- Create DataFrame and Calculate Derived Mean Metrics ---
    print("Creating DataFrame...")
    if frame_count == 0:
        print("WARNING: No frames were processed. Output CSV will be empty or may fail.")
        # Define columns for empty DataFrame to prevent error on save
        df_columns = [
            'frame', 'total_intensity', 'n_values', 'top_total_intensity',
            'background_total_intensity', 'top_background_intensity',
            'mean_intensity', 'mean_top_intensity',
            'mean_background_intensity', 'mean_top_background_intensity',
            'mean_final_intensity'
        ]
        df = pd.DataFrame(columns=df_columns)
    else:
        df = pd.DataFrame()
        df['frame'] = frame_number_list
        df['total_intensity'] = total_intensity_list                 # Main img: sum of all pixels
        df['n_values'] = n_values_list                               # Main img: count of pixels >= threshold
        df['top_total_intensity'] = top_total_intensity_list         # Main img: sum of top_n pixels
        df['background_total_intensity'] = background_total_intensity_list # Bg img: sum of all pixels
        df['top_background_intensity'] = top_background_intensity_list   # Bg img: sum of pixels at main img's top_n locations

        # Calculate mean intensity for pixels above threshold in main image
        with warnings.catch_warnings(): # Suppress RuntimeWarning for division by zero if n_values is 0
             warnings.simplefilter("ignore", category=RuntimeWarning)
             df['mean_intensity'] = np.divide(
                 df['total_intensity'], df['n_values'],
                 out=np.full_like(df['total_intensity'], np.nan, dtype=float), # Output NaN if n_values is 0
                 where=df['n_values'] != 0
            )

        # Mean of top_n brightest pixels from main image
        df['mean_top_intensity'] = df['top_total_intensity'] / top_n

        # Mean background intensity (original calculation: total background sum / top_n)
        # This uses the sum of *all* background pixels, scaled by top_n.
        df['mean_background_intensity'] = df['background_total_intensity'] / top_n

        # Mean background intensity from pixels at main image's top_n locations
        df['mean_top_background_intensity'] = df['top_background_intensity'] / top_n

        # Final mean intensity: (mean of top_n main img pixels) - (mean of bg img pixels at top_n locations)
        df['mean_final_intensity'] = df['mean_top_intensity'] - df['mean_top_background_intensity']

    # --- Save DataFrame to CSV ---
    try:
        print(f"Saving measurements to {output_csv_path}...")
        df.to_csv(output_csv_path, index=False, na_rep='NaN') # Save without DataFrame index, represent NaN as 'NaN'
        print('--- Image Measurement Script Finished Successfully ---')
    except Exception as e:
        print(f"ERROR: Failed to save output CSV to {output_csv_path}: {e}")
        sys.exit(1)

def apply_mask(raw_img_path, mask_img_path, background_img_path,
               masked_raw_out_path, masked_bg_out_path):
    """
    Applies a binary mask stack frame-by-frame to both a raw image stack
    and a background image (which can be a stack or a single 2D frame).

    Args:
        raw_img_path (str): Path to the raw input TIFF stack.
        mask_img_path (str): Path to the binary mask TIFF stack.
        background_img_path (str): Path to the background TIFF (stack or 2D).
        masked_raw_out_path (str): Output path for the masked raw image stack.
        masked_bg_out_path (str): Output path for the masked background image stack.

    Returns:
        None: Writes results directly to the output files.
    """
    print("--- Starting Image Masking Script ---")
    print(f"Raw Input: {raw_img_path}")
    print(f"Mask Input: {mask_img_path}")
    print(f"Background Input: {background_img_path}")
    print(f"Masked Raw Output: {masked_raw_out_path}")
    print(f"Masked Background Output: {masked_bg_out_path}")

    # --- Load Background Image ---
    # Load background first to determine if it's 2D or 3D
    try:
        background_data = tiff.imread(background_img_path)
        is_background_stack = len(background_data.shape) >= 3
        print(f"Background loaded. Shape: {background_data.shape}, Is Stack: {is_background_stack}")
    except FileNotFoundError:
        print(f"ERROR: Background image file not found: {background_img_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load background image {background_img_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Open Input Files and Output Writers ---
    try:
        # Use TiffFile for potentially large raw/mask stacks to read page-by-page
        with tiff.TiffFile(mask_img_path) as tif_mask, \
             tiff.TiffFile(raw_img_path) as tif_raw, \
             tiff.TiffWriter(masked_raw_out_path, bigtiff=True) as writer_raw_masked, \
             tiff.TiffWriter(masked_bg_out_path, bigtiff=True) as writer_bg_masked:

            num_mask_frames = len(tif_mask.pages)
            num_raw_frames = len(tif_raw.pages)
            print(f"Mask frames: {num_mask_frames}, Raw frames: {num_raw_frames}")

            # --- Frame Count Sanity Checks ---
            if num_mask_frames != num_raw_frames:
                warnings.warn(f"Warning: Mask ({num_mask_frames}) and Raw ({num_raw_frames}) frame counts differ! Processing minimum.")
                process_frames = min(num_mask_frames, num_raw_frames)
            else:
                process_frames = num_mask_frames

            if is_background_stack and len(background_data) != process_frames:
                 warnings.warn(f"Warning: Background stack length ({len(background_data)}) differs from mask/raw ({process_frames})! Processing minimum.")
                 process_frames = min(process_frames, len(background_data))

            if process_frames == 0:
                 print("Warning: No frames to process based on input lengths.")
                 return # Exit if nothing to do

            print(f"Processing {process_frames} frames...")

            # --- Process Frame by Frame ---
            for i in range(process_frames):
                mask = tif_mask.pages[i].asarray()
                raw_img = tif_raw.pages[i].asarray()

                # Determine the correct background frame
                if is_background_stack:
                    bg_img = background_data[i]
                else: # Background is 2D
                    bg_img = background_data # Use the same 2D image

                # Apply mask: where mask is True (non-zero), keep image value, else 0
                # Ensure mask is boolean for np.where if it isn't already
                mask_bool = mask > 0

                img_masked = np.where(mask_bool, raw_img, 0).astype(raw_img.dtype) # Preserve raw dtype
                bg_masked = np.where(mask_bool, bg_img, 0).astype(background_data.dtype) # Preserve bg dtype

                # Write the masked frames to their respective output files
                writer_raw_masked.write(img_masked, contiguous=True, photometric='minisblack')
                writer_bg_masked.write(bg_masked, contiguous=True, photometric='minisblack')

                #if (i + 1) % 100 == 0: # Optional progress update
                #    print(f"Processed frame {i+1}/{process_frames}")

            print(f"Finished processing {process_frames} frames.")

    except FileNotFoundError as e:
        print(f"ERROR: Input file not found: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An error occurred during processing or writing:")
        traceback.print_exc()
        # Attempt to clean up potentially incomplete output files? Optional.
        # try: os.remove(masked_raw_out_path) except OSError: pass
        # try: os.remove(masked_bg_out_path) except OSError: pass
        sys.exit(1)

    print("--- Image Masking Script Finished Successfully ---")
    

def apply_gaussian_subtract(input_path, output_path, sigma):
    """
    Applies a Gaussian blur to each frame of an input TIFF stack,
    subtracts the blurred version from the original, clips the result
    to the uint16 range [0, 65535], and saves the output stack.

    Args:
        input_path (str): Path to the input uint16 TIFF stack.
        output_path (str): Path to save the processed uint16 TIFF stack.
        sigma (float): Standard deviation for Gaussian kernel.
    """
    print("--- Starting Gaussian Blur Subtraction Script ---")
    print(f"Input Image: {input_path}")
    print(f"Output Image: {output_path}")
    print(f"Gaussian Sigma: {sigma}")

    output_dtype = np.uint16
    max_val_uint16 = np.iinfo(output_dtype).max # Get max value for uint16 (65535)

    try:
        # Open input for reading page-by-page and output for writing
        with tiff.TiffFile(input_path) as tif, \
             tiff.TiffWriter(output_path, bigtiff=True) as out_tif:

            n_frames = len(tif.pages)
            print(f"Processing {n_frames} frames...")

            for i, page in enumerate(tif.pages):
                img = page.asarray()

                # Ensure input is treated as uint16 if it isn't already
                if img.dtype != np.uint16:
                    warnings.warn(f"Frame {i}: Input dtype is {img.dtype}, expected uint16. Casting.")
                    img = img.astype(np.uint16)

                # Apply Gaussian blur
                # gaussian_filter handles different dtypes appropriately
                blurred = gaussian_filter(img, sigma=sigma)

                # Subtract and clip
                # Convert to float32 for subtraction to prevent uint underflow/overflow issues
                # Using float64 might be safer for precision but uses more memory
                subtracted_float = img.astype(np.float32) - blurred.astype(np.float32)

                # Clip values below 0
                subtracted_float[subtracted_float < 0] = 0
                # Clip values above the max for the target uint16 type
                subtracted_float[subtracted_float > max_val_uint16] = max_val_uint16

                # Convert back to the target output dtype (uint16)
                subtracted_final = subtracted_float.astype(output_dtype)

                # Write the processed frame with metadata hints
                out_tif.write(
                    subtracted_final,
                    contiguous=True,        # Hint for memory layout
                    photometric='minisblack' # Metadata for grayscale interpretation
                )

                if (i + 1) % 100 == 0: # Optional progress update
                    print(f"Processed frame {i+1}/{n_frames}")

            print(f"Finished processing {n_frames} frames.")

    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An error occurred during processing or writing:")
        traceback.print_exc()
        # Consider removing incomplete output file on error
        # try: os.remove(output_path) except OSError: pass
        sys.exit(1)

    print(f"--- Done saving processed stack to {output_path} ---")
 

    # function: subtract_background
    # input --> data 2d np.array, background as 2d np.array (any type!)
    # features: warn for saturation?
    # options: invert output, normalization

    # function: subtract_background_from_stack
    # input --> data as dask array pointer, background as 2d np.array
    # features: save to output_filepath, warn for saturation?
    # options: invert output, normalization

    # function: read_and_subtract_background_from_stack
    # input --> data_filepath, background_filepath, output_filepath
    # features: warn for saturation?
    # options: invert output, normalization

def stack_subtract_background_8bit(input_filepath, output_filepath, background_img_filepath,
                              invert=False, handle_saturation=False): 
    """
    Reads data as bigtiff wihtout noticing any metadata information.
    Data has to be 3 dim 2d(x,y) + time.
    Background as single 2d image
    Subtracts a background image from each frame of an input stack (uint8),
    optionally inverting the output. 

    This method doesn't work well: Adds the mean of the original background
    as an offset before clipping and saving as uint8. Optionally handles
    pixels saturated in both input and background by setting them to 0. Alternative: normalization

    Parameters:
    ----------
    input_filepath : str
        Input path to the uint8 tiff file/stack.
    output_filepath : str
        Path where the processed uint8 tiff file will be written.
    background_img_filepath : str
        Path to the uint8 background image (expected to be 2D).
    invert : bool, optional
        If False (default), calculates image - background.
        If True, invert both input and background images (255-value)
        before subtraction, effectively calculating background - image.
    handle_saturation : bool, optional
        If True, pixels saturated (255) in both the input frame and the
        original background will be set to 0 in the output frame.
        Defaults to False.

    Returns:
    ----------
    None
    """
    print("--- Starting Background Subtraction ---")
    print(f"Input: {input_filepath}")
    print(f"Background: {background_img_filepath}")
    print(f"Output: {output_filepath}")
    print(f"Invert: {invert}")
    print(f"Handle Saturation: {handle_saturation}") # Print new flag status

    # If you want to use this function only for uint8
    # 

    output_dtype = np.uint8
    max_val_uint8 = 255 # Max value for uint8
    saturation_value = 255 # Value considered saturated for uint8

    # --- Load Background Image ---
    try:
        # Ensure MicroscopeDataReader is imported correctly above
        reader_obj_background = MicroscopeDataReader(background_img_filepath, as_raw_tiff=True, raw_tiff_is_2d=True)
        # Compute immediately as it's needed repeatedly and expected to be small
        #bg_img_original = reader_obj_background. # directly get the image as numpy array
        bg_img_original = np.array(da.squeeze(reader_obj_background.dask_array))

        # Validate background type
        if bg_img_original.dtype != output_dtype:
            raise TypeError(f"Background image dtype is {bg_img_original.dtype}, expected {output_dtype}")
        if len(bg_img_original.shape) != 2:
             raise ValueError(f"Background image has shape {bg_img_original.shape}, expected 2D.")

        print(f"Background image loaded with shape: {bg_img_original.shape}")

    except FileNotFoundError:
        print(f"ERROR: Background image file not found: {background_img_filepath}")
        # better: reraise the exception for exception handling
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load or validate background image: {e}")
        traceback.print_exc() # nice one, doesen't it need the e as argument?
        # reraise error for better error handling!
        sys.exit(1)

    # --- Calculate Offset (Mean of Original Background) ---
    # Calculate before potential inversion
    background_offset = np.mean(bg_img_original)
    # better offset: max-min (2x?)
    print(f"Calculated background offset (mean): {background_offset:.2f}")

    # --- Prepare Background (Invert if specified) ---
    bg_img_processed = bg_img_original.copy() # Work on a copy
    # perhaps this can be done at the end?
    if invert:
        print("Inverting background image for subtraction...")
        bg_img_processed = cv2.bitwise_not(bg_img_processed)

    # Convert background to float32 once for calculations
    # good, should be at least one bigger than original data (uint8 -> float16)
    bg_img_float = bg_img_processed.astype(np.float32)

    # --- Load Input Image Stack Reader ---
    try:
        # Ensure MicroscopeDataReader is imported correctly above
        # Attempting to load directly as raw tiff
        reader_obj_video = MicroscopeDataReader(input_filepath, as_raw_tiff=True, raw_tiff_num_slices=1)

        # Get the dask array representation
        tif_stack = da.squeeze(reader_obj_video.dask_array)

        # Check input stack dtype *before* loop if possible from dask array metadata
        if tif_stack.dtype != output_dtype:
             raise TypeError(f"Input stack dtype is {tif_stack.dtype}, expected {output_dtype}")

        print(f"Input stack loaded (as Dask array). Shape: {tif_stack.shape}, Dtype: {tif_stack.dtype}")

    # Removed NameError exception block
    except FileNotFoundError:
        print(f"ERROR: Input file/folder not found: {input_filepath}")
        # reraise error for better error handling!
        sys.exit(1)
    except Exception as e:
        # This will now catch TypeError if the reader fails in that specific way,
        # along with other potential loading errors.
        # It will also catch NameError if MicroscopeDataReader is not imported/defined.
        print(f"ERROR: Failed to load input stack: {e}")
        traceback.print_exc()
        # reraise error for better error handling!
        sys.exit(1)

    # --- Process Frame by Frame ---
    try:
        with tiff.TiffWriter(output_filepath, bigtiff=True) as tif_writer:
            num_frames = tif_stack.shape[0] # Assuming first dimension is frames
            print(f"Processing {num_frames} frames...")

            for i, img_frame in enumerate(tif_stack): # use the dask array interface!
                # Compute the frame from Dask array to NumPy array
                img = np.array(img_frame) # This is the ORIGINAL frame data

                # Validate frame shape and type (redundant if checked above, but safe)
                if img.shape != bg_img_original.shape:
                    raise ValueError(f"Frame {i} shape {img.shape} differs from background shape {bg_img_original.shape}")
                if img.dtype != output_dtype:
                     warnings.warn(f"Frame {i} dtype is {img.dtype}, expected {output_dtype}. Casting.")
                     img = img.astype(output_dtype)

                # --- Invert Frame if specified ---
                img_processed = img.copy() # Use a copy for processing
                if invert:
                    img_processed = cv2.bitwise_not(img_processed)

                # --- Subtract, Offset, Clip, Convert ---
                # Convert current frame to float32
                img_float = img_processed.astype(np.float32)

                # Subtract background (float result, can be negative)
                subtracted_float = img_float - bg_img_float

                # Add the pre-calculated mean of the original background as offset
                offset_subtracted_float = subtracted_float + background_offset

                # CRITICAL: Clip the result to the valid uint8 range [0, 255]
                # clipping shouldn't accur, raise error but after handle_saturation
                # alternative: normalize if flag in the function is set
                clipped_float = np.clip(offset_subtracted_float, 0, max_val_uint8)

                # Convert back to the final uint8 type
                final_img = clipped_float.astype(output_dtype)

                # --- Handle Saturation (Optional) ---
                if handle_saturation: # do always!
                    # Find pixels saturated in BOTH original frame and original background
                    # what is the goal of this?
                    # should not happen!
                    # putting 255 to zero does the opposit you want?!
                    # throw warning!
                    # this way not possible for inverted images
                    saturated_mask = (img == saturation_value) & (bg_img_original == saturation_value)
                    # Set these pixels to 0 in the final output frame
                    final_img[saturated_mask] = 0

                # --- Write Frame ---
                tif_writer.write(
                    final_img,
                    photometric='minisblack', # Metadata hint
                    contiguous=True         # Memory layout hint
                )

                if (i + 1) % 100 == 0: # Optional progress update
                    # change to overwrite the line (end = . . . google)
                    print(f"Processed frame {i+1}/{num_frames}")

            print(f"Finished processing {num_frames} frames.")

    except Exception as e:
        print(f"\n--- ERROR during processing/writing loop ---")
        print(f"{type(e).__name__}: {e}")
        print("Traceback:")
        traceback.print_exc()
        print(f"Output file '{output_filepath}' may be incomplete or corrupted.")
        # re throw exception for better exception handling!
        sys.exit(1) # Exit on error during processing

    print(f"--- Successfully saved processed stack to {output_filepath} ---")


#if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # import glob
    #
    # project_path = '/Volumes/scratch/neurobiology/zimmer/ulises/active_sensing/epifluorescence_recordings/20220408/data/ZIM1661_worm3/'
    # img_path = '/Volumes/scratch/neurobiology/zimmer/ulises/active_sensing/epifluorescence_recordings/20220408/data/ZIM1661_worm3/2022-04-08_17-22-49_ZIM1661_BAG_worm3-channel-0-behaviour-/2022-04-08_17-22-49_ZIM1661_BAG_worm3-channel-0-behaviour-bigtiff.btf'
    #
    # dlc_coords = '/Volumes/scratch/neurobiology/zimmer/ulises/active_sensing/epifluorescence_recordings/20220408/data/ZIM1661_worm3/2022-04-08_17-22-49_ZIM1661_BAG_worm3-channel-0-behaviour-/2022-04-08_17-22-49_ZIM1661_BAG_worm3-channel-0-behaviour-bigtiffDLC_resnet50_new_worms_5_7_8Apr15shuffle1_57500.h5'
    #
    # center_coords = pd.read_csv(glob.glob(os.path.join(project_path, '*TablePos*'))[0])
    # center_coords = center_coords[['X', 'Y']].values
    # df = pd.read_hdf(dlc_coords)
    # df.head()
    # points = df[df.columns.levels[0][0]]['head'][['x', 'y']][:].values
    # with tiff.TiffFile(img_path) as tif:
    #     img_shape = tif.pages[0].asarray().shape
    #     print('img shape is', img_shape)
    # # img_shape = (900, 900)
    # # points = ([800, 400], [150, 500], [450, 460])
    # # center_coords = ([2, 0], [3, -1], [4, -2])
    # result = distance_to_image_center(img_shape, points)
    #
    # px2mm_ratio = 0.00325
    #
    # print(result)
    # #print(type(result))
    # result_mm = result * px2mm_ratio
    #
    # abs_coords = center_coords + result_mm
    # print(abs_coords)
    #
    # # plt.plot(points)
    # # plt.plot(abs_coords)
    # # plt.show()
    # # fig, ax = plt.subplots()
    # # ax.scatter(abs_coords[:, 0], abs_coords[:, 1])
    # abs_coords_df=pd.DataFrame(abs_coords)
    # print(os.path.join(project_path, 'nose_coords_mm.csv'))
    # abs_coords_df.to_csv(os.path.join(project_path, 'nose_coords_mm.csv'))
    # print('end')

# input_filepath='/Users/ulises.rey/local_data/epifluorescence/2022-04-08_16-12_ZIM1661_BAG_worm1_Ch1bigtiff_masked.btf'
# with tiff.TiffFile(input_filepath) as tif:
#     for i, page in enumerate(tif.pages):
#         img=page.asarray()
#         n_values, intensity = measure_mask(img, 250)
#         print(i, n_values, intensity)
#         #if intensity!=intensity2: print("False")
#         if i == 500: break
