import pandas as pd
import glob
import os
import numpy as np


def generate_absolute_coordinates(bodypart_coordinates, image_center_coordinates, stage_coordinates, pixel_dimensions):
    """

    :param bodypart_coords:
    :param image_center_coordinates:
    :param stage_coordinates:
    :param pixel_dimensions:
    :return:
    """
    image_center_coordinates = np.asarray(image_center_coordinates)

    result = image_center_coordinates - np.asarray(bodypart_coordinates)

    result_mm = result * pixel_dimensions

    # somehow x coordinates needed to be substracted, whereas y coordinates added
    absolute_coordinates_df = pd.DataFrame()
    absolute_coordinates_df['x'] = stage_coordinates[:, 0] + result_mm[:, 0] #for zim06 there was a - here, this was a subtraction
    absolute_coordinates_df['y'] = stage_coordinates[:, 1] - result_mm[:, 1] # for zim06 there was a + here, this was a sum

    return absolute_coordinates_df


def generate_absolute_coordinates_wrapper(project_path, image_center_coordinates, pixel_dimensions, bodypart_name):
    """

    :param project_path: str
    :param image_center_coordinates: tuple with the coordinates of the stage in the behaviour camera (not the center of the image necesarilly!)
    :param pixel_dimensions:
    :return:
    """
    stage_coordinates = pd.read_csv(glob.glob(os.path.join(project_path, '*TablePos*'))[0])
    stage_coordinates = stage_coordinates[['X', 'Y']].values

    dlc_df = pd.read_hdf(glob.glob(os.path.join(project_path, '*BH/output/*_filtered.h5'))[0])
    bodypart_coordinates = dlc_df[dlc_df.columns.levels[0][0]][bodypart_name][['x', 'y']][:].values

    absolute_coordinates_df = generate_absolute_coordinates(bodypart_coordinates, image_center_coordinates,
                                                            stage_coordinates, pixel_dimensions)

    absolute_coordinates_df.to_csv(os.path.join(project_path, bodypart_name + '_coords_mm.csv'))



def generate_absolute_coordinates_spline_wrapper(project_path, body_segment, image_center_coordinates, pixel_dimensions):
    """
    This function was created to get the absolute coordinates of the spline coordinates. There is another function which
    does a similar thing but directly taking the nose/head coordinate from the DLC prediction.
    This one is more general because it can work with any coordinate.
    
    :param project_path:
    :param body_segment:
    :param image_center_coordinates: 
    :param pixel_dimensions: 
    :return: 
    """
    stage_coordinates = pd.read_csv(glob.glob(os.path.join(project_path, '*TablePos*'))[0])
    # TODO: maybe there should be an if statement in case x,y are lowercase
    stage_coordinates = stage_coordinates[['X', 'Y']].values

    #dlc_df = pd.read_hdf(glob.glob(os.path.join(project_path, '*BH*/*BH*.h5'))[0])
    #bodypart_coordinates = dlc_df[dlc_df.columns.levels[0][0]]['head'][['x', 'y']][:].values

    spline_path = glob.glob(os.path.join(project_path, "*/*skeleton_merged_spline_data.csv"))[0]
    try:
        spline_df = pd.read_csv(spline_path, header=[0, 1], index_col=0)
    except:
        print("skeleton_merged_spline_data.csv could not be found/read")

    bodypart_coordinates = spline_df.loc[:][str(body_segment)][["x", "y"]].values

    absolute_coordinates_df = generate_absolute_coordinates(bodypart_coordinates, image_center_coordinates,
                                                            stage_coordinates, pixel_dimensions)

    absolute_coordinates_df.to_csv(os.path.join(project_path, 'bodypart'+str(body_segment)+'_coords_mm.csv'))
    
    
    return absolute_coordinates_df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # project_path = "/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221127/data/ZIM2165_Gcamp7b_worm1"
    # image_center_coordinates= (340,320) #(0,0)#(344, 350) #(317, 324) #(448, 464)
    # body_segment = 15
    # absolute_coordinates_df = generate_absolute_coordinates_spline_wrapper(project_path, body_segment=body_segment, image_center_coordinates=image_center_coordinates,
    #                                              pixel_dimensions=0.00325)
    # fig, ax = plt.subplots()
    # absolute_coordinates_df.plot(x='x', y='y', ax=ax)
    # stage_coordinates = pd.read_csv(glob.glob(os.path.join(project_path, '*TablePos*'))[0])
    # stage_coordinates.plot(x='X', y='Y', ax=ax)
    # plt.gca().invert_xaxis()
    # plt.gca().invert_yaxis()
    # plt.show()

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--project_path", required=True, help="path to project folder")
    ap.add_argument("-img_center_coords", "--image_center_coordinates", nargs='+', type=float, required=True, help="")
    ap.add_argument("-px_dim", "--pixel_dimensions", type=float, required=True, help="")
    ap.add_argument("-bodypart_name", "--bodypart_name", type=str, required=True, help="")
    args = vars(ap.parse_args())

    project_path = args['project_path']
    image_center_coordinates = tuple(args['image_center_coordinates'])
    print(image_center_coordinates)
    print(type(image_center_coordinates))
    pixel_dimensions = args['pixel_dimensions']
    bodypart_name = args['bodypart_name']

    generate_absolute_coordinates_wrapper(project_path, image_center_coordinates, pixel_dimensions, bodypart_name)
