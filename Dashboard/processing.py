
# ------------- What is in here? -------------

# This file contains all the processing steps needed for the vuelosophy eyetracker
# When you select a file that already has been processed, the step will be skipt
# Missing files will be detected and processed

# ------------- Imports -------------
import pickle
import re
import json

import pandas as pd
from sympy import Point3D, Line3D, Plane

from functools import partial
from multiprocessing import Pool

from classes import *
from helpers import *


# ------------- Settings -------------

pd.set_option('display.max_columns', 500)

# ------------- App -------------

# This definition will run in a seperate thread once a file is selected
# With every step, a message is send to the main queue
# These messages will serve as feedback for the user

def app(filepath, filename, queue):

    queue.put("START_PROCESSING_VUE")

    # -------------- Output paths --------------
    queue.put("STEP_VUE: Creating output paths")

    base_path = './Vuelosophy_IO/'
    path_h264 = ''
    path_mp4 = base_path + 'output_MP4/'
    path_json = base_path + 'output_JSON/'
    path_pkl = base_path + 'output_PKL/'

    # -------------- Output filenames --------------

    file_h264 = filepath
    file_mp4 = filename + '.mp4'
    file_json = filename + '.json'
    file_pkl = filename + '.pkl'

    queue.put("FILENAME:%s" % filename)

    # -------------- Create JSON file from H264 --------------
    queue.put("STEP_VUE: Creating JSON file")

    if not os.path.exists(path_json + file_json):
        h264_to_json(path_h264, file_h264, path_json, file_json)

    # -------------- Create and split dataframe from json --------------
    queue.put("STEP_VUE: Split Dataframe")

    if not os.path.exists(path_pkl + 'header_' + file_pkl) or not os.path.exists(path_pkl + "body_preprocessed_" + file_pkl):
        header, body = header_body_split(json_to_pandas_df(path_json, file_json))
        pd.to_pickle(header, path_pkl + 'header_' + file_pkl)
    else:
        header = pd.read_pickle(path_pkl + 'body_preprocessed_' + file_pkl)
        body = pd.read_pickle(path_pkl + 'body_preprocessed_' + file_pkl)

    # -------------- Create Frame object --------------
    frame_obj = Frame()

    cap = cv.VideoCapture(filepath)

    frame_obj.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_obj.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_obj.rate = int(cap.get(cv.CAP_PROP_FPS))
    frame_obj.total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    queue.put("FRAMES: %s" % str(body.shape[0]))

    cap.release()

    # -------------- Create Unity object --------------
    unity_obj = Unity()

    unity_obj.width = 13.333
    unity_obj.height = 10
    unity_obj.distance = 12.678

    # -------------- Preprocessing --------------
    queue.put("STEP_VUE: Preprocessing")

    if not os.path.exists(path_pkl + 'body_preprocessed_' + file_pkl):
        body = preprocess(body, unity_obj, frame_obj, path_pkl, file_pkl, queue)
    else:
        body = pd.read_pickle(path_pkl + 'body_preprocessed_' + file_pkl)

    # -------------- Processing video --------------
    queue.put("STEP_VUE:Creating video")

    if not os.path.exists(path_pkl + 'body_final_' + file_pkl) or not os.path.exists(path_mp4 + file_mp4):

        # -------------- Create Video object --------------
        video_obj = Video()

        video_obj.window_name = 'Eyetracker Vuelosophy'
        video_obj.nr_of_frames = frame_obj.total - 2
        video_obj.input_file = path_h264 + file_h264
        video_obj.output_file = path_mp4 + file_mp4

        # -------------- Create mp4 file --------------
        body = create_video(body, video_obj, frame_obj)
        body = interpolate(body, 30)

        pd.to_pickle(body, path_pkl + 'body_final_' + file_pkl)


    body = pd.read_pickle(path_pkl + 'body_final_' + file_pkl)

    queue.put("END_PROCESSING_VUE")


# ------------- Definitions -------------

# This section contains parts of code that are used during the processing
# Every block of code is explained above it

# This definition extracts metadata from the selected file (h264)
# This data is saved in a json file

def h264_to_json(path_h264, file_h264, path_json, file_json):

    # Open the file in Read Binary mode and read it
    with open(path_h264 + file_h264, 'rb') as f:
        data = f.read()

    # Use regular expressions to extract the coordinates
    result = re.findall(b'(?<=\xff\xff\xff\xff\xff)\{.+?(?=\x00\x00\x00\x01)', data)
    result = [str(part)[2:-1] for part in result]

    # Create JSON
    regular_json = '[' + ',\n'.join(result) + ']'

    # Prettify JSON
    pretty_json = json.loads(regular_json)

    # Create new JSON-file
    with open(path_json + file_json, 'w') as f:
        # This line can be used to compact the json file (on one line)
        # f.write(regularjson)

        f.write(json.dumps(pretty_json, indent=4))

# This definition takes the above created JSON and creates a pandas dataframe with it

def json_to_pandas_df(path_json, file_json):
    # Open JSON-file
    with open(path_json + file_json) as f:
        data = json.load(f)

    # Normalize data
    all = pd.json_normalize(data)

    # Return the whole dataframe
    return all

# This definition splits the above created pandas dataframe into the header info and the actual data

def header_body_split(all):
    # Locate header, rename it and save it
    header = all.loc[[0], ['width', 'height', 'frame_rate', 'frame_total']]
    header.rename(columns={'width': 'frame_width', 'height': 'frame_height'}, inplace=True)

    header['frame_width'] = header['frame_width'].astype(int)
    header['frame_height'] = header['frame_height'].astype(int)
    header['frame_total'] = header['frame_total'].astype(int)

    # Locate body, rename it and save it
    body = all[['num', 'ft.x', 'ft.y', 'ft.z']]
    body = body.iloc[1:]
    body.rename(columns={'num': 'frame_number', 'ft.x': 'norm_eye_3d_x', 'ft.y': 'norm_eye_3d_y', 'ft.z': 'norm_eye_3d_z'},
                   inplace=True)

    body['frame_number'] = body['frame_number'].astype(int)

    # Return the header converted to dictionary
    return header.iloc[0].to_dict(), body

# This definition is another group of smaller blocks of code
# First, the data from the above split dataframe will be processed over multiple cores
# Second, using this processed data, eye fixations are calculated
# Finally, the dataframe is pickled and saved in the allocated folder

def preprocess(body, unity, frame, path_pkl, file_pkl, queue):
    # Convert 3D coords to 2D coords
    body = multi_proc_vue(body, unity, frame, queue)

    # Calculate fixations
    body = get_fixations(body)

    body.drop(["norm_eye_3d_x", "norm_eye_3d_y", "norm_eye_3d_z", "px_eye_2d_x2", "px_eye_2d_y2"], axis=1, inplace=True)

    body['px_eye_2d_x'] = body['px_eye_2d_x'].astype(int)
    body['px_eye_2d_y'] = body['px_eye_2d_y'].astype(int)

    # Pickle dataframe body
    pd.to_pickle(body, path_pkl + 'body_preprocessed_' + file_pkl)

    return body

# This is the first of three definitions used above
# This definition prepares the data to be used in a multiprocessing solution
# and returns the processed data

def multi_proc_vue(body, unity, frame, queue):
    columns = body.columns.values.tolist()
    columns.append("px_eye_2d_x")
    columns.append("px_eye_2d_y")

    data = body.values.tolist()
    func = partial(coord_3d_to_2d, unity, frame)
    new_data = multi_proc(func, data, queue)

    body = pd.DataFrame(new_data, columns=columns)

    return body

# This definition is called in the definition above
# It creates (a) pool(s) of workers base on the amount of cores used to process
# Next, tells the pools to use a defined definition to process the data
# Finally returns the results

def multi_proc(func, data, queue, cores = 0):
    pool = None

    if cores > 0:
        print("Using %s cores" % cores)
        pool = Pool(cores)
    else:
        print("Using all cores")
        pool = Pool(None, f_init, [queue])

    print("Start processing ")

    results = pool.map(func, data)
    return results

def f_init(q):
    coord_3d_to_2d.q = q

# This is the definition called in the above definition
# It does the actual processing
# It converts the 3D coordinates from the eyetracker into 2D coordinates
# Those will be used to define positions in frames/pictures

def coord_3d_to_2d(unity, frame, data):

    # Variables
    unity_width_half = unity.width / 2
    unity_height_half = unity.height / 2
    unity_distance = unity.distance

    plane = Plane(Point3D(-unity_width_half, unity_height_half, -unity_distance),
                  Point3D(-unity_width_half, -unity_height_half, -unity_distance),
                  Point3D(unity_width_half, unity_height_half, -unity_distance))

    px_eye_2d_x, px_eye_2d_y = calculate_coord(plane, unity, frame, data[1], data[2], data[3])
    data.append(px_eye_2d_x)
    data.append(px_eye_2d_y)

    coord_3d_to_2d.q.put("STEP_VUE_SUB")

    return data

# This definition is an addition to the definition above

def calculate_coord(plane, unity, frame, norm_eye_3d_x, norm_eye_3d_y, norm_eye_3d_z):
    unity_width_half = unity.width / 2
    unity_height_half = unity.height / 2
    frame_width = frame.width
    frame_height = frame.height

    line = Line3D(Point3D(0, 0, 0),
                  Point3D(norm_eye_3d_x, norm_eye_3d_y, norm_eye_3d_z))

    intersection = plane.intersection(line)[0]
    px_eye_2d_x = map_value(intersection.x, -unity_width_half, unity_width_half, 0, frame_width)
    px_eye_2d_y = map_value(intersection.y, -unity_height_half, unity_height_half, frame_height, 0)

    return px_eye_2d_x, px_eye_2d_y

# This definition is an addition to the definition above

def map_value(value, left_min, left_max, right_min, right_max):
    left_span = left_max - left_min
    right_span = right_max - right_min
    value_scaled = (value - left_min) / left_span  # convert the left range into a 0-1 range
    return np.nan_to_num(right_min + (value_scaled * right_span))  # convert the 0-1 range into a value in the right range

# This definition is the second part of the "processing" definition
# It calculates fixatitions base on the movement of the eyes
# This uses the above calculated 2D coordinates

def get_fixations(body):
    # Position the data
    body['px_eye_2d_x2'] = body['px_eye_2d_x'].shift(-1)
    body['px_eye_2d_y2'] = body['px_eye_2d_y'].shift(-1)

    # Delete last row (empty)
    body = body[:-1]

    # Calculate the eye distance
    body['eye_dist'] = ((body['px_eye_2d_x2'] - body['px_eye_2d_x']) ** 2) + ((body['px_eye_2d_y2'] - body['px_eye_2d_y2']) ** 2)
    q1 = np.percentile(body['eye_dist'], 25)
    q3 = np.percentile(body['eye_dist'], 75)

    # Calculate the threshold
    threshhold = q3 + (1.5 * (q3 - q1))

    # Check if it is a fixation or not
    body['fixation'] = body.apply(lambda x: row_fixation(x, threshhold), axis=1)

    return body

# This definitions uses the above calculated values and decides if the movement is a fixation or not

def row_fixation(row, threshhold):
    if row.eye_dist <= threshhold:
        return 1
    else:
        return 0
