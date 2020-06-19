# ------------- Imports -------------

import pandas as pd

from classes import Frame, Video
from helpers import *

def pupil_labs(folderpath, queue):

    # -------------- Output paths --------------

    queue.put("STEP_PUP: Creating output paths")

    # ------------- MP4 -------------
    input_path_mp4 = folderpath + '/world.mp4'
    output_path_mp4 = 'Pupil_Labs_IO/output_MP4/world.mp4'

    # ------------- CSV -------------
    input_path_csv = folderpath + '/gaze_positions.csv'
    input_file_csv_fixations = folderpath + '/fixations.csv'

    # ------------- PKL -------------
    output_pkl = 'Pupil_Labs_IO/output_PKL/body.pkl'
    output_fixations = 'Pupil_Labs_IO/output_PKL/fixations.pkl'

    # -------------- Create Frame object --------------

    cap = cv.VideoCapture(input_path_mp4)

    frame_obj = Frame()

    frame_obj.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_obj.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_obj.rate = int(cap.get(cv.CAP_PROP_FPS))
    frame_obj.total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    queue.put("FRAMES: %s" % str(frame_obj.total))

    cap.release()


    # -------------- Create Video object --------------

    video_obj =  Video()

    video_obj.window_name = "Pupil Labs Core Eyetracker"
    video_obj.input_file = input_path_mp4
    video_obj.output_file = output_path_mp4
    video_obj.nr_of_frames = frame_obj.total

    # -------------- Gets the gaze positions --------------
    queue.put("STEP_PUP: Read gaze positions")

    body = pd.read_csv(input_path_csv)[['world_index', 'norm_pos_x', 'norm_pos_y']]
    body = body.rename(columns = {'world_index': 'frame_number', 'norm_pos_x': 'px_eye_2d_x', 'norm_pos_y': 'px_eye_2d_y'})
    body = body.drop_duplicates(subset = 'frame_number', keep = 'first').reset_index(drop=True)

    body['px_eye_2d_x'] = (body['px_eye_2d_x'] * frame_obj.width).astype(int)
    body['px_eye_2d_y'] = (body['px_eye_2d_y'] * frame_obj.height).astype(int)

    body['px_eye_2d_y'] = frame_obj.height - body['px_eye_2d_y']

    # -------------- Gets the fixations --------------
    queue.put("STEP_PUP: Calculate fixations")

    if not os.path.exists(output_fixations):
        fixations = pd.read_csv(input_file_csv_fixations)
        body['fixation'] = body.apply(lambda frame: check_fixation(frame, fixations, queue), axis=1)
        pd.to_pickle(body, output_fixations)
    else:
        body = pd.read_pickle(output_fixations)

    # model goedzetten
    net = cv.dnn.readNet("Weights/TinyWeightsZarinV4.weights",
                         "Configs/TinyConfigZarin.cfg")  # weight en configuration file ophalen
    classes = ["Jam", "Knife", "Bread", "Choco"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if not os.path.exists(output_path_mp4):
        queue.put("STEP_PUP: Creating video")

        body = create_video(body, video_obj, frame_obj)
        body = interpolate(body, 50)
        pd.to_pickle(body, output_pkl)

    queue.put("END_PROCESSING_PUP")

# This definition checks if the frame contains a fixation

def check_fixation(frame, fixations, queue):
    fixations = fixations[fixations['end_frame_index'] > frame['frame_number']]

    for index, row in fixations.iterrows():
        if frame['frame_number'] in range(row['start_frame_index'], row['end_frame_index'] + 1):
            queue.put("STEP_PUP_SUB")
            return 1

    queue.put("STEP_PUP_SUB")
    return 0


