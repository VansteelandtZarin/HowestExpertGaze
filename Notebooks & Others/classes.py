class Class_device:
    path = ''
    #H264
    input_path_h264 = ''
    input_file_h264 = ''
    #MP4
    output_path_mp4 = ''
    output_file_mp4 = ''
    #JSON
    output_path_json = ''
    output_file_json = ''
    #PKL
    output_path_pkl = ''
    output_file_pkl = ''


class Class_unity:
    width = 0.0
    height = 0
    distance = 0.0
    
class Class_frame:
    width = 0
    height = 0
    rate = 0.0
    total = 0
#     number = 0

    
class Class_video:
    window_name = ''
    input_file = ''
    output_file = ''
    nr_of_frames = 0

class Class_pup:
    input_path_mp4 = ''  # C:/HIT-lab/projects/HIT_EyeTracking/vendor_PUP/input_MP4/
    input_file_mp4 = ''  # world.mp4
    output_path_mp4 = ''  # C:/HIT-lab/projects/HIT_EyeTracking/vendor_PUP/output_MP4/
    output_file_mp4 = ''  # HIT_EyeTracking_world.mp4

    # CSV
    input_path_csv = ''  # C:/HIT-lab/projects/HIT_EyeTracking/vendor_PUP/input_CSV/
    input_file_csv = ''  # 'gaze_positions.csv'
    input_file_csv_fixations = ''  # fixations.csv'

    # PKL
    output_path_pkl = ''  # C:/HIT-lab/projects/HIT_EyeTracking/vendor_PUP/output_PKL/
    output_file_pkl_df_header = ''  # pkl_df_header.pkl
    output_file_pkl_df_body = ''  # pkl_df_body.pkl