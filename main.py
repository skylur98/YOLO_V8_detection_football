from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
# from team_assigner import TeamAssigner
# from player_ball_assigner import PlayerBallAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    #Read Video
    video_frames = read_video('input_videos/test_1.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                  
                                       stub_path='stubs/track_stubs.pk1')
    
    #Draw output
    #Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Get object positions 
    tracker.add_position_to_tracks(tracks)

     # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

      # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    #draw_annotations / sample
    # tracker = tracker.get_object_tracks(video_frames,read_from_stub=False,
                                        
    #                                     stub_path="stubs/track+stubs.pk1")
    # tracker = Tracker('models/last.pt')


    #Save Video
    save_video(video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()