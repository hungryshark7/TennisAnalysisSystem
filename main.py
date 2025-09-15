
from utils import (read_video,
                   save_video)

from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector, court_line_detector


def main():

    # Reading the video
    input_video_path = "input_videos/USOWSF2.mp4"
    video_frames = read_video(input_video_path)

    # Detecting Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detections_stub.pkl')  # Set to True to read from stub
    
    ball_tracker = BallTracker(model_path='models/last.pt')
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                  read_from_stub=True,
                                                  stub_path='tracker_stubs/ball_detections_stub.pkl')  # Set to True to read from stub

    #Court Line Detection Model
    # court_model_path = 'models/keypoints_model.pth'
    # court_line_detector = CourtLineDetector(court_model_path)
    # court_keypoints = court_line_detector.predict(video_frames[0])  # Predict on the first frame
    # Court Line Detection Model
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)

    # Predict per frame (returns a list of keypoints arrays, one per frame)
    court_keypoints_seq = court_line_detector.predict_frames(video_frames, smooth_alpha=0.2)

    # Draw output

    # Initialize output_video_frames with a copy of the original frames
    output_video_frames = video_frames.copy() 

    #Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections)
    #Draw ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #Draw court keypoints
    #output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw court keypoints (per-frame)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints_seq)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()

