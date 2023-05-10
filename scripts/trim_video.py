import cv2


def trim_video(input_file, output_file, start_sec, end_sec):
    # Open the input video file
    cap = cv2.VideoCapture(input_file)

    # Get the frame rate and total number of frames in the input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the start and end frame numbers based on the input start and end times
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Make sure the start and end frames are within the bounds of the video
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    # Set the starting position of the video to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create a VideoWriter object to write the output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    # Loop through the frames of the input video and write the frames to the output video
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    # Release the video files
    cap.release()
    out.release()

trim_video('./combined_new.mp4', 'trimmed.mp4', 0, 18)

# ffmpeg -i trimmed.mp4 -c:v libx264 -preset slow -crf 22 -c:a copy trimmed_new.mp4
