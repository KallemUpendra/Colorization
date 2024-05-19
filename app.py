from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import cv2
import numpy as np

app = Flask(__name__)

# Global variables to store progress
progress = 0

# Function to update progress
def update_progress(new_progress):
    global progress
    progress = new_progress

# Function to get current progress
@app.route('/progress')
def get_progress():
    return jsonify(progress)

# Function to colorize the image
def colorize_image(image_path):
    DIR = "model"
    PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
    MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")
    POINTS = os.path.join(DIR, "pts_in_hull.npy")

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Save colorized image with progress updates
    colorized_image_path = 'uploads/colorized_' + os.path.basename(image_path)
    total_pixels = image.shape[0] * image.shape[1]
    processed_pixels = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            processed_pixels += 1
            if processed_pixels % 1000 == 0:
                new_progress = processed_pixels / total_pixels * 100
                update_progress(new_progress)
            if processed_pixels == total_pixels:
                update_progress(100.00)
    cv2.imwrite(colorized_image_path, colorized)

    return colorized_image_path, 'image'

# Function to colorize the video
def colorize_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return None, None

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec for AVI format
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video
    out = cv2.VideoWriter('uploads/colorized_' + os.path.basename(video_path), fourcc, fps, (width, height))

    # Colorization parameters
    DIR = "model"
    PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
    MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")
    POINTS = os.path.join(DIR, "pts_in_hull.npy")

    # Load the colorization model
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Process each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Convert frame to LAB color space
        scaled = frame.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # Resize frame to match input size of the network
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # Set input and perform forward pass
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))

        # Merge L channel with predicted AB channels
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        # Write the colorized frame to output video
        out.write(colorized)

        # Update progress
        progress = frame_count / total_frames * 100
        update_progress(progress)

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return 'uploads/colorized_' + os.path.basename(video_path), 'video'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST','GET'])
def colorize():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension in ['.jpg', '.jpeg', '.png']:
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)
            colorized_image_path, original_type = colorize_image(image_path)
            if colorized_image_path:
                return render_template('index.html', colorized=colorized_image_path, original=image_path, original_type='image', colorized_type=original_type)
            else:
                return "Error processing image"
        elif file_extension in ['.mp4', '.avi', '.mov', '.webm']:
            video_path = os.path.join('uploads', file.filename)
            file.save(video_path)
            colorized_video_path, original_type = colorize_video(video_path)
            if colorized_video_path:
                return render_template('index.html', colorized=colorized_video_path, original=video_path, original_type='video', colorized_type=original_type)
            else:
                return "Error processing video"
        else:
            return "Unsupported file format"


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
