from flask import Flask, request, jsonify
import os
from sam_processor import process_video

app = Flask(__name__)

gva_src = '/Users/kjr/Python/GeoVideoTagging/'
tmp_folder = '/tmp/SAM2/'





# API endpoint for processing video data
# This endpoint will receive the video data and process it using SAM2
# The processed data will be returned as a JSON response    
#
# The video data is received as a JSON object with the following fields:    
# data_src: The source of the video data
# video_folder: The folder containing the video data
# video_file: The name of the video file
# video_extn: The extension of the video file
# video_fps: The frames per second of the video
# frame_start: The start frame number
# frame_end: The end frame number
# frame_annotation: The annotation for the frame
#
# Sample request:
# curl -X POST -H "Content-Type: application/json" -d '{"data_src": "KJR-DATA-02/Goondoi/Missions/", "video_folder" : "Tour_20241128/DJI_202411300738_001_Goondoi-flights-Innisfail/", "video_file": "DJI_20241130073900_0001_V", "video_extn": "MP4", "video_fps": 29.76, "frame_start": 994, "frame_end": 998, "frame_annotation": {"x1": 106, "y1": 281, "x2": 142, "y2": 294, "width": 737, "height": 415, "type": "rectangle", "tags": ["vessel"], "objectid": "pilot1"}}' http://localhost:3030/sam2 

@app.route('/sam2', methods=['POST'])
def process_data():
    try:
        # Get JSON data from the POST request
        data = request.get_json()
        
        # Check if data is valid
        if not data:
            return jsonify({"error": "Invalid or missing JSON data"}), 400

        # Extract required parameters
        data_src = data.get('data_src')
        video_folder = data.get('video_folder')
        video_file = data.get('video_file')
        video_extn = data.get('video_extn')
        video_fps = data.get('video_fps')
        frame_start = data.get('frame_start')
        frame_end = data.get('frame_end')
        direction = data.get('direction')
        frame_annotation = data.get('frame_annotation')

        # Validate required parameters
        required_params = [
            'data_src', 'video_folder', 'video_file', 
            'video_extn', 'video_fps', 'frame_start', 'frame_end',
            'frame_annotation', 'direction'
        ]
        
        missing_params = [param for param in required_params if data.get(param) is None]
        if missing_params:
            return jsonify({
                "error": f"Missing required parameters: {', '.join(missing_params)}"
            }), 400

        # Validate frame_annotation structure
        required_annotation_fields = [
            'x1', 'y1', 'x2', 'y2', 'width', 'height',
            'type', 'tags', 'objectid'
        ]
        
        if not all(field in frame_annotation for field in required_annotation_fields):
            return jsonify({
                "error": "Invalid frame_annotation structure"
            }), 400

        # Process the video and get annotations
        print('Valid parameters. Processing the video and getting annotations now...')
        annotations = process_video(
            data_src, video_folder, video_file, video_extn,
            video_fps, frame_start, frame_end, direction, frame_annotation,
            gva_src, tmp_folder
        )

        # Return the processed parameters as confirmation
        return jsonify({
            "gva_src": gva_src,
            "data_src": data_src,
            "video_folder": video_folder,
            "video_file": video_file,
            "video_extn": video_extn,
            "video_fps": video_fps,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "tmp_folder": tmp_folder,
            "frame_annotation": frame_annotation,
            "annotations": annotations,
            "status": "Successfully generated frames and annotated"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3030)

