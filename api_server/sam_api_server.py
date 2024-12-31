from flask import Flask, request, jsonify
import os
from sam_processor import process_video

app = Flask(__name__)

gva_src = '/Users/kjr/Python/GeoVideoTagging/'
tmp_folder = '/tmp/SAM2/'






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
        frame_annotation = data.get('frame_annotation')

        # Validate required parameters
        required_params = [
            'data_src', 'video_folder', 'video_file', 
            'video_extn', 'video_fps', 'frame_start', 'frame_end',
            'frame_annotation'
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
            video_fps, frame_start, frame_end, frame_annotation,
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

