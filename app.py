import os
# Fix OpenMP duplicate runtime issue before ML libs if you still use them in same process
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from video_processor import generate_processed_video, get_video_statistics, STATS_STORE

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {e}'}), 500

@app.route('/video_feed/<filename>')
def video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    try:
        return Response(generate_processed_video(filepath),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return jsonify({'error': f'Processing error: {e}'}), 500

@app.route('/process_video/<filename>', methods=['POST'])
def process_video(filename):
    """Legacy full processing endpoint (keeps compatibility)."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    try:
        stats = get_video_statistics(filepath)
        return jsonify({'success': True, **stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/current_stats/<filename>')
def current_stats(filename):
    """Return the latest stats (updated by generate_processed_video while streaming)."""
    if filename not in STATS_STORE:
        return jsonify({'success': False, 'error': 'No stats available yet'}), 404
    # copy to avoid exposing internal object
    s = STATS_STORE.get(filename, {}).copy()
    s['success'] = True
    return jsonify(s)

# serve uploaded files if needed
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
