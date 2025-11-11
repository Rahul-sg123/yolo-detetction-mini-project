import os
import time
import json
import math
import sqlite3
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from video_processor import generate_processed_video, get_video_statistics, STATS_STORE

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DEFAULT_VIDEO_FILENAME = "default_queue.mp4"
DB_FILE = "stats.db"

# --- NEW: Database Setup ---
def init_db():
    """Creates the stats table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queue_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                people_count INTEGER,
                wait_time REAL
            )
        """)
        conn.commit()
        conn.close()
        print(f"Database '{DB_FILE}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")

def log_stats_periodically():
    """Background thread: write float wait_time to DB"""
    while True:
        try:
            time.sleep(5)
            if DEFAULT_VIDEO_FILENAME in STATS_STORE:
                stats = STATS_STORE[DEFAULT_VIDEO_FILENAME]
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                people_count = int(stats.get('people_count', 0))
                processing_rate = float(stats.get('processing_rate', 2.0))
                wait_time = 0.0 if people_count == 0 else round(people_count / processing_rate, 1)

                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO queue_stats (timestamp, people_count, wait_time)
                    VALUES (?, ?, ?)
                """, (current_time, people_count, wait_time))
                # keep last 1000 rows
                cursor.execute("""
                    DELETE FROM queue_stats 
                    WHERE id NOT IN (
                        SELECT id FROM queue_stats 
                        ORDER BY id DESC 
                        LIMIT 1000
                    )
                """)
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Error in logging thread: {e}")
            time.sleep(1)

# --- END: Database Setup ---


@app.route('/')
def index():
    return render_template('index.html', default_video=DEFAULT_VIDEO_FILENAME)

@app.route('/history')
def get_history():
    """Return history with wait_time as float for chart consumption."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, people_count, wait_time
            FROM queue_stats
            ORDER BY id DESC
            LIMIT 200
        """)
        rows = cursor.fetchall()
        conn.close()
        history = []
        for r in reversed(rows):
            history.append({
                'timestamp': r['timestamp'],
                'people_count': int(r['people_count']),
                'wait_time': float(r['wait_time']) if r['wait_time'] is not None else 0.0
            })
        return jsonify(history)
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({"error": "Failed to fetch history data"}), 500
# --- END: History API Endpoint ---

@app.route('/video_feed/<filename>')
def video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        print(f"ERROR: Video file not found at {filepath}")
        return "Video file not found", 404
    try:
        return Response(generate_processed_video(filepath),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return jsonify({'error': f'Processing error: {e}'}), 500

@app.route('/current_stats/<filename>')
def get_current_stats(filename):
    """Return current stats; ensure wait_time is a float with one decimal."""
    try:
        if filename in STATS_STORE:
            stats = STATS_STORE[filename]
            people_count = int(stats.get('people_count', 0))
            processing_rate = float(stats.get('processing_rate', 2.0))
            # calculate float wait_time (consistent with video_processor)
            wait_time = 0.0 if people_count == 0 else round(people_count / processing_rate, 1)

            return jsonify({
                'success': True,
                'people_count': people_count,
                'wait_time': wait_time,          # float (e.g. 0.5, 1.0)
                'accuracy': stats.get('accuracy', 'N/A'),
                'avg_speed': stats.get('avg_speed', 0),
                'last_update': stats.get('last_update', time.time())
            })
        return jsonify({'success': False, 'error': 'No stats available'})
    except Exception as e:
        print(f"Error in get_current_stats: {e}")
        return jsonify({'success': False, 'error': 'internal error'}), 500

# --- LEGACY ROUTES (unchanged) ---
@app.route('/process_video/<filename>', methods=['POST'])
def process_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'error': 'File not found'}), 404
    try:
        stats = get_video_statistics(filepath)
        return jsonify({'success': True, **stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# ---------------------------------

if __name__ == '__main__':
    # --- NEW: Init DB and start thread ---
    init_db() # Create the database table
    
    # Start the background logging thread
    # daemon=True means the thread will close when the main app closes
    log_thread = threading.Thread(target=log_stats_periodically, daemon=True)
    log_thread.start()
    print("Background stats logger started.")
    # -------------------------------------
    
    print("=" * 50)
    print("🛕 Temple Queue Management System (v2 with DB)")
    print(f"🔄 Auto-processing: {DEFAULT_VIDEO_FILENAME}")
    print(f"🌐 Server running at: http://127.0.0.1:5000")
    print(f"🔑 Admin Access: Press Shift+A on webpage")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)