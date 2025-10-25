import os
import time
import json
import sqlite3
import threading
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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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
    """A background thread function to log stats to the DB."""
    while True:
        try:
            # Wait for 15 seconds before logging
            time.sleep(15)
            
            # Check if the stats store has data for our default video
            if DEFAULT_VIDEO_FILENAME in STATS_STORE:
                stats = STATS_STORE[DEFAULT_VIDEO_FILENAME]
                
                # Only log if there are people, to avoid empty charts
                if stats.get('people_count', 0) > 0:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO queue_stats (people_count, wait_time)
                        VALUES (?, ?)
                    """, (stats['people_count'], stats['wait_time']))
                    conn.commit()
                    conn.close()
                    print(f"Logged to DB: {stats['people_count']} people, {stats['wait_time']} min wait")
                    
        except Exception as e:
            print(f"Error in logging thread: {e}")

# --- END: Database Setup ---


@app.route('/')
def index():
    return render_template('index.html', default_video=DEFAULT_VIDEO_FILENAME)

# --- NEW: History API Endpoint ---
@app.route('/history')
def get_history():
    """Returns the last 200 data points as JSON for charting."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row # This lets us get columns by name
        cursor = conn.cursor()
        
        # Get the most recent 200 entries, oldest first
        cursor.execute("""
            SELECT timestamp, people_count, wait_time
            FROM queue_stats
            ORDER BY timestamp DESC
            LIMIT 200
        """)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert sqlite3.Row objects to standard dicts
        history_data = [dict(row) for row in reversed(rows)] # reversed() to get chronological order
        
        return jsonify(history_data)
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({"error": str(e)}), 500
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
def current_stats(filename):
    """Return the latest stats."""
    if filename not in STATS_STORE:
        return jsonify({'success': False, 'error': 'Stats not ready yet'}), 404
    
    s = STATS_STORE.get(filename, {}).copy()
    s['success'] = True
    return jsonify(s)

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