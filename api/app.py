"""
api/app.py
Flask REST API for the ML-Based IoT IDS.
Week 7 deliverable — skeleton ready from Week 2.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'detections.db')

# ── Database Setup ────────────────────────────────────────────────────────────

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            is_attack   INTEGER NOT NULL,
            attack_label TEXT,
            rf_confidence REAL,
            if_score    REAL,
            method      TEXT,
            verdict     TEXT,
            source_ip   TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_detection(result: dict, source_ip: str = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO detections
        (timestamp, is_attack, attack_label, rf_confidence, if_score, method, verdict, source_ip)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.datetime.utcnow().isoformat(),
        int(result['is_attack']),
        result['attack_label'],
        result['rf_confidence'],
        result['if_anomaly_score'],
        result['detection_method'],
        result['verdict'],
        source_ip or 'unknown'
    ))
    conn.commit()
    conn.close()


# ── Lazy-load model (avoids startup crash if models not yet trained) ──────────
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from utils.predict import HybridIDS
            _model = HybridIDS()
        except Exception as e:
            return None, str(e)
    return _model, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'IoT IDS API', 'version': '1.0.0'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Classify a single network traffic flow.

    Request body (JSON):
    {
        "features": [f1, f2, f3, ...],   // feature vector in training order
        "source_ip": "192.168.1.10"       // optional
    }

    Response:
    {
        "is_attack": true,
        "attack_label": "DDoS",
        "rf_attack_probability": 0.97,
        "rf_confidence": 0.97,
        "if_anomaly_score": 0.83,
        "detection_method": "Random Forest (high confidence)",
        "verdict": "ATTACK DETECTED"
    }
    """
    data = request.get_json(silent=True)
    if not data or 'features' not in data:
        return jsonify({'error': 'Request must include "features" array'}), 400

    model, err = get_model()
    if err:
        return jsonify({'error': f'Model not loaded: {err}'}), 503

    try:
        result = model.predict(data['features'])
        log_detection(result, source_ip=data.get('source_ip'))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Classify multiple flows in one request.

    Request body:
    { "flows": [[f1, f2, ...], [f1, f2, ...], ...] }
    """
    data = request.get_json(silent=True)
    if not data or 'flows' not in data:
        return jsonify({'error': 'Request must include "flows" array'}), 400

    model, err = get_model()
    if err:
        return jsonify({'error': f'Model not loaded: {err}'}), 503

    try:
        results = model.predict_batch(data['flows'])
        for r in results:
            log_detection(r)
        return jsonify({'results': results, 'count': len(results)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Return detection statistics from the SQLite log."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total = cursor.execute('SELECT COUNT(*) FROM detections').fetchone()[0]
    attacks = cursor.execute('SELECT COUNT(*) FROM detections WHERE is_attack=1').fetchone()[0]
    by_label = cursor.execute(
        'SELECT attack_label, COUNT(*) FROM detections WHERE is_attack=1 GROUP BY attack_label ORDER BY COUNT(*) DESC'
    ).fetchall()
    recent = cursor.execute(
        'SELECT timestamp, attack_label, verdict, source_ip FROM detections ORDER BY id DESC LIMIT 20'
    ).fetchall()

    conn.close()

    return jsonify({
        'total_flows_analysed': total,
        'total_attacks_detected': attacks,
        'normal_traffic': total - attacks,
        'attack_rate_pct': round(attacks / total * 100, 2) if total > 0 else 0,
        'attacks_by_category': [{'label': r[0], 'count': r[1]} for r in by_label],
        'recent_detections': [
            {'timestamp': r[0], 'label': r[1], 'verdict': r[2], 'source_ip': r[3]}
            for r in recent
        ]
    })


@app.route('/history', methods=['GET'])
def history():
    """Return last N detections. Query param: ?limit=100"""
    limit = request.args.get('limit', 100, type=int)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT * FROM detections ORDER BY id DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()

    cols = ['id', 'timestamp', 'is_attack', 'attack_label', 'rf_confidence',
            'if_score', 'method', 'verdict', 'source_ip']
    return jsonify({'detections': [dict(zip(cols, r)) for r in rows]})


if __name__ == '__main__':
    init_db()
    print('IoT IDS API starting on http://localhost:5000')
    print('Endpoints:')
    print('  GET  /health')
    print('  POST /predict')
    print('  POST /predict/batch')
    print('  GET  /stats')
    print('  GET  /history')
    app.run(debug=True, host='0.0.0.0', port=5000)
