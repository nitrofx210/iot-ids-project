"""
utils/predict.py
Hybrid RF + Isolation Forest inference pipeline.
Used by both the Flask API and the Streamlit dashboard.
"""
import numpy as np
import joblib
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / 'models'


class HybridIDS:
    def __init__(self):
        self.rf = joblib.load(MODEL_DIR / 'random_forest_binary.pkl')
        self.iso = joblib.load(MODEL_DIR / 'isolation_forest.pkl')
        self.scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
        self.label_encoder = joblib.load(MODEL_DIR / 'label_encoder.pkl')

        with open(MODEL_DIR / 'hybrid_thresholds.json') as f:
            thresholds = json.load(f)
        self.rf_threshold = thresholds['rf_confidence_threshold']
        self.if_threshold = thresholds['if_anomaly_threshold']

        try:
            self.rf_multi = joblib.load(MODEL_DIR / 'random_forest_multiclass.pkl')
        except FileNotFoundError:
            self.rf_multi = None

    def _normalise_if_score(self, raw_score: float, score_min: float = -0.5, score_max: float = 0.5) -> float:
        """Normalise raw IF decision score to [0, 1]. Higher = more anomalous."""
        return 1 - (raw_score - score_min) / (score_max - score_min)

    def predict(self, feature_vector: list) -> dict:
        """
        Run hybrid inference on a single feature vector.

        Args:
            feature_vector: list of float values in the same order as training features

        Returns:
            dict with keys: is_attack, attack_label, rf_confidence,
                            if_anomaly_score, detection_method, verdict
        """
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Stage 1: Random Forest
        rf_prob = self.rf.predict_proba(X_scaled)[0]
        rf_pred = int(np.argmax(rf_prob))
        rf_attack_prob = float(rf_prob[1])
        rf_confidence = float(rf_prob[rf_pred])

        # Stage 2: Isolation Forest
        if_raw = float(self.iso.decision_function(X_scaled)[0])
        if_score = float(self._normalise_if_score(if_raw))

        # Hybrid decision
        if rf_confidence >= self.rf_threshold:
            is_attack = bool(rf_pred == 1)
            method = 'Random Forest (high confidence)'
        else:
            if if_score >= self.if_threshold:
                is_attack = True
                method = 'Isolation Forest (anomaly detected)'
            else:
                is_attack = bool(rf_pred == 1)
                method = 'Random Forest (low confidence)'

        # Multi-class label
        attack_label = 'Normal'
        if is_attack:
            if self.rf_multi is not None:
                mc_pred = int(self.rf_multi.predict(X_scaled)[0])
                attack_label = self.label_encoder.inverse_transform([mc_pred])[0]
            else:
                attack_label = 'Attack (unclassified)'

        return {
            'is_attack': is_attack,
            'attack_label': attack_label,
            'rf_attack_probability': round(rf_attack_prob, 4),
            'rf_confidence': round(rf_confidence, 4),
            'if_anomaly_score': round(if_score, 4),
            'detection_method': method,
            'verdict': 'ATTACK DETECTED' if is_attack else 'NORMAL TRAFFIC'
        }

    def predict_batch(self, feature_matrix: list) -> list:
        """Run inference on a batch of feature vectors."""
        return [self.predict(row) for row in feature_matrix]
