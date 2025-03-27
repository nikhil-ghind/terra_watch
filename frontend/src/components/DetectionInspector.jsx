import React, { useState } from 'react';
import { submitCorrection } from '../api/corrections';

const CLASSES = ['no_change', 'deforestation', 'urban_expansion', 'water_shift', 'agriculture'];

export default function DetectionInspector({ detection, onClose }) {
  const [correctedClass, setCorrectedClass] = useState(detection.predicted_class);
  const [note, setNote] = useState('');
  const [status, setStatus] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await submitCorrection({ detection_id: detection.id, analyst_id: 'analyst_1', corrected_class: correctedClass, note });
      setStatus('Correction submitted!');
    } catch (err) {
      setStatus('Error: ' + err.message);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <h3>Detection Inspector</h3>
        <button onClick={onClose}>✕</button>
      </div>
      <p><b>Class:</b> {detection.predicted_class}</p>
      <p><b>Confidence:</b> {(detection.confidence * 100).toFixed(1)}%</p>
      <hr />
      <form onSubmit={handleSubmit}>
        <label>Corrected Class</label>
        <select value={correctedClass} onChange={e => setCorrectedClass(e.target.value)} style={{ display: 'block', width: '100%', marginBottom: 8 }}>
          {CLASSES.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <label>Note</label>
        <input value={note} onChange={e => setNote(e.target.value)} style={{ display: 'block', width: '100%', marginBottom: 8 }} />
        <button type="submit">Submit Correction</button>
      </form>
      {status && <p style={{ color: status.startsWith('Error') ? 'red' : 'green' }}>{status}</p>}
    </div>
  );
}
