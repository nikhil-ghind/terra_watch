import React, { useState, useEffect } from 'react';
import { fetchScenes } from '../api/scenes';
import { fetchDetections } from '../api/detections';
import SceneList from '../components/SceneList';
import DetectionInspector from '../components/DetectionInspector';

export default function Dashboard() {
  const [scenes, setScenes] = useState([]);
  const [selectedScene, setSelectedScene] = useState(null);
  const [detections, setDetections] = useState([]);
  const [selectedDetection, setSelectedDetection] = useState(null);

  useEffect(() => {
    fetchScenes().then(setScenes).catch(console.error);
  }, []);

  const handleSelectScene = (scene) => {
    setSelectedScene(scene);
    fetchDetections({ scene_id: scene.id }).then(setDetections).catch(console.error);
  };

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'sans-serif' }}>
      <div style={{ width: 280, borderRight: '1px solid #ddd', overflowY: 'auto', padding: 16 }}>
        <h2>Scenes</h2>
        <SceneList scenes={scenes} onSelect={handleSelectScene} selected={selectedScene} />
      </div>
      <div style={{ flex: 1, padding: 16 }}>
        <h2>Detections {selectedScene ? `— ${selectedScene.name}` : ''}</h2>
        {detections.length === 0 && <p>Select a scene to view detections.</p>}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {detections.map(d => (
            <div key={d.id} style={{ padding: 8, border: '1px solid #aaa', borderRadius: 4, cursor: 'pointer', background: selectedDetection?.id === d.id ? '#e8f4fd' : 'white' }}
                 onClick={() => setSelectedDetection(d)}>
              <div><b>{d.predicted_class}</b></div>
              <div>{(d.confidence * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </div>
      {selectedDetection && (
        <div style={{ width: 320, borderLeft: '1px solid #ddd', padding: 16 }}>
          <DetectionInspector detection={selectedDetection} onClose={() => setSelectedDetection(null)} />
        </div>
      )}
    </div>
  );
}
