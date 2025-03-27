import React from 'react';

const STATUS_COLORS = { pending: '#888', processing: '#f90', done: '#0a0', failed: '#f00' };

export default function SceneList({ scenes, onSelect, selected }) {
  return (
    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
      {scenes.map(s => (
        <li key={s.id} onClick={() => onSelect(s)}
            style={{ padding: '8px 12px', marginBottom: 4, cursor: 'pointer', borderRadius: 4,
                     background: selected?.id === s.id ? '#e8f4fd' : '#f9f9f9',
                     border: '1px solid #ddd' }}>
          <div style={{ fontWeight: 600 }}>{s.name}</div>
          <div style={{ fontSize: 12, color: STATUS_COLORS[s.status] || '#888' }}>{s.status}</div>
          <div style={{ fontSize: 11, color: '#999' }}>{new Date(s.acquired_at).toLocaleDateString()}</div>
        </li>
      ))}
    </ul>
  );
}
