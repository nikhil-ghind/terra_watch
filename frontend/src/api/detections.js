import client from './client';
export const fetchDetections = (params) => client.get('/detections/', { params }).then(r => r.data);
export const fetchDetection = (id) => client.get(`/detections/${id}`).then(r => r.data);
