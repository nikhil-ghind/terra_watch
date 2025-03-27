import client from './client';
export const fetchScenes = () => client.get('/scenes/').then(r => r.data);
export const fetchScene = (id) => client.get(`/scenes/${id}`).then(r => r.data);
export const uploadScene = (formData) => client.post('/scenes/', formData, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data);
