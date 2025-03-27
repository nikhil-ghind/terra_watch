import client from './client';
export const submitCorrection = (payload) => client.post('/corrections/', payload).then(r => r.data);
