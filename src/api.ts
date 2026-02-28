import Constants from 'expo-constants';
import { PredictResponse } from './types';

// API base URL — set EXPO_PUBLIC_API_URL in your environment or update below.
// Deployed backend:   "https://your-app.up.railway.app"
// Same-LAN devices:   "http://192.168.1.6:8000"
export const API_BASE =
  process.env.EXPO_PUBLIC_API_URL ??
  (Constants.expoConfig?.extra?.apiUrl as string | undefined) ??
  'https://unrealistic-lailah-godlessly.ngrok-free.dev';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

const BASE_HEADERS: Record<string, string> = {
  'ngrok-skip-browser-warning': 'skip',
};

/** Health check — returns true if backend is reachable. */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000), headers: BASE_HEADERS });
    const data = await res.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

/** Text-only prediction. */
export async function predictText(text: string): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...BASE_HEADERS },
    body: JSON.stringify({ text }),
  });
  return handleResponse<PredictResponse>(res);
}

/** Audio-only prediction. */
export async function predictAudio(audioUri: string): Promise<PredictResponse> {
  const ext = audioUri.split('.').pop()?.toLowerCase() || 'wav';
  const mimeMap: Record<string, string> = {
    wav: 'audio/wav', m4a: 'audio/m4a', mp4: 'audio/mp4',
    caf: 'audio/x-caf', '3gp': 'audio/3gpp', aac: 'audio/aac',
  };
  const form = new FormData();
  form.append('audio', { uri: audioUri, name: `recording.${ext}`, type: mimeMap[ext] ?? 'audio/wav' } as any);
  const res = await fetch(`${API_BASE}/predict/audio`, { method: 'POST', body: form, headers: BASE_HEADERS });
  return handleResponse<PredictResponse>(res);
}

/** Multimodal prediction with optional audio file URI. */
export async function predictMultimodal(
  text: string,
  audioUri?: string
): Promise<PredictResponse> {
  if (!audioUri) return predictText(text);

  // Derive real extension and MIME from the URI so librosa gets the right bytes
  const ext = audioUri.split('.').pop()?.toLowerCase() || 'wav';
  const mimeMap: Record<string, string> = {
    wav: 'audio/wav',
    m4a: 'audio/m4a',
    mp4: 'audio/mp4',
    caf: 'audio/x-caf',
    '3gp': 'audio/3gpp',
    aac: 'audio/aac',
  };
  const mimeType = mimeMap[ext] ?? 'audio/wav';

  const form = new FormData();
  form.append('text', text);
  form.append('audio', {
    uri: audioUri,
    name: `recording.${ext}`,
    type: mimeType,
  } as any);

  const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form, headers: BASE_HEADERS });
  return handleResponse<PredictResponse>(res);
}
