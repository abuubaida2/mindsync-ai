import AsyncStorage from '@react-native-async-storage/async-storage';
import { HistoryEntry } from './types';

const KEY = 'mindsync_history';

export async function loadHistory(): Promise<HistoryEntry[]> {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export async function saveEntry(entry: HistoryEntry): Promise<void> {
  const history = await loadHistory();
  const updated = [entry, ...history].slice(0, 50); // keep last 50
  await AsyncStorage.setItem(KEY, JSON.stringify(updated));
}

export async function clearHistory(): Promise<void> {
  await AsyncStorage.removeItem(KEY);
}
