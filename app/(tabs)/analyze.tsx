import { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TextInput, TouchableOpacity,
  ScrollView, ActivityIndicator, Alert, KeyboardAvoidingView, Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/colors';
import { predictText, predictAudio, predictMultimodal, getApiBase } from '@/api';
import { saveEntry } from '@/store';
import { HistoryEntry } from '@/types';

type RecordingState = 'idle' | 'recording' | 'done';
type AnalysisMode = 'text' | 'voice' | 'both';

const MODES: { key: AnalysisMode; label: string; icon: string }[] = [
  { key: 'text',  label: 'Text Only',  icon: 'text-outline' },
  { key: 'voice', label: 'Voice Only', icon: 'mic-outline' },
  { key: 'both',  label: 'Both',       icon: 'git-merge-outline' },
];

const WAV_OPTIONS: Audio.RecordingOptions = {
  android: {
    extension: '.wav',
    outputFormat: Audio.AndroidOutputFormat.DEFAULT,
    audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 256000,
  },
  ios: {
    extension: '.wav',
    outputFormat: Audio.IOSOutputFormat.LINEARPCM,
    audioQuality: Audio.IOSAudioQuality.HIGH,
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 256000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  web: { mimeType: 'audio/wav', bitsPerSecond: 256000 },
};

export default function AnalyzeScreen() {
  const router = useRouter();
  const [mode, setMode] = useState<AnalysisMode>('both');
  const [text, setText] = useState('');
  const [recordState, setRecordState] = useState<RecordingState>('idle');
  const [audioUri, setAudioUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const recordingRef = useRef<Audio.Recording | null>(null);

  // â”€â”€ Mode switch â€” reset inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  function switchMode(m: AnalysisMode) {
    setMode(m);
    setText('');
    setAudioUri(null);
    setRecordState('idle');
    if (recordingRef.current) {
      recordingRef.current.stopAndUnloadAsync().catch(() => {});
      recordingRef.current = null;
    }
  }

  // â”€â”€ Audio recording â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  async function startRecording() {
    try {
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) { Alert.alert('Permission required', 'Please allow microphone access.'); return; }
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const { recording } = await Audio.Recording.createAsync(WAV_OPTIONS);
      recordingRef.current = recording;
      setRecordState('recording');
    } catch (e) {
      Alert.alert('Error', `Could not start recording: ${e}`);
    }
  }

  async function stopRecording() {
    try {
      if (!recordingRef.current) return;
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;
      setAudioUri(uri ?? null);
      setRecordState('done');
    } catch (e) {
      Alert.alert('Error', `Could not stop recording: ${e}`);
    }
  }

  function clearAudio() {
    setAudioUri(null);
    setRecordState('idle');
  }

  // â”€â”€ Readiness check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const canSubmit =
    (mode === 'text'  && text.trim().length > 0) ||
    (mode === 'voice' && recordState === 'done' && !!audioUri) ||
    (mode === 'both'  && text.trim().length > 0 && recordState === 'done' && !!audioUri);

  // â”€â”€ Submit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  async function handleAnalyze() {
    if (!canSubmit) return;
    setLoading(true);
    try {
      let result;
      if (mode === 'text') {
        result = await predictText(text.trim());
      } else if (mode === 'voice') {
        result = await predictAudio(audioUri!);
      } else {
        result = await predictMultimodal(text.trim(), audioUri!);
      }
      const entry: HistoryEntry = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        text: mode === 'voice' ? '[Voice recording]' : text.trim(),
        result,
      };
      await saveEntry(entry);
      router.push({ pathname: '/results', params: { data: JSON.stringify(result), text: entry.text } });
    } catch (err: any) {
      Alert.alert(
        'Analysis failed',
        `${err.message ?? 'Unknown error.'}

URL: ${getApiBase()}`,
      );
    } finally {
      setLoading(false);
    }
  }

  const showText  = mode === 'text'  || mode === 'both';
  const showVoice = mode === 'voice' || mode === 'both';

  return (
    <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView style={styles.container} contentContainerStyle={styles.content}>

        {/* â”€â”€ Mode selector â”€â”€ */}
        <Text style={styles.sectionLabel}>Analysis Mode</Text>
        <View style={styles.modeRow}>
          {MODES.map(m => (
            <TouchableOpacity
              key={m.key}
              style={[styles.modeBtn, mode === m.key && styles.modeBtnActive]}
              onPress={() => switchMode(m.key)}
            >
              <Ionicons name={m.icon as any} size={16} color={mode === m.key ? Colors.white : Colors.textMuted} />
              <Text style={[styles.modeBtnText, mode === m.key && styles.modeBtnTextActive]}>{m.label}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* â”€â”€ Text input â”€â”€ */}
        {showText && (
          <>
            <Text style={styles.label}>What was said?</Text>
            <TextInput
              style={styles.textInput}
              placeholder="Enter the utterance to analyzeâ€¦"
              placeholderTextColor={Colors.textMuted}
              multiline
              numberOfLines={5}
              value={text}
              onChangeText={setText}
              textAlignVertical="top"
            />
            <Text style={styles.label}>Try an example</Text>
            {EXAMPLES.map(ex => (
              <TouchableOpacity key={ex} style={styles.exampleChip} onPress={() => setText(ex)}>
                <Text style={styles.exampleText}>{ex}</Text>
              </TouchableOpacity>
            ))}
          </>
        )}

        {/* â”€â”€ Voice recording â”€â”€ */}
        {showVoice && (
          <>
            <Text style={styles.label}>Voice Recording</Text>
            <Text style={styles.hint}>
              {mode === 'both'
                ? 'Record the same utterance aloud â€” model detects incongruence between tone and words.'
                : 'Record your voice â€” model reads vocal energy and pitch patterns.'}
            </Text>
            <View style={styles.audioRow}>
              {recordState === 'idle' && (
                <TouchableOpacity style={styles.recordBtn} onPress={startRecording}>
                  <Ionicons name="mic" size={22} color={Colors.white} />
                  <Text style={styles.recordBtnText}>Start Recording</Text>
                </TouchableOpacity>
              )}
              {recordState === 'recording' && (
                <TouchableOpacity style={[styles.recordBtn, styles.recordingActive]} onPress={stopRecording}>
                  <Ionicons name="stop" size={22} color={Colors.white} />
                  <Text style={styles.recordBtnText}>Stop Recording</Text>
                </TouchableOpacity>
              )}
              {recordState === 'done' && audioUri && (
                <View style={styles.doneRow}>
                  <Ionicons name="checkmark-circle" size={22} color={Colors.congruent} />
                  <Text style={styles.doneText}>Recording captured</Text>
                  <TouchableOpacity onPress={clearAudio} style={styles.clearBtn}>
                    <Ionicons name="close-circle" size={20} color={Colors.textMuted} />
                  </TouchableOpacity>
                </View>
              )}
            </View>
          </>
        )}

        {/* â”€â”€ Analyze button â”€â”€ */}
        <TouchableOpacity
          style={[styles.analyzeBtn, (!canSubmit || loading) && styles.analyzeBtnDisabled]}
          onPress={handleAnalyze}
          disabled={!canSubmit || loading}
        >
          {loading ? (
            <ActivityIndicator color={Colors.white} />
          ) : (
            <>
              <Ionicons name="analytics" size={20} color={Colors.white} />
              <Text style={styles.analyzeBtnText}>Analyze</Text>
            </>
          )}
        </TouchableOpacity>

        <Text style={styles.disclaimer}>
          âš  Research prototype only. Not a clinical diagnostic tool.
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const EXAMPLES = [
  "I'm fine, everything is totally fine.",
  "I honestly don't even care anymore.",
  "This situation makes me incredibly angry.",
  "I'm not sure how I feel about all this.",
];

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  content: { padding: 16, paddingBottom: 60 },
  sectionLabel: { color: Colors.textPrimary, fontWeight: '700', fontSize: 15, marginBottom: 10, marginTop: 8 },
  modeRow: { flexDirection: 'row', gap: 8, marginBottom: 4 },
  modeBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 6, paddingVertical: 10, borderRadius: 12,
    backgroundColor: Colors.bgCard, borderWidth: 1, borderColor: Colors.border,
  },
  modeBtnActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  modeBtnText: { color: Colors.textMuted, fontWeight: '600', fontSize: 12 },
  modeBtnTextActive: { color: Colors.white },
  label: { color: Colors.textPrimary, fontWeight: '700', fontSize: 15, marginBottom: 8, marginTop: 16 },
  hint: { color: Colors.textSecondary, fontSize: 12, marginBottom: 10, marginTop: -6 },
  textInput: {
    backgroundColor: Colors.bgInput, color: Colors.textPrimary,
    borderRadius: 12, borderWidth: 1, borderColor: Colors.border,
    padding: 14, fontSize: 15, minHeight: 110,
  },
  audioRow: { marginBottom: 8 },
  recordBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: Colors.primary, borderRadius: 12, paddingVertical: 14, gap: 8,
  },
  recordingActive: { backgroundColor: Colors.alert },
  recordBtnText: { color: Colors.white, fontWeight: '700', fontSize: 15 },
  doneRow: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    padding: 12, backgroundColor: Colors.bgCard, borderRadius: 12,
  },
  doneText: { color: Colors.congruent, fontWeight: '600', flex: 1 },
  clearBtn: { padding: 4 },
  exampleChip: {
    backgroundColor: Colors.bgCard, borderRadius: 10, padding: 12,
    marginBottom: 6, borderWidth: 1, borderColor: Colors.border,
  },
  exampleText: { color: Colors.textSecondary, fontSize: 13 },
  analyzeBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: Colors.primary, borderRadius: 16,
    paddingVertical: 16, marginTop: 20, gap: 8,
  },
  analyzeBtnDisabled: { opacity: 0.4 },
  analyzeBtnText: { color: Colors.white, fontWeight: '800', fontSize: 17 },
  disclaimer: { color: Colors.textMuted, fontSize: 11, textAlign: 'center', marginTop: 16, lineHeight: 16 },
});
