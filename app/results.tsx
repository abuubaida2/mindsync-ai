import { useLocalSearchParams } from 'expo-router';
import {
  View, Text, StyleSheet, ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/colors';
import { PredictResponse, EmotionCluster } from '@/types';

const CLUSTER_COLORS: Record<EmotionCluster, string> = {
  Distress: Colors.Distress,
  Resilience: Colors.Resilience,
  Aggression: Colors.Aggression,
  Ambiguity: Colors.Ambiguity,
};

export default function ResultsScreen() {
  const { data, text } = useLocalSearchParams<{ data: string; text: string }>();
  const result: PredictResponse = JSON.parse(data ?? '{}');

  const jsd = result.incongruence_score ?? 0;
  const jsdPct = Math.round(jsd * 100);
  const gaugeColor = jsd > 0.5 ? Colors.alert : jsd > 0.3 ? Colors.warning : Colors.congruent;

  // Detect analysis mode from clinical_message prefix
  const msg = result.clinical_message ?? '';
  const isTextOnly  = msg.startsWith('📝');
  const isVoiceOnly = msg.startsWith('🎙');
  const isBothMode  = !isTextOnly && !isVoiceOnly;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>

      {/* Input text */}
      {text ? (
        <View style={styles.inputCard}>
          <Text style={styles.inputLabel}>Input</Text>
          <Text style={styles.inputText}>"{text}"</Text>
        </View>
      ) : null}

      {/* Predicted emotion */}
      <View style={[styles.predCard, { borderColor: CLUSTER_COLORS[result.predicted_emotion] }]}>
        <Text style={styles.predLabel}>Predicted Emotion</Text>
        <Text style={[styles.predValue, { color: CLUSTER_COLORS[result.predicted_emotion] }]}>
          {result.predicted_emotion}
        </Text>
        <View style={styles.streamRow}>
          {!isVoiceOnly && <StreamTag label="Text" value={result.text_emotion} />}
          {!isTextOnly  && <StreamTag label="Audio" value={result.audio_emotion} />}
        </View>
      </View>

      {/* Clinical alert */}
      <View style={[styles.alertCard, result.clinical_alert ? styles.alertCardRed : styles.alertCardGreen]}>
        <Ionicons
          name={result.clinical_alert ? 'warning' : 'checkmark-circle'}
          size={22}
          color={result.clinical_alert ? Colors.alert : Colors.congruent}
        />
        <Text style={[styles.alertText, { color: result.clinical_alert ? Colors.alert : Colors.congruent }]}>
          {result.clinical_message}
        </Text>
      </View>

      {/* JSD gauge — only meaningful when both modalities present */}
      <View style={styles.gaugeCard}>
        <Text style={styles.sectionTitle}>Incongruence Score (δ)</Text>
        {isBothMode ? (
          <>
            <Text style={[styles.jsdValue, { color: gaugeColor }]}>{jsd.toFixed(4)}</Text>
            <View style={styles.gaugeTrack}>
              <View style={[styles.gaugeFill, { width: `${jsdPct}%` as any, backgroundColor: gaugeColor }]} />
            </View>
            <View style={styles.gaugeLabels}>
              <Text style={styles.gaugeLabel}>0 — Congruent</Text>
              <Text style={[styles.gaugeLabel, { color: Colors.alert }]}>1 — Incongruent</Text>
            </View>
            <View style={styles.thresholdLine} />
            <Text style={styles.thresholdText}>Threshold δ = 0.5</Text>
          </>
        ) : (
          <>
            <Text style={[styles.jsdValue, { color: Colors.textMuted }]}>N/A</Text>
            <Text style={styles.naHint}>
              {isTextOnly
                ? 'Record a voice clip alongside the text to measure cross-modal incongruence.'
                : 'Type the same utterance in text mode to measure cross-modal incongruence.'}
            </Text>
          </>
        )}
      </View>

      {/* Probability bars */}
      <ProbSection title={isBothMode ? 'Fused Probabilities' : 'Probabilities'} probs={result.fused_probabilities} />
      {!isVoiceOnly && <ProbSection title="Text Stream" probs={result.text_probabilities} />}
      {!isTextOnly  && <ProbSection title="Audio Stream" probs={result.audio_probabilities} />}
    </ScrollView>
  );
}

function StreamTag({ label, value }: { label: string; value: EmotionCluster }) {
  return (
    <View style={[styles.streamTag, { borderColor: CLUSTER_COLORS[value] }]}>
      <Text style={styles.streamTagLabel}>{label}: </Text>
      <Text style={[styles.streamTagValue, { color: CLUSTER_COLORS[value] }]}>{value}</Text>
    </View>
  );
}

function ProbSection({ title, probs }: { title: string; probs: Record<string, number> }) {
  const sorted = Object.entries(probs).sort(([, a], [, b]) => b - a);
  return (
    <View style={styles.probCard}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {sorted.map(([label, prob]) => {
        const pct = Math.round(prob * 100);
        const color = CLUSTER_COLORS[label as EmotionCluster] ?? Colors.primary;
        return (
          <View key={label} style={styles.probRow}>
            <Text style={[styles.probLabel, { color }]}>{label}</Text>
            <View style={styles.probTrack}>
              <View style={[styles.probFill, { width: `${pct}%` as any, backgroundColor: color }]} />
            </View>
            <Text style={styles.probPct}>{pct}%</Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  content: { padding: 16, paddingBottom: 40 },

  inputCard: { backgroundColor: Colors.bgCard, borderRadius: 12, padding: 14, marginBottom: 12, borderWidth: 1, borderColor: Colors.border },
  inputLabel: { color: Colors.textMuted, fontSize: 11, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 },
  inputText: { color: Colors.textSecondary, fontSize: 14, fontStyle: 'italic', lineHeight: 20 },

  predCard: { backgroundColor: Colors.bgCard, borderRadius: 16, padding: 20, marginBottom: 12, borderWidth: 2, alignItems: 'center' },
  predLabel: { color: Colors.textMuted, fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 },
  predValue: { fontSize: 32, fontWeight: '800', marginBottom: 12 },
  streamRow: { flexDirection: 'row', gap: 10 },
  streamTag: { flexDirection: 'row', borderWidth: 1, borderRadius: 20, paddingHorizontal: 10, paddingVertical: 4 },
  streamTagLabel: { color: Colors.textMuted, fontSize: 12 },
  streamTagValue: { fontWeight: '700', fontSize: 12 },

  alertCard: { borderRadius: 14, padding: 14, marginBottom: 12, borderWidth: 1.5, flexDirection: 'row', gap: 10, alignItems: 'flex-start' },
  alertCardRed: { backgroundColor: Colors.alert + '15', borderColor: Colors.alert },
  alertCardGreen: { backgroundColor: Colors.congruent + '15', borderColor: Colors.congruent },
  alertText: { fontSize: 13, flex: 1, lineHeight: 19 },

  gaugeCard: { backgroundColor: Colors.bgCard, borderRadius: 14, padding: 16, marginBottom: 12, borderWidth: 1, borderColor: Colors.border },
  sectionTitle: { color: Colors.textPrimary, fontWeight: '700', fontSize: 15, marginBottom: 10 },
  jsdValue: { fontSize: 28, fontWeight: '800', marginBottom: 10 },
  gaugeTrack: { height: 14, backgroundColor: Colors.border, borderRadius: 7, overflow: 'hidden', marginBottom: 6 },
  gaugeFill: { height: 14, borderRadius: 7 },
  gaugeLabels: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  gaugeLabel: { color: Colors.textMuted, fontSize: 11 },
  thresholdLine: { height: 1, backgroundColor: Colors.textMuted, opacity: 0.4, marginVertical: 4 },
  thresholdText: { color: Colors.textMuted, fontSize: 11, textAlign: 'center' },
  naHint: { color: Colors.textMuted, fontSize: 12, lineHeight: 18, marginTop: 4 },

  probCard: { backgroundColor: Colors.bgCard, borderRadius: 14, padding: 16, marginBottom: 12, borderWidth: 1, borderColor: Colors.border },
  probRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 8, gap: 8 },
  probLabel: { width: 90, fontWeight: '700', fontSize: 12 },
  probTrack: { flex: 1, height: 12, backgroundColor: Colors.border, borderRadius: 6, overflow: 'hidden' },
  probFill: { height: 12, borderRadius: 6 },
  probPct: { color: Colors.textMuted, fontSize: 12, width: 36, textAlign: 'right' },
});
