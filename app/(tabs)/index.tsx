import { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView,
  TouchableOpacity, ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/colors';
import { checkHealth } from '@/api';
import { loadHistory } from '@/store';
import { HistoryEntry } from '@/types';

const CLUSTER_ICONS: Record<string, keyof typeof Ionicons.glyphMap> = {
  Distress: 'sad-outline',
  Resilience: 'happy-outline',
  Aggression: 'flash-outline',
  Ambiguity: 'help-circle-outline',
};

export default function HomeScreen() {
  const router = useRouter();
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [recent, setRecent] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    checkHealth().then(setApiOk);
    loadHistory().then(h => setRecent(h.slice(0, 3)));
  }, []);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Hero */}
      <LinearGradient
        colors={['#1A2A4A', '#0F1117']}
        style={styles.hero}
      >
        <Text style={styles.heroTitle}>🧠 MindSync</Text>
        <Text style={styles.heroSub}>
          Multimodal AI for{'\n'}Mental Health Monitoring
        </Text>
        <Text style={styles.heroDev}>by Abu Ubaida</Text>
        <TouchableOpacity
          style={styles.heroBtn}
          onPress={() => router.push('/analyze')}
        >
          <Ionicons name="mic" size={18} color={Colors.white} />
          <Text style={styles.heroBtnText}>Start Analysis</Text>
        </TouchableOpacity>
      </LinearGradient>

      {/* About */}
      <View style={styles.aboutCard}>
        <Ionicons name="person-circle-outline" size={22} color={Colors.primary} />
        <View style={{ flex: 1 }}>
          <Text style={styles.aboutName}>Abu Ubaida</Text>
          <Text style={styles.aboutRole}>Researcher & Developer · MindSync AI</Text>
        </View>
      </View>

      {/* API status */}
      <View style={styles.statusRow}>
        <View style={[styles.statusDot, { backgroundColor: apiOk === null ? Colors.textMuted : apiOk ? Colors.congruent : Colors.alert }]} />
        <Text style={styles.statusText}>
          {apiOk === null ? 'Connecting to server…' : apiOk ? 'Server connected' : 'Server offline — check your IP in src/api.ts'}
        </Text>
        {apiOk === null && <ActivityIndicator size="small" color={Colors.primary} style={{ marginLeft: 8 }} />}
      </View>

      {/* Clusters legend */}
      <Text style={styles.sectionTitle}>Emotion Clusters</Text>
      <View style={styles.clusterGrid}>
        {(['Distress', 'Resilience', 'Aggression', 'Ambiguity'] as const).map(c => (
          <View key={c} style={[styles.clusterCard, { borderColor: Colors[c] }]}>
            <Ionicons name={CLUSTER_ICONS[c]} size={24} color={Colors[c]} />
            <Text style={[styles.clusterLabel, { color: Colors[c] }]}>{c}</Text>
          </View>
        ))}
      </View>

      {/* Recent */}
      {recent.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Recent Analyses</Text>
          {recent.map(entry => (
            <TouchableOpacity
              key={entry.id}
              style={styles.recentCard}
              onPress={() => router.push({ pathname: '/results', params: { data: JSON.stringify(entry.result), text: entry.text } })}
            >
              <Text style={styles.recentText} numberOfLines={1}>{entry.text}</Text>
              <View style={styles.recentMeta}>
                <Text style={[styles.recentEmotion, { color: Colors[entry.result.predicted_emotion] }]}>
                  {entry.result.predicted_emotion}
                </Text>
                {entry.result.clinical_alert && (
                  <Ionicons name="warning" size={14} color={Colors.alert} style={{ marginLeft: 6 }} />
                )}
                <Text style={styles.recentTime}>
                  {new Date(entry.timestamp).toLocaleDateString()}
                </Text>
              </View>
            </TouchableOpacity>
          ))}
          <TouchableOpacity onPress={() => router.push('/history')}>
            <Text style={styles.viewAll}>View all history →</Text>
          </TouchableOpacity>
        </>
      )}

      {/* Footer credit */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>Developed by Abu Ubaida</Text>
        <Text style={styles.footerSub}>MindSync AI · Research Prototype · 2026</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  content: { padding: 16, paddingBottom: 40 },
  hero: { borderRadius: 16, padding: 28, marginBottom: 16, alignItems: 'center' },
  heroTitle: { fontSize: 32, fontWeight: '800', color: Colors.white, marginBottom: 8 },
  heroSub: { fontSize: 15, color: Colors.textSecondary, textAlign: 'center', lineHeight: 22, marginBottom: 20 },
  heroBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: Colors.primary, paddingVertical: 12, paddingHorizontal: 24, borderRadius: 30, gap: 8 },
  heroBtnText: { color: Colors.white, fontWeight: '700', fontSize: 15 },
  statusRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 20, padding: 12, backgroundColor: Colors.bgCard, borderRadius: 10 },
  statusDot: { width: 10, height: 10, borderRadius: 5, marginRight: 8 },
  statusText: { color: Colors.textSecondary, fontSize: 13, flex: 1 },
  sectionTitle: { fontSize: 17, fontWeight: '700', color: Colors.textPrimary, marginBottom: 12, marginTop: 4 },
  clusterGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 20 },
  clusterCard: { flex: 1, minWidth: '44%', backgroundColor: Colors.bgCard, borderRadius: 12, borderWidth: 1.5, padding: 16, alignItems: 'center', gap: 6 },
  clusterLabel: { fontWeight: '600', fontSize: 13 },
  recentCard: { backgroundColor: Colors.bgCard, borderRadius: 12, padding: 14, marginBottom: 8, borderWidth: 1, borderColor: Colors.border },
  recentText: { color: Colors.textPrimary, fontSize: 14, marginBottom: 6 },
  heroDev: { color: Colors.primary, fontSize: 13, fontWeight: '600', marginTop: -12, marginBottom: 16, opacity: 0.9 },
  aboutCard: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: Colors.bgCard, borderRadius: 12, padding: 14, marginBottom: 16, borderWidth: 1, borderColor: Colors.primary + '44' },
  aboutName: { color: Colors.textPrimary, fontWeight: '700', fontSize: 15 },
  aboutRole: { color: Colors.textMuted, fontSize: 12, marginTop: 2 },
  footer: { alignItems: 'center', paddingTop: 20, paddingBottom: 8, borderTopWidth: 1, borderTopColor: Colors.border, marginTop: 12 },
  footerText: { color: Colors.textSecondary, fontSize: 13, fontWeight: '600' },
  footerSub: { color: Colors.textMuted, fontSize: 11, marginTop: 4 },
  recentMeta: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  recentEmotion: { fontWeight: '700', fontSize: 13 },
  recentTime: { color: Colors.textMuted, fontSize: 12, marginLeft: 'auto' },
  viewAll: { color: Colors.primary, textAlign: 'right', marginBottom: 8, fontSize: 13 },
});
