import { useEffect, useState, useCallback } from 'react';
import {
  View, Text, StyleSheet, FlatList,
  TouchableOpacity, Alert,
} from 'react-native';
import { useRouter, useFocusEffect } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/colors';
import { loadHistory, clearHistory } from '@/store';
import { HistoryEntry } from '@/types';

export default function HistoryScreen() {
  const router = useRouter();
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useFocusEffect(
    useCallback(() => {
      loadHistory().then(setHistory);
    }, [])
  );

  function handleClear() {
    Alert.alert('Clear History', 'Delete all saved analyses?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive',
        onPress: async () => { await clearHistory(); setHistory([]); },
      },
    ]);
  }

  if (history.length === 0) {
    return (
      <View style={styles.empty}>
        <Ionicons name="time-outline" size={52} color={Colors.textMuted} />
        <Text style={styles.emptyText}>No analyses yet.</Text>
        <Text style={styles.emptyHint}>Tap Analyze to get started.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={history}
        keyExtractor={item => item.id}
        contentContainerStyle={{ padding: 16, paddingBottom: 40 }}
        ListHeaderComponent={
          <TouchableOpacity style={styles.clearBtn} onPress={handleClear}>
            <Ionicons name="trash-outline" size={16} color={Colors.alert} />
            <Text style={styles.clearText}>Clear history</Text>
          </TouchableOpacity>
        }
        renderItem={({ item }) => (
          <TouchableOpacity
            style={[
              styles.card,
              item.result.clinical_alert && styles.cardAlert,
            ]}
            onPress={() =>
              router.push({ pathname: '/results', params: { data: JSON.stringify(item.result), text: item.text } })
            }
          >
            <Text style={styles.cardText} numberOfLines={2}>{item.text}</Text>
            <View style={styles.cardMeta}>
              <View style={[styles.badge, { backgroundColor: Colors[item.result.predicted_emotion] + '33', borderColor: Colors[item.result.predicted_emotion] }]}>
                <Text style={[styles.badgeText, { color: Colors[item.result.predicted_emotion] }]}>
                  {item.result.predicted_emotion}
                </Text>
              </View>
              {item.result.clinical_alert && (
                <View style={styles.alertBadge}>
                  <Ionicons name="warning" size={12} color={Colors.alert} />
                  <Text style={styles.alertBadgeText}>Incongruent</Text>
                </View>
              )}
              <Text style={styles.timestamp}>
                {new Date(item.timestamp).toLocaleDateString()}
              </Text>
            </View>
            <Text style={styles.score}>
              δ = {item.result.incongruence_score.toFixed(3)}
            </Text>
          </TouchableOpacity>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  empty: { flex: 1, backgroundColor: Colors.bg, alignItems: 'center', justifyContent: 'center', gap: 10 },
  emptyText: { color: Colors.textSecondary, fontSize: 17, fontWeight: '600' },
  emptyHint: { color: Colors.textMuted, fontSize: 13 },
  clearBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-end', marginBottom: 12 },
  clearText: { color: Colors.alert, fontSize: 13 },
  card: { backgroundColor: Colors.bgCard, borderRadius: 14, padding: 14, marginBottom: 10, borderWidth: 1, borderColor: Colors.border },
  cardAlert: { borderColor: Colors.alert + '88' },
  cardText: { color: Colors.textPrimary, fontSize: 14, marginBottom: 8, lineHeight: 20 },
  cardMeta: { flexDirection: 'row', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
  badge: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 20, borderWidth: 1 },
  badgeText: { fontSize: 12, fontWeight: '700' },
  alertBadge: { flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: Colors.alert + '22', paddingHorizontal: 7, paddingVertical: 3, borderRadius: 20 },
  alertBadgeText: { color: Colors.alert, fontSize: 11, fontWeight: '600' },
  timestamp: { color: Colors.textMuted, fontSize: 11, marginLeft: 'auto' },
  score: { color: Colors.textMuted, fontSize: 11, marginTop: 6 },
});
