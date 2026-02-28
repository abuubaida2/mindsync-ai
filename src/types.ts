export type EmotionCluster = 'Distress' | 'Resilience' | 'Aggression' | 'Ambiguity';

export interface PredictResponse {
  predicted_emotion: EmotionCluster;
  text_emotion: EmotionCluster;
  audio_emotion: EmotionCluster;
  fused_probabilities: Record<EmotionCluster, number>;
  text_probabilities: Record<EmotionCluster, number>;
  audio_probabilities: Record<EmotionCluster, number>;
  incongruence_score: number;
  clinical_alert: boolean;
  clinical_message: string;
}

export interface HistoryEntry {
  id: string;
  timestamp: number;
  text: string;
  result: PredictResponse;
}
