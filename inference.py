from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
from librosa import load, resample

def predict_emotion(audio_path):
  """
  """

  feature_extractor = AutoFeatureExtractor.from_pretrained("abletobetable/spec_soul_ast")
  model = ASTForAudioClassification.from_pretrained("abletobetable/spec_soul_ast")

  # from file to array
  audio, sampling_rate = load(audio_path)

  if sampling_rate != 16000:
      audio = resample(audio, target_sr = 16000, orig_sr=sampling_rate)

  # audio file is decoded on the fly
  inputs = feature_extractor(audio, 
                             sampling_rate=16000,
                             return_tensors="pt")

  with torch.no_grad():
      logits = model(**inputs).logits

  predicted_class_ids = torch.argmax(logits, dim=-1).item()
  predicted_label = model.config.id2label[predicted_class_ids]
  
  return predicted_label
