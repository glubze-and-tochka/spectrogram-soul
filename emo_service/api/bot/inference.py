from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import torch.nn.functional as F
from librosa import load, resample


async def predict_emotion(audio_path):
    """Take path t audio file and return label."""

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "abletobetable/spec_soul_ast"
    )
    model = ASTForAudioClassification.from_pretrained("abletobetable/spec_soul_ast")

    # from file to array
    audio, sampling_rate = load(audio_path)

    if sampling_rate != 16000:
        audio = resample(audio, target_sr=16000, orig_sr=sampling_rate)

    # audio file is decoded on the fly
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_ids = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_ids]
    
    # labels with scores
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]

    return predicted_label, outputs
