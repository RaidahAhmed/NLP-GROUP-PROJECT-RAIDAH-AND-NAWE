import gradio as gr
from transformers import pipeline

MODEL_ID = "RaidahAhmed/RaidahandNawe-sentiment-model"
clf = pipeline("text-classification", model=MODEL_ID)

def predict(text):
    if not text.strip():
        return {}
    r = clf(text)[0]
    other = "NEGATIVE" if r["label"] == "POSITIVE" else "POSITIVE"
    return {r["label"]: r["score"], other: 1 - r["score"]}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Type a sentence..."),
    outputs=gr.Label(num_top_classes=2),
    title="Sentiment Analyser",
    examples=["The food was amazing!", "Worst service ever."],
)
demo.launch(
