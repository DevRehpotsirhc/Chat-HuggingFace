from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

device = 0 if torch.cuda.is_available() else -1

# -------- NLP -------- #
analizador = pipeline(
    "sentiment-analysis", # type:ignore
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device
)

clasificador = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=device
)

respondedor = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-base-squad2",
    device=device
)

detector = pipeline(
    "text-classification", 
    model="papluca/xlm-roberta-base-language-detection",
    device=device
)

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
traductor = M2M100ForConditionalGeneration.from_pretrained(model_name)



# -------- CV -------- #

detector = pipeline(
    "object-detection", #type:ignore
    model="facebook/detr-resnet-50",
    device=device
)

transcriptor = pipeline(
    "image-to-text", 
    model="nlpconnect/vit-gpt2-image-captioning",
    device=device 
)