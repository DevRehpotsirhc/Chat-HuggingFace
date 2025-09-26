from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoModelForCausalLM, AutoTokenizer
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

extractor = pipeline(
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

# Arch
model_name_ac = "katanemo/Arch-Router-1.5B"
model_ac = AutoModelForCausalLM.from_pretrained(
    model_name_ac, device_map="auto", torch_dtype="auto", trust_remote_code=True
)
tokenizer_ac = AutoTokenizer.from_pretrained(model_name_ac)