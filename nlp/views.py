from django.shortcuts import render
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import re


clasificador = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device="cuda"
)

def clasificar(texto: str, etiquetas: list):
    clasificacion = clasificador(texto, candidate_labels=etiquetas)
    return clasificacion

respondedor = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-base-squad2",
    device="cuda"
)

def responder(pregunta: str, contexto: str):
    return respondedor(pregunta, contexto)["answer"] # type:ignore

detector = pipeline(
    "text-classification", 
    model="papluca/xlm-roberta-base-language-detection",
    device="cuda"
)

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
traductor = M2M100ForConditionalGeneration.from_pretrained(model_name)

IDIOMAS = {
    # Español
    "español": "es",
    "spanish": "es",
    "espagnol": "es",
    "spanisch": "es",
    "es": "es",

    # Inglés
    "inglés": "en",
    "english": "en",
    "anglais": "en",
    "englisch": "en",
    "en": "en",

    # Francés
    "francés": "fr",
    "french": "fr",
    "français": "fr",
    "französisch": "fr",
    "fr": "fr",

    # Alemán
    "alemán": "de",
    "german": "de",
    "deutsch": "de",
    "allemand": "de",
    "de": "de",
}

def detectar_idioma(texto: str):
    texto_lower = texto.lower()

    for nombre, codigo in IDIOMAS.items():
        patron = r"\b" + re.escape(nombre) + r"\b"
        if re.search(patron, texto_lower):
            return codigo

    return "en"

def traducir(texto: str) -> str:
    idioma_destino = detectar_idioma(texto)

    match = re.search(r":\s*(.+)$", texto)
    contenido = match.group(1) if match else texto

    idioma_origen = detector(contenido, top_k=1)[0]["label"]

    tokenizer.src_lang = idioma_origen
    inputs = tokenizer(contenido, return_tensors="pt").to(traductor.device)

    generated_tokens = traductor.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id(idioma_destino)
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]