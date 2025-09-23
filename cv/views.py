from django.shortcuts import render
from transformers import pipeline
from nlp.views import traductor, tokenizer

detector = pipeline(
    "object-detection", #type:ignore
    model="facebook/detr-resnet-50",
    device="cuda"
)

def obj_img(imagen: str, objeto: str):
    respuestas = detector(imagen)
    
    for r in respuestas:
        if r["label"].lower() == objeto.lower() and r["score"] >= .6:
            return True
    
    return False

transcriptor = pipeline(
    "image-to-text", 
    model="nlpconnect/vit-gpt2-image-captioning", 
    device="cuda"
)

def traducir_texto(texto: str, idioma_des: str = "en") -> str:
    idioma_src = "en"
    tokenizer.src_lang = idioma_src

    encoded = tokenizer(texto, return_tensors="pt")

    generated_tokens = traductor.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(idioma_des)
    )

    traducido = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return traducido

def desc_img(imagen: str):
    respuesta = transcriptor(imagen)[0]["generated_text"]
    respuesta = traducir_texto(respuesta, "es")
    return respuesta