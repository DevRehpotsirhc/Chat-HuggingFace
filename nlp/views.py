from django.shortcuts import render
from transformers import pipeline

clasificador = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)

def clasificar(texto: str, etiquetas: list):
    clasificacion = clasificador(texto, candidate_labels=etiquetas)
    return clasificacion

