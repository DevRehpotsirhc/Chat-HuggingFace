from django.shortcuts import render
from transformers import pipeline

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

