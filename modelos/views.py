from django.shortcuts import render
from .modelos import analizador, respondedor, extractor, traductor, tokenizer, clasificador
import re
from django.http import JsonResponse 

def responder(pregunta: str, contexto: str):
    respuesta = respondedor(pregunta, contexto)["answer"] # type:ignore
    return JsonResponse({"response": respuesta})

contexto = """
Laura Martínez es una reconocida abogada corporativa. Ha trabajado en múltiples casos internacionales y también imparte docencia universitaria. Su experiencia y conocimientos la han convertido en una líder en su sector.

Pedro Sánchez, por su parte, es contador y trabaja actualmente en una firma internacional. Sus especialidades incluyen auditoría financiera y optimización fiscal. También participa en conferencias para mantenerse al día.

Christopher es programador back-end con amplia experiencia en desarrollo con Django y Python.
En los últimos años, ha trabajado en proyectos relacionados con inteligencia artificial e implementación de aplicaciones en contenedores. Sus colegas elogian su capacidad para integrar modelos de IA en entornos de producción.

Finalmente, Sofía Ramírez es periodista tecnológica. Publica artículos en medios nacionales y cubre eventos de innovación. También es invitada frecuente en mesas redondas sobre el impacto de la IA en la sociedad.
"""

def analizar(texto: str):
    respuesta = analizador(texto)
    return JsonResponse({"response": respuesta})


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

def traducir(texto: str):
    idioma_destino = detectar_idioma(texto)

    match = re.search(r":\s*(.+)$", texto)
    contenido = match.group(1) if match else texto

    idioma_origen = extractor(contenido, top_k=1)[0]["label"] #type:ignore

    tokenizer.src_lang = idioma_origen
    inputs = tokenizer(contenido, return_tensors="pt").to(traductor.device)

    generated_tokens = traductor.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id(idioma_destino)
    )
    respuesta = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return JsonResponse({"response": respuesta})


def procesar_texto(request):
    resultado = None
    accion = None

    if request.method == "POST":
        texto = request.POST["texto"]

        if texto:
            etiquetas = ["hace una pregunta", "da una opinión", "pide traducir"]
            clasificacion = clasificador(texto, candidate_labels=etiquetas)
            accion = clasificacion["labels"][0] #type:ignore

            if accion == "hace una pregunta":
                resultado = responder(texto, contexto)
            elif accion == "da una opinión":
                resultado = analizar(texto)
            elif accion == "pide traducir":
                resultado = traducir(texto)

    return render(request, "procesar.html", {
        "accion": accion,
        "resultado": resultado,
    })