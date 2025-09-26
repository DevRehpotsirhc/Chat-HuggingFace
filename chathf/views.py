from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from typing import Any, Dict, List
import json

# ____________ MODEL LOADING ________________________
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import pipeline
import re

# Importaciones de modelos
from modelos.views import analizar, traducir, responder, contexto
from modelos.modelos import extractor, clasificador, model_ac, tokenizer_ac






#  _______________ app routing for views _________________ #
def home(request):
    return render(request, "home.html")


#  __________ app routing for various endpoints __________ #
@csrf_exempt
def arch_router(request):
    data = json.loads(request.body)
    message = data['message']
    
    print("message is right here: _______________________________________________ ", message)

    message = traducir(message) 


    # Please use our provided prompt for best performance
    TASK_INSTRUCTION = """
    You are a helpful assistant designed to find the best suited route.
    You are provided with route description within <routes></routes> XML tags:
    <routes>

    {routes}

    </routes>

    <conversation>

    {conversation}

    </conversation>
    """

    FORMAT_PROMPT = """
    Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags.  Follow the instruction:
    1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
    2. You must analyze the route descriptions and find the best match route for user latest intent. 
    3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

    Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
    {"route": "route_name"} 
    """

    # Define route config
    route_config = [
        {
            "name": "question answering",
            "description": "Answering questions about a given text",
        },
        {
            "name": "translate",
            "description": "performing translation tasks of a given text",
        },
        {
            "name": "text clasification",
            "description": "identifying the clasification of a given text",
        },
        {
            "name": "object detection",
            "description": "identifying an object from a given image",
        },
        {
            "name": "describing an image",
            "description": "describing what is being shown in a given image",
        },
        {
            "name": "unknown topic",
            "description": "identifying the user prompt as a not previously registered topic"
        }
    ]

    # Helper function to create the system prompt for our model
    def format_prompt(
        route_config: List[Dict[str, Any]], conversation: List[Dict[str, Any]]
    ):
        return (
            TASK_INSTRUCTION.format(
                routes=json.dumps(route_config), conversation=json.dumps(conversation)
            )
            + FORMAT_PROMPT
        )

    # Define conversations

    conversation = [
        {
            "role": "user",
            "content": message,
        }
    ]

    route_prompt = format_prompt(route_config, conversation)

    messages = [
        {"role": "user", "content": route_prompt},
    ]

    input_ids = tokenizer_ac.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model_ac.device)

    # 2. Generate
    generated_ids = model_ac.generate(
        input_ids=input_ids,  # or just positional: model.generate(input_ids, …)
        max_new_tokens=512,
    )

    # 3. Strip the prompt from each sequence
    prompt_lengths = input_ids.shape[1]  # same length for every row here
    generated_only = [
        output_ids[prompt_lengths:]  # slice off the prompt tokens
        for output_ids in generated_ids
    ]

    # 4. Decode if you want text
    response = tokenizer_ac.batch_decode(generated_only, skip_special_tokens=True)[0]

    print("MODEL CLASSIFYER RESPONSE: _______ \n", response)

    if "question answering" in response:
        return responder(message, contexto)
    elif "translate" in response:
        return traducir(message)
    elif "unknown topic" in response:
        return "No se pudo encontrar una petición adecuada"
    


# ___________________ TRANSALATION LOGIC ____________________________
model_name_tr = "facebook/m2m100_418M"
tokenizer_tr = M2M100Tokenizer.from_pretrained(model_name_tr)
traductor = M2M100ForConditionalGeneration.from_pretrained(model_name_tr)

def traducir(texto: str) -> str:

    match = re.search(r":\s*(.+)$", texto)
    contenido = match.group(1) if match else texto

    idioma_origen = extractor(contenido, top_k=1)[0]["label"]

    tokenizer_tr.src_lang = idioma_origen
    inputs = tokenizer_tr(contenido, return_tensors="pt").to(traductor.device)

    generated_tokens = traductor.generate(
        **inputs,
        forced_bos_token_id=tokenizer_tr.get_lang_id("en")
    )
    response = tokenizer_tr.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print("THIS IS INSIDE THE TRANSLATE FUNCTION: ___________", response)
    
    return response