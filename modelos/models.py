from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse
from typing import Any, Dict, List
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import pipeline
import re
import ast

from .hf_models import HF_Models

# --------- Esto es un error de sintaxis para django, Models es un modulo de django para usar el ORM y lo estás sobreescribiendo ---------
class Models:

    @staticmethod
    def traducir(texto: str) -> str:

        tokenizer, model = HF_Models.traductor()
        detector = HF_Models.detector()

        match = re.search(r":\s*(.+)$", texto)
        contenido = match.group(1) if match else texto

        idioma_origen = detector(contenido, top_k=1)[0]["label"]

        tokenizer.src_lang = idioma_origen
        inputs = tokenizer(contenido, return_tensors="pt").to(model.device)

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id("en")
        )
        response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print("THIS IS INSIDE THE TRANSLATE FUNCTION: ___________", response)
        
        return response

    @staticmethod
    def router(message):

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

        def format_prompt(
            route_config: List[Dict[str, Any]], conversation: List[Dict[str, Any]]
        ):
            return (
                TASK_INSTRUCTION.format(
                    routes=json.dumps(route_config), conversation=json.dumps(conversation)
                )
                + FORMAT_PROMPT
            )

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
        
        tokenizer, model = HF_Models.router()

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
        )

        prompt_lengths = input_ids.shape[1]
        generated_only = [
            output_ids[prompt_lengths:]
            for output_ids in generated_ids
        ]

        response_text = tokenizer.batch_decode(generated_only, skip_special_tokens=True)[0]
        
        return response_text

    # Estas funciones las usas dentro de la clase como si fueran métodos pero no usas self, podrías usarla fuera perfectamente 
    def QA(question, context):
        if not context:
            return {"error": "No context given"}
        if not question:
            return {"error": "No question given"}

        return {"answer": HF_Models.respondedor(question, context)} #type:ignore

    def describe_images(image):
        if not image:
            return {"error": "No image given"}
        
        return {"answer": HF_Models.transcriptor(image)} #type:ignore

    def detector(image):
        if not image:
            return {"error": "No image given"}
        
        return {"answer": HF_Models.detector(image)} #type:ignore