from typing import Any, Dict, List
import json
from .hf_models import HF_Models

class Models_Wrapper:
    def __init__(self):
        self._translator = None
        self._router = None
        self._transcriptor_pipeline = None
        self._detector_pipeline = None
        self._qa_pipeline = None

    @property
    def translator(self):
        if self._translator is None:
            print("Loading translator model...")
            self._translator = HF_Models.traductor()
        return self._translator

    @property
    def router_model(self):
        if self._router is None:
            print("Loading router model...")
            self._router = HF_Models.router()
        return self._router

    @property
    def transcriptor_pipeline(self):
        if self._transcriptor_pipeline is None:
            print("Loading transcriptor pipeline...")
            self._transcriptor_pipeline = HF_Models.transcriptor
        return self._transcriptor_pipeline

    @property
    def detector_pipeline(self):
        if self._detector_pipeline is None:
            print("Loading detector pipeline...")
            self._detector_pipeline = HF_Models.detector
        return self._detector_pipeline

    @property
    def qa_pipeline(self):
        if self._qa_pipeline is None:
            print("Loading QA pipeline...")
            self._qa_pipeline = HF_Models.QA
        return self._qa_pipeline

    def router(self, message):

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
        
        tokenizer_rt, model_rt = self.router_model

        input_ids = tokenizer_rt.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model_rt.device)

        generated_ids = model_rt.generate(
            input_ids=input_ids,
            max_new_tokens=512,
        )

        prompt_lengths = input_ids.shape[1]
        generated_only = [
            output_ids[prompt_lengths:]
            for output_ids in generated_ids
        ]

        response_text = tokenizer_rt.batch_decode(generated_only, skip_special_tokens=True)[0]
        
        return response_text

    def traducir(self, texto: str) -> str:
        match = re.search(r":\s*(.+)$", texto)
        contenido = match.group(1) if match else texto

        tokenizer_tr, model_tr = self.translator 

        idioma_origen = self.detector_pipeline(contenido, top_k=1)[0]["label"]

        tokenizer_tr.src_lang = idioma_origen
        inputs = tokenizer_tr(contenido, return_tensors="pt").to(model_tr.device)

        generated_tokens = model_tr.generate(
            **inputs,
            forced_bos_token_id=tokenizer_tr.get_lang_id("en")
        )
        return tokenizer_tr.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def qa(self, question, context):
        if not context:
            return {"error": "No context given"}
        if not question:
            return {"error": "No question given"}
        return {"answer": self.qa_pipeline(question, context)}

    def describe_images(self, image):
        if not image:
            return {"error": "No image given"}
        return {"answer": self.transcriptor_pipeline(image)}

    def detect(self, image):
        if not image:
            return {"error": "No image given"}
        return {"answer": self.detector_pipeline(image)}
