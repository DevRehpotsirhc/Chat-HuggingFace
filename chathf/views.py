from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse
import json
import ast

def home(request):
    return render(request, "chathf/home.html")

@csrf_exempt
def arch_router(request):
    from modelos.models_wrapper import Models_Wrapper as Models

    data = json.loads(request.body)
    message = data['message']
    uploaded_file = request.FILES.get('file')

    img_valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    text_valid_extensions = ('.txt', '.csv')

    if not uploaded_file.name.lower().endswith(img_valid_extensions + text_valid_extensions):
        return JsonResponse({"error": "Invalid file extension. Only JPG, JPEG, PNG, WEBP, TXT, and CSV are allowed."}, status=415)
    
    classified_prompt = Models.router(message)

    print("\n ______________________________________ \n RAW RESPONSE:", classified_prompt, type(classified_prompt) , "\n ______________________________________ \n ")

    try:
        response_dict = ast.literal_eval(classified_prompt)
        route = response_dict.get("route", "unknown_topic")
        print("\n ______________________________________ \n ROUTE:", route,  "\n ______________________________________ \n ")

        if route == 'unknown topic':
            return JsonResponse(
                {"message": " Ingresaste una pregunta invalida. Recuerda que puedo responder en base a temas de: clasificacion de texto, traduccion, deteccion de objetos, descripcion de imagenes y respuestas de preguntas en base a un texto"},
                status=406
            )
        else:
            
            match route:
                case 'question answering':
                    uploaded_file = request.FILES.get('file')
                    if uploaded_file is None:
                        return JsonResponse({"error": "No file was given in the request"}, status=406)
                    if uploaded_file.name.lower().endswith(text_valid_extensions):
                        return JsonResponse({"error": "File must be a TXT file."}, status=415)
                    
                    context = uploaded_file.read().decode('utf-8')

                    result = Models.QA(message, context)

                    if "error" in result:
                        return JsonResponse(result, status=406)

                    return JsonResponse(result, status=200)
                case 'translate':
                    pass
                case 'text clasification':
                    pass
                case 'object detection':
                    uploaded_file = request.FILES.get('file')
                    if uploaded_file is None:
                        return JsonResponse({"error": "No file was given in the request"}, status=406)
                    if uploaded_file.name.lower().endswith(img_valid_extensions):
                        return JsonResponse({"error": "File must be an IMG file."}, status=415)

                    result = Models.detector(uploaded_file)

                    if "error" in result:
                        return JsonResponse(result, status=406)

                    return JsonResponse(result, status=200)
                
                case 'describing an image':
                    uploaded_file = request.FILES.get('file')

                    if uploaded_file is None:
                        return JsonResponse({"error": "No file was given in the request"}, status=406)
                    if not uploaded_file.name.lower().endswith(img_valid_extensions):
                        return JsonResponse({"error": "Invalid file extension. Only JPG, JPEG, PNG, and WEBP are allowed."}, status=415)
                    
                    result = Models.describe_images(uploaded_file)

                    if "error" in result:
                        return JsonResponse(result, status=406)

                    return JsonResponse(result, status=200)

                case 'unknown topic':
                    return JsonResponse(
                        {"message": " Ingresaste una pregunta invalida. Recuerda que puedo responder en base a temas de: clasificacion de texto, traduccion, deteccion de objetos, descripcion de imagenes y respuestas de preguntas en base a un texto"},
                        status=406
                    )

    except Exception as e:
        print("Parse error:", e)
        return JsonResponse(
            {"error": "Failed to parse model output", "details": str(e)},
            status=400
        )
