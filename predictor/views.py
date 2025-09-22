import json
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .onnx_utils import HeartONNX, FEATURE_ORDER, FEATURE_SCHEMA, validate_features, label_text, recommendation_for


model = HeartONNX()


def index(request: HttpRequest) -> HttpResponse:
    fields = [{"name": n, "sch": FEATURE_SCHEMA[n]} for n in FEATURE_ORDER]
    return render(request, "index.html", {"fields": fields})


@csrf_exempt
def api_predict(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Malformed JSON"}, status=400)
    try:
        feats, errors = validate_features(data)
        if errors:
            return JsonResponse({"error": "Validation failed", "details": errors}, status=400)
        y, p = model.predict(feats)
        return JsonResponse({
            "label": label_text(y),
            "confidence": round(p*100.0, 1),
            "recommendation": recommendation_for(p),
        })
    except FileNotFoundError as e:
        return JsonResponse({"error": "Model not found", "details": str(e)}, status=500)
    except Exception as e:
        import traceback
        return JsonResponse({
            "error": "Prediction failed",
            "details": str(e),
            "trace": traceback.format_exc(limit=1)
        }, status=500)
