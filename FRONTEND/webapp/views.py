from django.shortcuts import render, redirect
import os
import numpy as np
import pandas as pd
import pickle
from django.conf import settings
import xgboost as xgb
import json
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages

# ── PATH SETUP ────────────────────────────────────────────────
FRONTEND_DIR = settings.BASE_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(FRONTEND_DIR, ".."))

HEART_MODEL_PATH   = os.path.join(PROJECT_ROOT, "heart_model.pkl")
HEART_ENCODER_PATH = os.path.join(PROJECT_ROOT, "heart_encoder.pkl")
HEART_JSON_PATH    = os.path.join(PROJECT_ROOT, "heart_knowledge.json")

if os.path.exists(HEART_JSON_PATH):
    with open(HEART_JSON_PATH, "r") as f:
        heart_knowledge = json.load(f)
else:
    heart_knowledge = {}

DATASET_DIR        = os.path.join(PROJECT_ROOT, "DATASET")
BLOOD_MODEL_PATH   = os.path.join(FRONTEND_DIR, "blood_disease_model.pkl")
BLOOD_ENCODER_PATH = os.path.join(FRONTEND_DIR, "blood_disease_encoder.pkl")

MEDICAL_JSON_PATH  = os.path.join(PROJECT_ROOT, "medical_knowledge.json")
if os.path.exists(MEDICAL_JSON_PATH):
    with open(MEDICAL_JSON_PATH, "r") as f:
        medical_knowledge = json.load(f)
else:
    medical_knowledge = {}

# ── LOAD MAIN DISEASE MODEL ───────────────────────────────────
_native_candidates = [
    os.path.join(FRONTEND_DIR, "xgboost.json"),
    os.path.join(FRONTEND_DIR, "xgboost.model"),
    os.path.join(DATASET_DIR,  "xgboost.json"),
    os.path.join(DATASET_DIR,  "xgboost.model"),
]
_native_path = next((p for p in _native_candidates if os.path.exists(p)), None)

if _native_path:
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier()
        xgb_model.load_model(_native_path)
    except Exception:
        booster = xgb.Booster()
        booster.load_model(_native_path)

        class BoosterWrapper:
            def __init__(self, b): self.booster = b
            def predict(self, X):
                dm   = xgb.DMatrix(np.array(X))
                preds = self.booster.predict(dm)
                return np.argmax(preds, axis=1)

        xgb_model = BoosterWrapper(booster)
else:
    xgb_model = pickle.load(open(os.path.join(DATASET_DIR, "xgboost.pkl"), "rb"))

# ── SUPPORT DATA ──────────────────────────────────────────────
description  = pd.read_csv(os.path.join(DATASET_DIR, "description.csv"))
precautions  = pd.read_csv(os.path.join(DATASET_DIR, "precautions_df.csv"))
medications  = pd.read_csv(os.path.join(DATASET_DIR, "medications.csv"))
diets        = pd.read_csv(os.path.join(DATASET_DIR, "diets.csv"))
workout      = pd.read_csv(os.path.join(DATASET_DIR, "workout_df.csv"))

_train_df    = pd.read_csv(os.path.join(DATASET_DIR, "Training.csv"), nrows=0)
feature_cols = list(_train_df.columns[:-1])
symptoms_dict = {c.strip().replace(" ", "_"): i for i, c in enumerate(feature_cols)}

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(pd.read_csv(os.path.join(DATASET_DIR, "Training.csv"))["prognosis"])
diseases_list = {i: d for i, d in enumerate(le.classes_)}


# ── HELPERS ───────────────────────────────────────────────────
def helper(dis):
    desc = description[description["Disease"] == dis]["Description"].values[0]
    pre  = precautions[precautions["Disease"] == dis].iloc[0, 1:].tolist()
    med  = medications[medications["Disease"] == dis]["Medication"].tolist()
    die  = diets[diets["Disease"] == dis]["Diet"].tolist()
    wrk  = workout[workout["disease"] == dis]["workout"].tolist()
    return desc, pre, med, die, wrk

def safe_float(v, d=0.0):
    try: return float(v)
    except: return d

def safe_int(v, d=0):
    try: return int(float(v))
    except: return d

def range_to_avg(value, default=0):
    try:
        if "-" in value:
            low, high = value.split("-")
            return (float(low) + float(high)) / 2
        return float(value)
    except: return default

def process_input(symptoms):
    vec = np.zeros(len(symptoms_dict))
    for s in symptoms.split(","):
        s = s.strip()
        if s in symptoms_dict:
            vec[symptoms_dict[s]] = 1
    return vec

def get_predicted_value(symptoms):
    vec = process_input(symptoms)
    return diseases_list[xgb_model.predict([vec])[0]]


# ══════════════════════════════════════════════════════════════
#  VIEWS
# ══════════════════════════════════════════════════════════════

def home(request):
    """
    Login page.
    Only users who have registered (exist in DB) can log in.
    """
    if request.method == "POST":
        username = request.POST.get("name", "").strip()
        password = request.POST.get("password", "")

        if not username:
            messages.error(request, "Please enter your username.")
            return render(request, "index.html")

        if not password:
            messages.error(request, "Please enter your password.")
            return render(request, "index.html")

        # ✅ Only registered users will authenticate successfully
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("input")          # go to main app
        else:
            # Could be wrong password OR user doesn't exist — same message for security
            messages.error(request, "Invalid username or password. Please register first if you don't have an account.")
            return render(request, "index.html")

    return render(request, "index.html")


def input(request):
    """Symptom input page — only accessible after login."""
    if not request.user.is_authenticated:
        messages.error(request, "Please login to access this page.")
        return redirect("home")

    return render(request, "input.html")


def register_view(request):
    """
    Register a new user.
    On success → redirect to register_success page.
    """
    if request.method == "POST":
        username = request.POST.get("name", "").strip()
        password = request.POST.get("password", "")

        if not username:
            return render(request, "register.html", {"error": "Username cannot be empty."})

        if not password:
            return render(request, "register.html", {"error": "Password cannot be empty."})

        if len(password) < 6:
            return render(request, "register.html", {"error": "Password must be at least 6 characters."})

        if User.objects.filter(username=username).exists():
            return render(request, "register.html", {"error": "Username already taken. Please choose another."})

        User.objects.create_user(username=username, password=password)

        # ✅ Redirect to attractive success page, pass username for personalisation
        return redirect(f"/register-success/?username={username}")

    return render(request, "register.html")


def register_success(request):
    """Attractive post-registration success / redirect page."""
    username = request.GET.get("username", "there")
    return render(request, "register_success.html", {"username": username})


# ── PREDICTION VIEWS (unchanged) ─────────────────────────────

def output(request):
    if not request.user.is_authenticated:
        return redirect("home")

    if request.method != "POST":
        return render(request, "input.html")

    mode         = request.POST.get("mode")
    symptoms_raw = request.POST.get("symptoms", "")

    # BLOOD
    if mode == "blood":
        blood_model   = pickle.load(open(BLOOD_MODEL_PATH,   "rb"))
        blood_encoder = pickle.load(open(BLOOD_ENCODER_PATH, "rb"))

        X = np.array([[
            safe_int(request.POST.get("age")),
            0 if request.POST.get("gender") == "male" else 1,
            range_to_avg(request.POST.get("temperature", "0-0")),
            range_to_avg(request.POST.get("pulse",       "0-0")),
            range_to_avg(request.POST.get("bp_sys",      "0-0")),
            range_to_avg(request.POST.get("bp_dia",      "0-0")),
            range_to_avg(request.POST.get("spo2",        "0-0")),
            range_to_avg(request.POST.get("hemoglobin",  "0-0")),
            range_to_avg(request.POST.get("wbc",         "0-0")),
            range_to_avg(request.POST.get("platelets",   "0-0")),
            range_to_avg(request.POST.get("rbc",         "0-0")),
            range_to_avg(request.POST.get("esr",         "0-0")),
            1 if request.POST.get("crp")    == "High"     else 0,
            1 if request.POST.get("dengue") == "Positive" else 0,
            1 if request.POST.get("malaria")== "Positive" else 0,
            1 if request.POST.get("widal")  == "Positive" else 0,
        ]])

        disease = blood_encoder.inverse_transform(blood_model.predict(X))[0]
        details = medical_knowledge.get(disease, {})

        return render(request, "output.html", {
            "mode": "blood",
            "predicted_disease": disease,
            "description":  details.get("detailed_description", []),
            "cause":        details.get("cause",       []),
            "treatment":    details.get("treatment",   []),
            "medicines":    details.get("medicines",   []),
            "diet":         details.get("diet",        []),
            "workout":      details.get("workout",     []),
            "precautions":  details.get("precautions", []),
        })

    # HEART
    if mode == "heart":
        heart_model   = pickle.load(open(HEART_MODEL_PATH,   "rb"))
        heart_encoder = pickle.load(open(HEART_ENCODER_PATH, "rb"))

        X = np.array([[
            1 if request.POST.get("Gender") == "Male" else 0,
            safe_float(request.POST.get("age")),
            safe_float(request.POST.get("education")),
            1 if request.POST.get("currentSmoker")   == "yes" else 0,
            safe_float(request.POST.get("cigsPerDay")),
            1 if request.POST.get("BPMeds")          == "yes" else 0,
            1 if request.POST.get("prevalentStroke") == "yes" else 0,
            1 if request.POST.get("prevalentHyp")    == "yes" else 0,
            1 if request.POST.get("diabetes")        == "yes" else 0,
            safe_float(request.POST.get("totChol")),
            safe_float(request.POST.get("sysBP")),
            safe_float(request.POST.get("diaBP")),
            safe_float(request.POST.get("BMI")),
            safe_float(request.POST.get("heartRate")),
            safe_float(request.POST.get("glucose")),
        ]])

        pred    = heart_model.predict(X)
        disease = heart_encoder.inverse_transform(pred)[0]
        details = heart_knowledge.get(disease, {})

        return render(request, "output.html", {
            "mode": "heart",
            "predicted_disease": disease,
            "description":  details.get("description", []),
            "cause":        details.get("cause",       []),
            "treatment":    details.get("treatment",   []),
            "medicines":    details.get("medicines",   []),
            "diet":         details.get("diet",        []),
            "workout":      details.get("workout",     []),
            "precautions":  details.get("precautions", []),
        })

    # FEVER
    if mode == "fever":
        model       = pickle.load(open(os.path.join(FRONTEND_DIR, "fever_model.pkl"),      "rb"))
        med_encoder = pickle.load(open(os.path.join(FRONTEND_DIR, "medicine_encoder.pkl"), "rb"))

        X = [[
            safe_int(request.POST.get("age")),
            0 if request.POST.get("gender") == "male" else 1,
            safe_float(request.POST.get("bmi")),
            0, 0,
            1 if request.POST.get("headache")   == "yes" else 0,
            1 if request.POST.get("body_ache")  == "yes" else 0,
            1 if request.POST.get("fatigue")    == "yes" else 0,
            3 if request.POST.get("temperature") >= "39" else 2,
        ]]

        pred     = model.predict(X)[0]
        medicine = med_encoder.inverse_transform([pred])[0]

        return render(request, "output.html", {
            "mode": "fever",
            "predicted_disease": medicine,
            "description": "Recommended medicine based on fever condition",
        })

    # NORMAL (symptom-based)
    if symptoms_raw:
        disease = get_predicted_value(symptoms_raw)
        desc, pre, med, die, wrk = helper(disease)

        return render(request, "output.html", {
            "mode": "normal",
            "predicted_disease": disease,
            "description":  desc,
            "precautions":  pre,
            "medications":  med,
            "diet":         die,
            "workout":      wrk,
        })

    return render(request, "input.html")


def blood_input(request):
    if not request.user.is_authenticated:
        return redirect("home")
    return render(request, "blood_input.html")


def heart_input(request):
    if not request.user.is_authenticated:
        return redirect("home")
    return render(request, "heart_input.html", {
        "range_age":    range(30, 81),
        "range_cigs":   range(0, 41),
        "range_chol":   range(100, 401, 10),
        "range_sysBP":  range(90, 201, 5),
        "range_diaBP":  range(60, 131, 5),
        "range_bmi":    range(15, 46),
        "range_hr":     range(60, 111),
        "range_glucose":range(60, 201, 5),
    })