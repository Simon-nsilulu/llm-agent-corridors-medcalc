"""Конфигурация пилота / Configuration du pilote.

Калькуляторы с формулами вида произведения степеней: в логарифмической метрике
константа Липшица M_f = сумма модулей показателей степени (точно, глобально).
Calculateurs à formule produit de puissances : en métrique logarithmique,
M_f = somme des |exposants| (exacte, globale).
"""

SIGMA = 0.01          # масштаб логарифмической метрики: 1 % / échelle : 1 %
TAU = 0.05            # полуширина коридора C2 в ln-единицах (≈ ±5 %) / demi-largeur du couloir C2
REL_TOL = 1e-6        # допуск на округление официальных реализаций / tolérance d'arrondi

# Диапазоны правдоподобия в канонических единицах. ИЛЛЮСТРАТИВНЫЕ, НЕ КЛИНИЧЕСКИЕ.
# Plages de plausibilité en unités canoniques. ILLUSTRATIVES, NON CLINIQUES.
PARAM_KINDS = {
    "weight":      {"unit": "kg",        "range": (0.5, 350.0)},
    "height":      {"unit": "m",         "range": (0.3, 2.5)},
    "age":         {"unit": "years",     "range": (1.0, 120.0)},
    "heart_rate":  {"unit": "bpm",       "range": (20.0, 300.0)},
    "qt":          {"unit": "msec",      "range": (150.0, 800.0)},
    "glucose":     {"unit": "mg/dL",     "range": (10.0, 2000.0)},
    "insulin":     {"unit": "µIU/mL",    "range": (0.1, 1000.0)},
    "enzyme":      {"unit": "U/L",       "range": (1.0, 20000.0)},
    "platelets":   {"unit": "count/L",   "range": (1e9, 2e12)},
    "bmi":         {"unit": "kg/m^2",    "range": (10.0, 60.0)},
}

# Для каждого калькулятора: ключи Relevant Entities -> вид параметра и показатель степени.
# Pour chaque calculateur : clés de Relevant Entities -> type de paramètre et exposant.
CALCULATORS = {
    6:  {"name": "Body Mass Index (BMI)",
         "params": {"weight": ("weight", 1.0), "height": ("height", -2.0)},
         "out_range": (5.0, 150.0)},
    60: {"name": "Body Surface Area Calculator",
         "params": {"weight": ("weight", 0.5), "height": ("height", 0.5)},
         "out_range": (0.1, 4.0)},
    11: {"name": "QTc Bazett Calculator",
         "params": {"QT Interval": ("qt", 1.0), "Heart Rate or Pulse": ("heart_rate", 0.5)},
         "out_range": (200.0, 900.0)},
    31: {"name": "HOMA-IR",
         "params": {"Insulin": ("insulin", 1.0), "Glucose": ("glucose", 1.0)},
         "out_range": (0.01, 500.0)},
    19: {"name": "Fibrosis-4 (FIB-4) Index",
         "params": {"age": ("age", 1.0), "Aspartate aminotransferase": ("enzyme", 1.0),
                    "Platelet count": ("platelets", -1.0), "Alanine aminotransferase": ("enzyme", -0.5)},
         "out_range": (0.01, 200.0)},
    61: {"name": "Target weight",
         "params": {"Body Mass Index (BMI)": ("bmi", 1.0), "height": ("height", 2.0)},
         "out_range": (2.0, 300.0)},
}

# Единицы, которые агент 1 может указывать (как в наборе данных).
# Unités que l'agent 1 peut indiquer (comme dans le jeu de données).
ALLOWED_UNITS = {
    "weight": ["kg", "g", "lbs"],
    "height": ["cm", "m", "in", "ft"],
    "age": ["years", "months"],
    "heart_rate": ["beats per minute"],
    "qt": ["msec"],
    "glucose": ["mg/dL", "mmol/L"],
    "insulin": ["µIU/mL", "pmol/L", "ng/mL"],
    "enzyme": ["U/L"],
    "platelets": ["10^9/L", "10^3/µL", "µL", "mm^3", "L"],
    "bmi": ["kg/m^2"],
}


def lipschitz_M(cid):
    """M_f = sum |exponents| in log metric (all coordinates share SIGMA)."""
    return sum(abs(e) for _, e in CALCULATORS[cid]["params"].values())
