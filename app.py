from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'nutriviosn_app')

nutrition_info = {
    'Apple': {'Cal': 52, 'Prot': '0.3g', 'Carb': '14g', 'Fat': '0.2g'},
    'Banana': {'Cal': 89, 'Prot': '1.1g', 'Carb': '23g', 'Fat': '0.3g'},
    'Biryani': {'Cal': 190, 'Prot': '9g', 'Carb': '25g', 'Fat': '6g'},
    'Burger': {'Cal': 250, 'Prot': '12g', 'Carb': '30g', 'Fat': '10g'},
    'Chicken_Karahi': {'Cal': 180, 'Prot': '22g', 'Carb': '3g', 'Fat': '9g'},
    'Daal_Chawal': {'Cal': 150, 'Prot': '6g', 'Carb': '28g', 'Fat': '2g'},
    'Egg': {'Cal': 143, 'Prot': '13g', 'Carb': '0.7g', 'Fat': '9.5g'},
    'Fries': {'Cal': 312, 'Prot': '3.4g', 'Carb': '41g', 'Fat': '15g'},
    'Ice cream': {'Cal': 207, 'Prot': '3.5g', 'Carb': '24g', 'Fat': '11g'},
    'Orange': {'Cal': 47, 'Prot': '0.9g', 'Carb': '12g', 'Fat': '0.1g'},
    'Pizza': {'Cal': 266, 'Prot': '11g', 'Carb': '33g', 'Fat': '10g'},
    'Shawarma': {'Cal': 230, 'Prot': '15g', 'Carb': '25g', 'Fat': '8g'},
    'tea': {'Cal': 2, 'Prot': '0.1g', 'Carb': '0.3g', 'Fat': '0g'},
    'water': {'Cal': 0, 'Prot': '0g', 'Carb': '0g', 'Fat': '0g'}
}

junk_foods = ['Pizza', 'Burger', 'Biryani', 'Fries', 'Ice cream', 'Shawarma']
food_history = [] 

try:
    model_layer = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
except Exception as e:
    model_layer = None

@app.route('/', methods=['GET', 'POST'])
def index():
    label = conf = img_path = info = full_advice = None
    active_tab = 'scanner'
    
    if request.method == 'POST':
        if 'file' in request.files:
            active_tab = 'scanner'
            file = request.files['file']
            portion = float(request.form.get('portion', 1.0))
            if file.filename != '':
                if not os.path.exists('static'): os.makedirs('static')
                path = os.path.join(BASE_DIR, 'static', 'temp.jpg')
                file.save(path)
                img = Image.open(path).convert('RGB').resize((224, 224))
                img_array = np.expand_dims(np.array(img)/255.0, axis=0).astype('float32')
                if model_layer:
                    preds = list(model_layer(img_array).values())[0]
                    label = list(nutrition_info.keys())[np.argmax(preds.numpy())]
                    conf = f"{float(np.max(preds)*100):.1f}%"
                    
                    # Nutritional Data Logic
                    base_data = nutrition_info.get(label).copy()
                    scaled_cal = round(base_data['Cal'] * portion)
                    base_data['Cal'] = scaled_cal
                    info = base_data
                    img_path = 'static/temp.jpg'
                    
                    f_type = "Junk" if label in junk_foods else "Healthy"
                    now = datetime.now().strftime("%d %b %H:%M")
                    food_history.append({'food': label, 'type': f_type, 'cal': scaled_cal, 'time': now})

        elif 'weight' in request.form:
            active_tab = 'consult'
            age = int(request.form.get('age', 20))
            if age < 18:
                plan = ["Breakfast: 🥛 Milk & Boiled Eggs", "Lunch: 🥘 Daal Chawal & Salad", "Dinner: 🍗 Grilled Chicken"]
            elif age >= 18 and age <= 50:
                plan = ["Breakfast: 🥣 Oats or Omelet", "Lunch: 🥗 Brown Rice & Lentils", "Dinner: 🍲 Light Veggie Soup"]
            else:
                plan = ["Breakfast: 🥣 Porridge & Fruit", "Lunch: 🥣 Soft Mash Daal", "Dinner: 🥦 Steamed Veggies"]

            full_advice = {
                'summary': [f"Age: {age} years", f"Weight: {request.form.get('weight')}kg"],
                'follow': ["Oats", "Leafy Greens", "Grilled Protein", "3-4 Liters Water"],
                'avoid': ["Added Sugars", "Carbonated Drinks", "Deep Fried Junk Food"],
                'diet': plan
            }

    # Calculation for DAILY Summary (Based on current session)
    h_count = sum(1 for f in food_history if f['type'] == 'Healthy')
    total_cal = sum(f['cal'] for f in food_history)
    total = len(food_history)
    progress = {
        'healthy': h_count, 'junk': total - h_count,
        'percent': int((h_count / total) * 100) if total > 0 else 0,
        'daily_cal': total_cal
    }
    return render_template('index.html', label=label, conf=conf, img_path=img_path, info=info, full_advice=full_advice, active_tab=active_tab, history=food_history, progress=progress)

if __name__ == '__main__':
    app.run(debug=True)