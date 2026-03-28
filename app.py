from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import json
import os
import numpy as np
import glob
import requests
import shutil
import random
from werkzeug.security import generate_password_hash, check_password_hash
from functools import lru_cache
import torch
import sys
import datetime
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deep_collaborative_filtering import DeepCollaborativeFiltering

print("初始化Flask应用...")

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = 86400

IMAGE_CACHE = {}
RECOMMENDATION_CACHE = {}
DISH_DESCRIPTION_CACHE = {}

def safe_use_model(model, method_name, *args, **kwargs):
    if not model or not hasattr(model, method_name):
        print(f"模型不存在或缺少方法 {method_name}")
        return None
    
    model.trained = True
    
    method = getattr(model, method_name)
    return method(*args, **kwargs)

deep_recommender = None

def log_performance(func_name, start_time):
    import time
    end_time = time.time()
    print(f"[性能] {func_name} 耗时 {(end_time - start_time) * 1000:.2f}ms")

def load_users():
    with open('users.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        users_data = data['users']
        for user_id, user_info in users_data.items():
            user_info['password'] = generate_password_hash(user_info['password'])
        return users_data

def save_users(users_data):
    with open('users.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        original_users = original_data['users']
    
    users_to_save = {'users': {}}
    
    for user_id, user_info in users_data.items():
        if user_id in original_users:
            users_to_save['users'][user_id] = {
                'password': original_users[user_id]['password'],
                'name': user_info['name'],
                'age': user_info['age']
            }
        else:
            users_to_save['users'][user_id] = {
                'password': user_info['raw_password'],
                'name': user_info['name'],
                'age': user_info['age']
            }
    
    with open('users.json', 'w', encoding='utf-8') as f:
        json.dump(users_to_save, f, indent=2, ensure_ascii=False)
        
    return True

users = load_users()

with open('menu_en.json', 'r', encoding='utf-8') as f:
    menu_data = json.load(f)

def get_image_path(item_id, category='', subcategory=''):
    default_image = url_for('static', filename='images/default.jpg')
    
    if not item_id:
        return default_image
    
    prefix = ''.join([c for c in item_id if c.isalpha()])
    
    dish_prefixes = ['sc', 'gd', 'hn', 'sd', 'js', 'zj']
    drink_prefixes = ['t', 'j', 'sm', 'sp']
    
    if prefix in dish_prefixes:
        image_filename = f"images/dishes/{item_id}.jpg"
    elif prefix in drink_prefixes:
        image_filename = f"images/drinks/{item_id}.jpg"
    elif 'drinks' in str(category).lower() or 'tea' in str(category).lower() or 'coffee' in str(category).lower():
        if 'coffee' in str(category).lower() or 'coffee' in str(subcategory).lower():
            image_filename = f"images/drinks/coffee/{item_id}.jpg"
        elif 'fruit' in str(category).lower() or 'fruit' in str(subcategory).lower():
            image_filename = f"images/drinks/fruit_tea/{item_id}.jpg"
        elif 'milk' in str(category).lower() or 'milk' in str(subcategory).lower():
            image_filename = f"images/drinks/milk_tea/{item_id}.jpg"
        else:
            image_filename = f"images/drinks/{item_id}.jpg"
    elif 'cake' in str(category).lower() or 'cake' in str(subcategory).lower():
        image_filename = f"images/cakes/{item_id}.jpg"
    else:
        image_filename = f"images/dishes/{item_id}.jpg"
    

    image_path = os.path.join(app.static_folder, image_filename)
    if os.path.exists(image_path):
        return url_for('static', filename=image_filename)
    return default_image

@app.context_processor
def inject_users():
    return dict(users=users, get_image_path=get_image_path)

@app.route('/')
def index():
    sichuan_dishes = menu_data['dishes'].get('sichuan', [])[:3]
    cantonese_dishes = menu_data['dishes'].get('cantonese', [])[:3]
    hunan_dishes = menu_data['dishes'].get('hunan', [])[:3]
    
    tea_drinks = menu_data['drinks'].get('tea', [])[:3]
    
    popular_products = get_popular_items(6)
    
    return render_template('index.html',
                          sichuan_dishes=sichuan_dishes,
                          cantonese_dishes=cantonese_dishes,
                          hunan_dishes=hunan_dishes,
                          tea_drinks=tea_drinks,
                          popular_products=popular_products)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')
        
        if user_id in users and check_password_hash(users[user_id]['password'], password):
            session['user_id'] = user_id
            flash('Login successful! Welcome back.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again!', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        name = request.form.get('name')
        age = request.form.get('age')
        
        if user_id in users:
            flash('Username already exists!', 'danger')
        elif not user_id or not password or not name:
            flash('All fields are required!', 'danger')
        elif password != password_confirm:
            flash('Passwords do not match!', 'danger')
        else:
            users[user_id] = {
                'password': generate_password_hash(password),
                'name': name,
                'age': int(age) if age else 25,
                'raw_password': password
            }
            
            if save_users(users):
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Registration failed, please try again.', 'danger')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out successfully!', 'info')
    return redirect(url_for('login'))

@app.route('/menu')
def menu():
    import time
    start_time = time.time()
    
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    dishes_dict = menu_data['dishes']
    drinks_dict = menu_data['drinks']
    
    cuisine_names = {
        'sichuan': 'Sichuan Cuisine',
        'cantonese': 'Cantonese Cuisine',
        'hunan': 'Hunan Cuisine',
        'shandong': 'Shandong Cuisine',
        'jiangsu': 'Jiangsu Cuisine',
        'zhejiang': 'Zhejiang Cuisine'
    }
    
    drink_names = {
        'tea': 'Tea',
        'juice': 'Juice',
        'soy_milk': 'Soy Milk',
        'soup': 'Soup'
    }
    
    log_performance("menu", start_time)
    return render_template('menu.html', 
                           dishes=dishes_dict,
                           drinks=drinks_dict,
                           cuisine_names=cuisine_names,
                           drink_names=drink_names)

@lru_cache(maxsize=100)
def find_product(item_id):
    for category_key, category_items in menu_data['dishes'].items():
        for item in category_items:
            if item['id'] == item_id:
                product = item.copy()
                product['category_type'] = category_key
                product['item_type'] = 'dish'  # 标记为菜品
                return product
    
    for category_key, category_items in menu_data['drinks'].items():
        for item in category_items:
            if item['id'] == item_id:
                product = item.copy()
                product['category_type'] = category_key
                product['item_type'] = 'drink'  # 标记为饮品
                return product
    
    return None

def generate_dish_description_with_hunyuan(dish_name, dish_description):
    cache_key = f"dish_{dish_name}"
    
    if cache_key in DISH_DESCRIPTION_CACHE:
        return DISH_DESCRIPTION_CACHE[cache_key]
    
    hunyuan_api_key = '放你的密钥信息'
    os.environ['HUNYUAN_API_KEY'] = hunyuan_api_key
    
    try:
        client = OpenAI(
            api_key=os.environ.get("HUNYUAN_API_KEY"),
            base_url="https://api.hunyuan.cloud.tencent.com/v1",
        )
        
        prompt = f"""
        请为以下中国菜品生成详细的介绍和制作配方，以JSON格式输出：
        
        菜品名称：{dish_name}
        简介：{dish_description}
        
        请返回以下JSON格式，且全部使用英文输出：
        {{
            "detailed_description": "详细的菜品介绍，包括历史、特点、口味等（200字左右）",
            "ingredients": [
                {{"item": "食材名称", "amount": "数量", "unit": "单位"}},
            ],
            "cooking_steps": [
                "步骤1：详细描述",
                "步骤2：详细描述",
                "步骤3：详细描述"
            ],
            "tips": [
                "烹饪技巧1",
                "烹饪技巧2"
            ],
            "nutrition": "营养价值说明"
        }}
        
        确保返回有效的JSON格式，使用UTF-8编码。
        """
        
        completion = client.chat.completions.create(
            model="hunyuan-turbos-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            extra_body={
                "enable_enhancement": True,
            },
        )
        
        response_text = completion.choices[0].message.content
        
        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()
        
        dish_data = json.loads(json_text)
        
        DISH_DESCRIPTION_CACHE[cache_key] = dish_data
        
        return dish_data
    except Exception as e:
        print(f"混元API调用失败: {e}")
        return {
            "detailed_description": dish_description,
            "ingredients": [],
            "cooking_steps": [],
            "tips": [],
            "nutrition": "营养丰富"
        }

@app.route('/product/<item_id>')
def product_detail(item_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    product = find_product(item_id)
    
    if not product:
        flash('Product not found!', 'danger')
        return redirect(url_for('menu'))
    
    if product.get('item_type') == 'dish':
        enhanced_info = generate_dish_description_with_hunyuan(
            product['name'],
            product.get('description', '')
        )
        product['detailed_description'] = enhanced_info.get('detailed_description', product.get('description', ''))
        product['ingredients'] = enhanced_info.get('ingredients', [])
        product['cooking_steps'] = enhanced_info.get('cooking_steps', [])
        product['tips'] = enhanced_info.get('tips', [])
        product['nutrition'] = enhanced_info.get('nutrition', '')
    
    return render_template('product_detail.html', product=product)

@app.route('/cart')
def cart():
    cart_items = session.get('cart', [])
    
    total = sum(item['price'] * item['quantity'] for item in cart_items)
    
    recommended_products = []
    if 'user_id' in session:
        user_recommendations = get_user_recommendations(session['user_id'], num_recommendations=3)
        if user_recommendations:
            recommended_products = user_recommendations
        else:
            recommended_products = get_popular_items(3)
    else:
        recommended_products = get_popular_items(3)
    
    return render_template('cart.html', 
                          cart_items=cart_items, 
                          total=total, 
                          recommended_products=recommended_products)

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    item_id = request.form.get('item_id')
    quantity = int(request.form.get('quantity', 1))
    
    if not item_id:
        return jsonify({'success': False, 'message': 'Missing item ID'})
    
    if 'cart' not in session:
        session['cart'] = []
    
    product = find_product(item_id)
    
    if not product:
        return jsonify({'success': False, 'message': 'Product not found'})
    
    for item in session['cart']:
        if item['id'] == item_id:
            item['quantity'] += quantity
            session.modified = True
            return jsonify({'success': True, 'message': 'Cart quantity updated', 'cart_count': len(session['cart'])})
    
    cart_item = {
        'id': item_id,
        'name': product['name'],
        'price': product['price'],
        'quantity': quantity,
        'category': product.get('category', '')
    }
    
    session['cart'].append(cart_item)
    session.modified = True
    
    return jsonify({'success': True, 'message': 'Added to cart', 'cart_count': len(session['cart'])})

@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    data = request.get_json()
    if data and 'item_id' in data:
        item_id = data['item_id']
    else:
        item_id = request.form.get('item_id')
        
    if not item_id:
        return jsonify({'success': False, 'message': 'Missing item ID'})
    
    if 'cart' not in session:
        return jsonify({'success': False, 'message': 'Cart is empty'})
    
    session['cart'] = [item for item in session['cart'] if item['id'] != item_id]
    session.modified = True
    
    return jsonify({'success': True, 'message': 'Removed from cart'})

@app.route('/checkout', methods=['POST'])
def checkout():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if 'cart' in session and session['cart']:
        session['cart'] = []
        session.modified = True
        flash('Order submitted successfully! Thank you for your purchase.', 'success')
    else:
        flash('Cart is empty, cannot submit order.', 'warning')
    
    return redirect(url_for('index'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_info = users.get(user_id, {})
    user_name = user_info.get('name', '访客')
    user_age = user_info.get('age', '')
    
    user_orders = []
    try:
        with open('orders_en.json', 'r', encoding='utf-8') as f:
            orders_data = json.load(f)
            if isinstance(orders_data, dict) and 'orders' in orders_data:
                all_orders = orders_data['orders']
            else:
                all_orders = orders_data if isinstance(orders_data, list) else []
            
            for order in all_orders:
                if order.get('customer_id') == user_id:
                    user_orders.append(order)
            
            user_orders.sort(key=lambda x: x.get('datetime', ''), reverse=True)
    except Exception as e:
        print(f"加载订单数据失败: {e}")
        user_orders = []
    
    total_orders = len(user_orders)
    total_spent = sum(order.get('total_amount', 0) for order in user_orders)
    total_items = sum(len(order.get('items', [])) for order in user_orders)
    
    return render_template('profile.html', 
                         user_id=user_id, 
                         user_name=user_name, 
                         user_age=user_age,
                         orders=user_orders,
                         total_orders=total_orders,
                         total_spent=total_spent,
                         total_items=total_items)

@app.route('/clear-cache')
def clear_cache():
    global IMAGE_CACHE, DISH_DESCRIPTION_CACHE
    IMAGE_CACHE = {}
    DISH_DESCRIPTION_CACHE = {}
    find_product.cache_clear()
    return "缓存已清除"

def load_recommendation_model():
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"警告: 模型文件 {model_path} 不存在。")
        return None
    
    print("加载预训练的深度协同过滤模型...")
    model = DeepCollaborativeFiltering("menu_en.json", "orders_en.json")
    model.trained = True
    model.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.model.eval()
    
    print("模型加载成功，准备推荐")
    return model

def load_orders_data():
    with open('orders_en.json', 'r', encoding='utf-8') as f:
        orders_data = json.load(f)
        user_order_history = {}
        item_popularity = {}
        
        if isinstance(orders_data, dict) and 'orders' in orders_data:
            orders_data = orders_data['orders']
        
        if not isinstance(orders_data, list):
            print(f"orders_data不是列表而是 {type(orders_data)}")
            return {}, {}
            
        for order in orders_data:
            if not isinstance(order, dict):
                continue
                
            customer_id = order.get('customer_id')
            if not customer_id:
                continue
            
            if customer_id not in user_order_history:
                user_order_history[customer_id] = []
            
            items = order.get('items', [])
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                    
                user_order_history[customer_id].append(item)
                
                item_id = item.get('id')
                if item_id:
                    if item_id not in item_popularity:
                        item_popularity[item_id] = 0
                    item_popularity[item_id] += 1
        
        return user_order_history, item_popularity

def get_all_items():
    all_items = []
    
    for category_key, category_items in menu_data['dishes'].items():
        for item in category_items:
            product = item.copy()
            product['category_type'] = category_key
            all_items.append(product)
    
    for category_key, category_items in menu_data['drinks'].items():
        for item in category_items:
            product = item.copy()
            product['category_type'] = category_key
            all_items.append(product)
    
    return all_items

def get_user_recommendations(user_id, num_recommendations=6):
    cache_key = f"recom_{user_id}_{num_recommendations}"
    if cache_key in RECOMMENDATION_CACHE:
        print(f"使用缓存的推荐: {cache_key}")
        return RECOMMENDATION_CACHE[cache_key]
    
    model = load_recommendation_model()
    
    if model is None:
        print("无法加载预训练模型，使用基于热度的推荐。")
        return get_popular_items(num_recommendations)
    
    print(f"为用户 {user_id} 生成推荐")
    recommendations = safe_use_model(model, "get_user_recommendations", user_id, top_n=num_recommendations)
    
    if not recommendations:
        print(f"无法为用户 {user_id} 生成推荐，使用热门菜品。")
        return get_popular_items(num_recommendations)
    
    RECOMMENDATION_CACHE[cache_key] = recommendations
    
    return recommendations

def get_popular_items(num_items=6):
    popular_products = []
    
    all_items = get_all_items()
    
    random.shuffle(all_items)
    
    return all_items[:num_items]

@app.route('/recommendations')
def recommendations():
    if 'user_id' not in session:
        flash('Please login to get personalized recommendations', 'warning')
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    print(f"为用户 {user_id} 生成推荐...")
    
    user_order_history, _ = load_orders_data()
    
    if user_id not in user_order_history or not user_order_history[user_id]:
        error_message = "Insufficient data to generate personalized recommendations. Please make some purchases first."
        return render_template('recommendations.html', error_message=error_message)
    
    recommended_items = get_user_recommendations(user_id)
    
    if not recommended_items:
        error_message = "Sorry, unable to generate personalized recommendations. We are working to improve the system."
        return render_template('recommendations.html', error_message=error_message)
    
    return render_template('recommendations.html', recommendations=recommended_items)

@app.route('/check_login_status')
def check_login_status():
    return jsonify({
        'logged_in': 'user_id' in session
    })

if __name__ == '__main__':
    app.run(debug=True)
