"""
Product-specific configurations for Electronics and Beauty recommender systems
"""

PRODUCTS = {
    'Electronics': {
        'name': 'Electronics',
        'icon': '⚡',
        'data_dir': './data/Electronics',
        'model_type': 'projector',
        'categories': {
            "All Products": "All",
            "Gaming": "Gaming",
            "Storage": "SSD",
            "Keyboard": "Keyboard",
            "Mouse": "Mouse",
            "Camera": "Camera",
            "Audio": "Headphone",
            "Mobile": "Phone",
            "Cables": "Cable"
        }
    },
    
    'Beauty': {
        'name': 'Beauty',
        'icon': '💄',
        'data_dir': './data/Beauty',
        'model_type': 'two_tower',
        'categories': {
            "All Products": "All",
            "Skincare": "Skin",
            "Makeup": "Makeup",
            "Haircare": "Hair",
            "Fragrance": "Perfume",
            "Tools": "Brush",
            "Nails": "Nail",
            "Bath & Body": "Body",
            "Wellness": "Health"
        }
    }
}

UI_CONFIG = {
    'items_per_page_options': [12, 24, 48],
    'default_items_per_page': 12,
    'cols_per_row_browse': 4,
    'cols_per_row_search': 3,
    'max_recommendations': 10
}