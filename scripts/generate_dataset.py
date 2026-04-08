"""
Synthetic Fashion Dataset Generator
====================================
Generates a realistic fashion dataset with:
- 5,000 fashion items across 4 categories
- 1,200 curated outfits (3-5 items each)
- Synthetic product images (color/pattern based)
- Rich text descriptions
- Simulated user interactions (50K events from 500 users)

This replaces the discontinued Polyvore dataset with a fully
open-source alternative suitable for training compatibility models.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Configuration ────────────────────────────────────────────────
NUM_ITEMS = 5000
NUM_OUTFITS = 1200
NUM_USERS = 500
NUM_INTERACTIONS = 50000
IMAGE_SIZE = (224, 224)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ─── Fashion Taxonomy ─────────────────────────────────────────────
CATEGORIES = {
    "top": {
        "subcategories": [
            "t-shirt", "blouse", "sweater", "hoodie", "polo shirt",
            "tank top", "crop top", "button-down shirt", "cardigan", "jacket"
        ],
        "weight": 0.30,
    },
    "bottom": {
        "subcategories": [
            "jeans", "chinos", "shorts", "skirt", "trousers",
            "joggers", "leggings", "cargo pants", "wide-leg pants", "mini skirt"
        ],
        "weight": 0.25,
    },
    "shoes": {
        "subcategories": [
            "sneakers", "boots", "loafers", "sandals", "heels",
            "flats", "running shoes", "oxford shoes", "slip-ons", "ankle boots"
        ],
        "weight": 0.25,
    },
    "accessory": {
        "subcategories": [
            "watch", "sunglasses", "belt", "hat", "scarf",
            "necklace", "bracelet", "backpack", "handbag", "earrings"
        ],
        "weight": 0.20,
    },
}

COLORS = {
    "black": (30, 30, 30),
    "white": (245, 245, 245),
    "navy": (0, 0, 128),
    "red": (200, 30, 30),
    "blue": (70, 130, 200),
    "green": (60, 140, 60),
    "beige": (210, 190, 160),
    "grey": (140, 140, 140),
    "brown": (139, 90, 43),
    "pink": (230, 150, 170),
    "olive": (107, 142, 35),
    "burgundy": (128, 0, 32),
}

MATERIALS = ["cotton", "polyester", "denim", "leather", "silk", "wool", "linen", "synthetic"]
PATTERNS = ["solid", "striped", "plaid", "floral", "graphic", "abstract"]
SEASONS = ["spring", "summer", "fall", "winter"]
GENDERS = ["unisex", "men", "women"]
OCCASIONS = ["casual", "formal", "sporty", "business", "party", "outdoor"]

PRICE_RANGES = {
    "top": (15, 120),
    "bottom": (20, 150),
    "shoes": (30, 200),
    "accessory": (10, 250),
}

# ─── Style Compatibility Rules ────────────────────────────────────
# Defines which combinations tend to look good together
STYLE_PROFILES = [
    {
        "name": "casual_everyday",
        "tops": ["t-shirt", "hoodie", "polo shirt", "sweater"],
        "bottoms": ["jeans", "chinos", "shorts", "joggers"],
        "shoes": ["sneakers", "slip-ons", "loafers"],
        "accessories": ["watch", "sunglasses", "backpack", "hat"],
        "colors": ["black", "white", "navy", "blue", "grey", "beige"],
        "seasons": ["spring", "summer", "fall"],
    },
    {
        "name": "formal_classic",
        "tops": ["button-down shirt", "blouse", "cardigan"],
        "bottoms": ["trousers", "chinos", "skirt"],
        "shoes": ["oxford shoes", "loafers", "heels", "flats"],
        "accessories": ["watch", "belt", "necklace", "handbag"],
        "colors": ["black", "white", "navy", "grey", "beige", "burgundy"],
        "seasons": ["fall", "winter", "spring"],
    },
    {
        "name": "streetwear",
        "tops": ["hoodie", "t-shirt", "crop top", "jacket"],
        "bottoms": ["joggers", "cargo pants", "jeans", "shorts"],
        "shoes": ["sneakers", "boots", "running shoes"],
        "accessories": ["hat", "sunglasses", "backpack", "bracelet"],
        "colors": ["black", "white", "olive", "red", "grey"],
        "seasons": ["spring", "summer", "fall"],
    },
    {
        "name": "summer_light",
        "tops": ["tank top", "crop top", "t-shirt", "blouse"],
        "bottoms": ["shorts", "mini skirt", "wide-leg pants", "leggings"],
        "shoes": ["sandals", "sneakers", "flats", "slip-ons"],
        "accessories": ["sunglasses", "hat", "earrings", "necklace"],
        "colors": ["white", "pink", "blue", "beige", "green"],
        "seasons": ["summer", "spring"],
    },
    {
        "name": "winter_cozy",
        "tops": ["sweater", "hoodie", "cardigan", "jacket"],
        "bottoms": ["jeans", "trousers", "leggings", "wide-leg pants"],
        "shoes": ["boots", "ankle boots", "sneakers"],
        "accessories": ["scarf", "hat", "watch", "belt"],
        "colors": ["brown", "burgundy", "navy", "olive", "grey", "black"],
        "seasons": ["winter", "fall"],
    },
    {
        "name": "sporty_active",
        "tops": ["tank top", "t-shirt", "hoodie", "crop top"],
        "bottoms": ["joggers", "leggings", "shorts"],
        "shoes": ["running shoes", "sneakers"],
        "accessories": ["watch", "sunglasses", "backpack", "hat"],
        "colors": ["black", "white", "blue", "red", "grey", "green"],
        "seasons": ["spring", "summer", "fall", "winter"],
    },
]

# Color compatibility matrix (colors that pair well)
COLOR_COMPATIBILITY = {
    "black": ["white", "red", "blue", "grey", "beige", "pink", "navy"],
    "white": ["black", "navy", "blue", "red", "green", "beige", "brown"],
    "navy": ["white", "beige", "grey", "red", "brown"],
    "red": ["black", "white", "navy", "grey", "beige"],
    "blue": ["white", "beige", "grey", "brown", "navy"],
    "green": ["white", "beige", "brown", "black", "navy"],
    "beige": ["navy", "brown", "black", "white", "burgundy", "olive"],
    "grey": ["black", "white", "navy", "red", "blue", "pink"],
    "brown": ["beige", "white", "navy", "green", "olive"],
    "pink": ["black", "white", "grey", "navy", "beige"],
    "olive": ["brown", "beige", "black", "white", "burgundy"],
    "burgundy": ["beige", "navy", "black", "grey", "white", "olive"],
}


def generate_synthetic_image(
    category: str, subcategory: str, color_name: str, pattern: str, item_id: int
) -> Image.Image:
    """Generate a synthetic fashion item image with category-specific shapes and patterns."""
    img = Image.new("RGB", IMAGE_SIZE, (250, 250, 250))
    draw = ImageDraw.Draw(img)
    base_color = COLORS[color_name]

    # Category-specific silhouettes
    if category == "top":
        # T-shape silhouette
        draw.rectangle([62, 30, 162, 180], fill=base_color, outline=(0, 0, 0), width=2)
        draw.rectangle([20, 30, 62, 100], fill=base_color, outline=(0, 0, 0), width=2)   # left sleeve
        draw.rectangle([162, 30, 204, 100], fill=base_color, outline=(0, 0, 0), width=2)  # right sleeve
        draw.arc([90, 15, 134, 50], 0, 360, fill=(0, 0, 0), width=2)  # neckline
    elif category == "bottom":
        # Pants/skirt silhouette
        draw.rectangle([62, 20, 162, 90], fill=base_color, outline=(0, 0, 0), width=2)   # waist
        draw.rectangle([62, 90, 108, 200], fill=base_color, outline=(0, 0, 0), width=2)   # left leg
        draw.rectangle([116, 90, 162, 200], fill=base_color, outline=(0, 0, 0), width=2)  # right leg
    elif category == "shoes":
        # Shoe silhouette
        draw.ellipse([40, 80, 184, 160], fill=base_color, outline=(0, 0, 0), width=2)
        draw.rectangle([40, 120, 184, 155], fill=base_color, outline=(0, 0, 0), width=2)
        draw.rectangle([30, 155, 194, 170], fill=(80, 80, 80), outline=(0, 0, 0), width=2)  # sole
    else:  # accessory
        # Circle/rectangle for accessories
        draw.ellipse([52, 42, 172, 182], fill=base_color, outline=(0, 0, 0), width=2)

    # Add pattern overlay
    if pattern == "striped":
        for y in range(0, IMAGE_SIZE[1], 16):
            draw.line([(0, y), (IMAGE_SIZE[0], y)], fill=(255, 255, 255, 80), width=2)
    elif pattern == "plaid":
        for y in range(0, IMAGE_SIZE[1], 20):
            draw.line([(0, y), (IMAGE_SIZE[0], y)], fill=(255, 255, 255, 60), width=1)
        for x in range(0, IMAGE_SIZE[0], 20):
            draw.line([(x, 0), (x, IMAGE_SIZE[1])], fill=(255, 255, 255, 60), width=1)
    elif pattern == "floral":
        for _ in range(8):
            cx = random.randint(50, 174)
            cy = random.randint(50, 174)
            r = random.randint(5, 12)
            petal_color = tuple(min(255, c + 60) for c in base_color)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=petal_color)
    elif pattern == "graphic":
        # Simple geometric shapes
        draw.polygon([(112, 60), (145, 130), (79, 130)], outline=(255, 255, 255), width=2)

    # Add subtle text label
    try:
        font = ImageFont.load_default()
        draw.text((10, 205), f"{subcategory}", fill=(100, 100, 100), font=font)
    except Exception:
        pass

    return img


def generate_description(category: str, subcategory: str, color: str, material: str,
                         pattern: str, season: str, occasion: str, gender: str) -> str:
    """Generate a realistic product description."""
    templates = [
        f"A stylish {color} {material} {subcategory} perfect for {occasion} wear. "
        f"Features a {pattern} design ideal for the {season} season.",

        f"This {gender}'s {color} {subcategory} is crafted from premium {material}. "
        f"The {pattern} pattern makes it a versatile choice for {occasion} occasions.",

        f"Elevate your {season} wardrobe with this {color} {material} {subcategory}. "
        f"Designed for {occasion} settings with a modern {pattern} finish.",

        f"Classic {color} {subcategory} made from comfortable {material}. "
        f"A {season} essential with {pattern} detailing for {occasion} style.",
    ]
    return random.choice(templates)


def generate_items(output_dir: Path) -> list[dict]:
    """Generate all fashion items with images and metadata."""
    items = []
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    item_id = 0
    for category, info in CATEGORIES.items():
        n_items = int(NUM_ITEMS * info["weight"])
        for _ in range(n_items):
            subcategory = random.choice(info["subcategories"])
            color = random.choice(list(COLORS.keys()))
            material = random.choice(MATERIALS)
            pattern = random.choice(PATTERNS)
            season = random.choice(SEASONS)
            gender = random.choice(GENDERS)
            occasion = random.choice(OCCASIONS)
            price_min, price_max = PRICE_RANGES[category]
            price = round(random.uniform(price_min, price_max), 2)

            # Generate and save image
            img = generate_synthetic_image(category, subcategory, color, pattern, item_id)
            img_path = f"images/{category}_{item_id:05d}.png"
            img.save(image_dir / f"{category}_{item_id:05d}.png")

            description = generate_description(
                category, subcategory, color, material, pattern, season, occasion, gender
            )

            item = {
                "item_id": item_id,
                "category": category,
                "subcategory": subcategory,
                "color": color,
                "material": material,
                "pattern": pattern,
                "season": season,
                "gender": gender,
                "occasion": occasion,
                "price": price,
                "image_path": img_path,
                "description": description,
            }
            items.append(item)
            item_id += 1

    return items


def generate_outfits(items: list[dict]) -> list[dict]:
    """Generate outfit compositions using style compatibility rules."""
    items_by_cat = {}
    for item in items:
        items_by_cat.setdefault(item["category"], []).append(item)

    outfits = []
    for outfit_id in range(NUM_OUTFITS):
        profile = random.choice(STYLE_PROFILES)

        outfit_items = []
        # Pick one item from each category based on style profile
        for cat, subcat_key, acc_key in [
            ("top", "tops", None),
            ("bottom", "bottoms", None),
            ("shoes", "shoes", None),
            ("accessory", None, "accessories"),
        ]:
            valid_subcats = profile.get(subcat_key or acc_key, [])
            valid_colors = profile.get("colors", list(COLORS.keys()))

            candidates = [
                i for i in items_by_cat[cat]
                if i["subcategory"] in valid_subcats and i["color"] in valid_colors
            ]

            if not candidates:
                candidates = items_by_cat[cat]

            chosen = random.choice(candidates)
            outfit_items.append(chosen["item_id"])

        # Optionally add a second accessory (30% chance)
        if random.random() < 0.3:
            extra = random.choice(items_by_cat["accessory"])
            if extra["item_id"] not in outfit_items:
                outfit_items.append(extra["item_id"])

        outfits.append({
            "outfit_id": outfit_id,
            "item_ids": outfit_items,
            "style": profile["name"],
            "season": random.choice(profile.get("seasons", SEASONS)),
        })

    return outfits


def generate_user_interactions(items: list[dict], outfits: list[dict]) -> tuple[list, list]:
    """Generate simulated user interaction data."""
    users = []
    for user_id in range(NUM_USERS):
        preferred_styles = random.sample(
            [p["name"] for p in STYLE_PROFILES],
            k=random.randint(1, 3)
        )
        preferred_colors = random.sample(list(COLORS.keys()), k=random.randint(3, 6))

        users.append({
            "user_id": user_id,
            "preferred_styles": preferred_styles,
            "preferred_colors": preferred_colors,
            "gender": random.choice(GENDERS),
            "age_group": random.choice(["18-24", "25-34", "35-44", "45+"]),
        })

    interaction_types = ["view", "click", "add_to_cart", "purchase", "save"]
    interaction_weights = [0.50, 0.25, 0.12, 0.08, 0.05]

    interactions = []
    for _ in range(NUM_INTERACTIONS):
        user = random.choice(users)
        item = random.choice(items)
        interaction_type = random.choices(interaction_types, weights=interaction_weights, k=1)[0]

        # Boost probability for items matching user preferences
        score = 1.0
        if item["color"] in user["preferred_colors"]:
            score += 0.5

        # Generate timestamp (last 180 days)
        days_ago = random.randint(0, 180)
        timestamp = f"2026-{random.randint(1, 4):02d}-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z"

        interactions.append({
            "user_id": user["user_id"],
            "item_id": item["item_id"],
            "interaction_type": interaction_type,
            "timestamp": timestamp,
            "score": score,
        })

    return users, interactions


def generate_train_val_test_splits(outfits: list[dict], output_dir: Path):
    """Split outfits into train/val/test at outfit level (no data leakage)."""
    random.shuffle(outfits)
    n = len(outfits)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": outfits[:train_end],
        "val": outfits[train_end:val_end],
        "test": outfits[val_end:],
    }

    for split_name, split_outfits in splits.items():
        with open(output_dir / f"outfits_{split_name}.json", "w") as f:
            json.dump(split_outfits, f, indent=2)
        print(f"  {split_name}: {len(split_outfits)} outfits")


def main():
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Fashion Dataset Generator")
    print("=" * 60)

    print(f"\n[1/5] Generating {NUM_ITEMS} fashion items with images...")
    items = generate_items(output_dir)
    with open(output_dir / "items.json", "w") as f:
        json.dump(items, f, indent=2)
    print(f"  Created {len(items)} items across {len(CATEGORIES)} categories")

    print(f"\n[2/5] Generating {NUM_OUTFITS} outfit compositions...")
    outfits = generate_outfits(items)
    with open(output_dir / "outfits.json", "w") as f:
        json.dump(outfits, f, indent=2)
    print(f"  Created {len(outfits)} outfits with style-aware pairing")

    print(f"\n[3/5] Generating user interactions ({NUM_USERS} users, {NUM_INTERACTIONS} events)...")
    users, interactions = generate_user_interactions(items, outfits)
    with open(output_dir / "users.json", "w") as f:
        json.dump(users, f, indent=2)
    with open(output_dir / "interactions.json", "w") as f:
        json.dump(interactions, f, indent=2)
    print(f"  Created {len(users)} user profiles and {len(interactions)} interactions")

    print("\n[4/5] Generating compatibility pairs...")
    # Generate positive and negative pairs for training
    positive_pairs = []
    for outfit in outfits:
        ids = outfit["item_ids"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                positive_pairs.append({
                    "item_a": ids[i],
                    "item_b": ids[j],
                    "compatible": True,
                    "outfit_id": outfit["outfit_id"],
                })

    # Generate hard negatives
    item_by_id = {item["item_id"]: item for item in items}
    items_by_cat = {}
    for item in items:
        items_by_cat.setdefault(item["category"], []).append(item["item_id"])

    negative_pairs = []
    for pair in positive_pairs[:len(positive_pairs)]:
        item_b = item_by_id[pair["item_b"]]
        cat = item_b["category"]
        neg_id = random.choice(items_by_cat[cat])
        while neg_id == pair["item_b"]:
            neg_id = random.choice(items_by_cat[cat])
        negative_pairs.append({
            "item_a": pair["item_a"],
            "item_b": neg_id,
            "compatible": False,
            "outfit_id": pair["outfit_id"],
        })

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    with open(output_dir / "compatibility_pairs.json", "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"  Created {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs")

    print("\n[5/5] Creating train/val/test splits...")
    generate_train_val_test_splits(outfits, output_dir)

    # Dataset statistics
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    cat_counts = {}
    for item in items:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:12s}: {count:5d} items")
    print(f"  {'TOTAL':12s}: {len(items):5d} items")
    print(f"  Outfits:      {len(outfits)}")
    print(f"  Users:        {len(users)}")
    print(f"  Interactions: {len(interactions)}")
    print(f"  Pairs:        {len(all_pairs)}")
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
