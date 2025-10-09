import os, time, requests, sys, random

ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")
assert ACCESS_KEY

DEST = "unsplash_images"
os.makedirs(DEST, exist_ok=True)

TOPICS = [
    "city street", "architecture", "skyscraper", "bridge", "subway", "abandoned building",
    "modern house", "old town", "market street", "construction site", "industrial zone",
    "rooftop view", "window reflections", "interior design", "train station", "airport",
    "parking lot", "urban night", "graffiti wall", "shopping mall",

    "mountain", "forest", "river", "waterfall", "beach", "desert", "ocean", "lake",
    "canyon", "valley", "sunrise", "sunset", "foggy morning", "stormy sky", "snowy landscape",
    "volcano", "flower field", "tree bark", "island", "coastline",

    "wildlife", "bird", "cat", "dog", "horse", "fish", "butterfly", "lion", "tiger", "elephant",
    "farm animals", "reptile", "monkey", "insect", "penguin",

    "portrait", "smiling person", "crowd", "street photography", "hands", "eyes",
    "fashion", "worker", "child", "elderly person", "couple", "family", "athlete", "artist",
    "student", "people at work", "people running", "dance", "emotion",

    "technology", "computer", "smartphone", "robot", "factory", "car", "motorcycle",
    "airplane", "ship", "satellite dish", "drone", "laboratory", "server room",
    "machine close-up", "electronic circuit", "tools", "3d printer", "camera lens",

    "food", "restaurant", "coffee cup", "fruit", "bread", "street food", "kitchen",
    "wine glass", "picnic", "market stall", "chef cooking", "ice cream", "breakfast table",

    "painting", "sculpture", "museum", "concert", "stage performance", "film set", "bookstore",
    "handmade craft", "street art", "calligraphy", "instrument close-up",

    "night sky", "city lights", "rainy day", "fog", "snow", "sunny weather", "storm clouds",
    "fire", "reflection", "shadow", "mirror", "bokeh lights", "neon signs", "minimalist interior",
    "macro texture", "abstract pattern", "glass surface", "wood texture", "metal surface"
]

def download_url(url, out_path):
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(1024*64):
            f.write(chunk)

def fetch_unsplash(query="nature", n_images=500):
    per_page = 30
    page = 1
    downloaded = 0
    headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}

    while downloaded < n_images:
        resp = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "page": page, "per_page": per_page},
            headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for r in results:
            if downloaded >= n_images:
                break
            img_url = r["urls"]["raw"] + "&w=2048"
            photo_id = r["id"]
            out = os.path.join(DEST, f"{photo_id}.jpg")
            try:
                download_url(img_url, out)
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"{downloaded} images downloaded.")
            except Exception as e:
                print("skip", photo_id, e)
                continue

        page += 1
        time.sleep(0.5)

if __name__ == "__main__":

    total_images = 200
    if len(sys.argv) > 1:
        try:
            total_images = int(sys.argv[1])
        except ValueError:
            print("Usage: python get_images.py [number_of_images]")
            sys.exit(1)

    random.shuffle(TOPICS)
    n_topics = len(TOPICS)
    images_per_topic = max(1, total_images // n_topics)

    print(f"Target: {total_images} images total")
    print(f"{n_topics} topics, -> {images_per_topic} per topic")

    downloaded_total = 0
    for topic in TOPICS:
        n_images_topic = random.randint(images_per_topic - 2, images_per_topic + 2)
        n_images_topic = max(3, n_images_topic)
        fetch_unsplash(topic, n_images=n_images_topic)
        downloaded_total += n_images_topic

    print(f"\n{downloaded_total} images downloaded in total")
