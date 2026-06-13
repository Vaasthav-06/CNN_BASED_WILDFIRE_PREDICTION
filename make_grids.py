import os
import math
from PIL import Image

def make_grid(image_paths, output_filename, columns=2):
    """Stitches multiple images into a single grid image to save LLM tokens."""
    if not image_paths:
        return
    
    # Open all images
    images = [Image.open(p).convert("RGB") for p in image_paths]
    img_w, img_h = images[0].size
    
    # Calculate grid size
    rows = math.ceil(len(images) / columns)
    grid_w = columns * img_w
    grid_h = rows * img_h
    
    # Create blank canvas
    grid_canvas = Image.new("RGB", (grid_w, grid_h), color="white")
    
    # Paste images
    for index, img in enumerate(images):
        img = img.resize((img_w, img_h)) 
        row = index // columns
        col = index % columns
        grid_canvas.paste(img, (col * img_w, row * img_h))
        
    grid_canvas.save(output_filename)
    print(f"✅ Saved grid: {output_filename} (Contains {len(images)} charts)")

def compress_project_visuals():
    base_dir = "outputs"
    case_studies = ["corbett", "jyotikuchi", "laisong", "similipal"]
    
    # Gather all Grad-CAMs / Regional Visuals
    viz_images = []
    for region in case_studies:
        viz_dir = os.path.join(base_dir, "case_studies", region, "visualizations")
        if os.path.exists(viz_dir):
            for f in os.listdir(viz_dir):
                if f.endswith(".png") or f.endswith(".jpg"):
                    viz_images.append(os.path.join(viz_dir, f))
    
    # Gather Comparative Analysis Charts
    comp_images = []
    comp_dir = os.path.join(base_dir, "comparative_analysis")
    if os.path.exists(comp_dir):
        for f in os.listdir(comp_dir):
            if f.endswith(".png") or f.endswith(".jpg"):
                comp_images.append(os.path.join(comp_dir, f))

    os.makedirs("outputs/summary_grids", exist_ok=True)
    
    # Create Regional Grid (4 images)
    if viz_images:
        make_grid(viz_images[:4], "outputs/summary_grids/01_Regional_Case_Studies.png", columns=2)
        
    # Create Comparative Grids (Batches of 6)
    if comp_images:
        max_images = 6
        for i in range(0, len(comp_images), max_images):
            chunk = comp_images[i : i + max_images]
            part_number = (i // max_images) + 1
            filename = f"outputs/summary_grids/02_Comparative_Metrics_Part_{part_number}.png"
            make_grid(chunk, filename, columns=2)

if __name__ == "__main__":
    compress_project_visuals()