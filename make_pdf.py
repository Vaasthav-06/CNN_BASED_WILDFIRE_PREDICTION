import os
from PIL import Image
import PIL.JpegImagePlugin
def create_pdf_report():
    # 1. Define where the images are located
    base_dir = "outputs"
    case_studies = ["corbett", "jyotikuchi", "laisong", "similipal"]
    
    all_images = []
    
    # 2. Gather the 16 case study visualizations (4 per region)
    for region in case_studies:
        viz_dir = os.path.join(base_dir, "case_studies", region, "visualizations")
        if os.path.exists(viz_dir):
            for file_name in os.listdir(viz_dir):
                if file_name.endswith(".png") or file_name.endswith(".jpg"):
                    all_images.append(os.path.join(viz_dir, file_name))
                    
    # 3. Gather the 28 comparative analysis images
    comp_dir = os.path.join(base_dir, "comparative_analysis")
    if os.path.exists(comp_dir):
        for file_name in os.listdir(comp_dir):
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                all_images.append(os.path.join(comp_dir, file_name))
                
    if len(all_images) == 0:
        print("No images found to compile.")
        return

    # 4. Open images and convert to RGB (required for PDF)
    opened_images = []
    for img_path in all_images:
        img = Image.open(img_path).convert("RGB")
        opened_images.append(img)
        
    # 5. Save all images as a single PDF
    output_pdf = "outputs/Full_Wildfire_Visual_Report.pdf"
    first_image = opened_images[0]
    remaining_images = opened_images[1:]
    
    first_image.save(
        output_pdf, 
        save_all=True, 
        append_images=remaining_images
    )
    
    print(f"✅ Successfully compiled {len(all_images)} images into {output_pdf}")

if __name__ == "__main__":
    create_pdf_report()