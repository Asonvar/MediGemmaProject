import json

# Target the specific file the user is currently looking at
file_path = '/Users/krishnaagrawal/Desktop/MediGemmaProject/MedGemmaProject (1).ipynb'

# Update Step 3 to accept Base64 string directly in a Textbox, bypassing Gradio's internal file upload router completely.
DEBUG_CELL_SOURCE = [
    "import gradio as gr\n",
    "import nest_asyncio\n",
    "from fastapi.middleware.cors import CORSMiddleware\n",
    "from PIL import Image, ImageDraw\n",
    "import traceback\n",
    "import base64\n",
    "from io import BytesIO\n",
    "\n",
    "nest_asyncio.apply()\n",
    "\n",
    "def analyze_medical_scan(image_b64, text_prompt):\n",
    "    print(\"🚀 ANALYSIS REQUEST RECEIVED\")\n",
    "    try:\n",
    "        if not image_b64 or image_b64 == '': \n",
    "            print(\"❌ No image data received\")\n",
    "            return None, \"No image uploaded.\"\n",
    "        \n",
    "        print(f\"✅ Base64 String Received (Length: {len(image_b64)})\")\n",
    "        \n",
    "        # Decode Base64 to PIL Image\n",
    "        if \",\" in image_b64:\n",
    "            image_b64 = image_b64.split(\",\")[1]\n",
    "        image_data = base64.b64decode(image_b64)\n",
    "        image_rgb = Image.open(BytesIO(image_data)).convert(\"RGB\")\n",
    "        print(f\"📸 Decoded Image Size: {image_rgb.size}\")\n",
    "        \n",
    "        # 1. MedGemma Diagnosis\n",
    "        print(\"🧠 Running MedGemma diagnosis...\")\n",
    "        messages = [{\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": image_rgb}, {\"type\": \"text\", \"text\": text_prompt}]}]\n",
    "        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(model.device, dtype=torch.float16)\n",
    "        \n",
    "        with torch.no_grad():\n",
    "            outputs = model.generate(**inputs, max_new_tokens=512)\n",
    "        \n",
    "        report = processor.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)\n",
    "        print(\"✅ Report generated\")\n",
    "\n",
    "        # 2. OWL-ViT Extraction\n",
    "        print(\"🎯 Running OWL-ViT grounding...\")\n",
    "        target = \"abnormality\" \n",
    "        for keyword in [\"fracture\", \"pneumonia\", \"tumor\", \"lesion\", \"nodule\"]: \n",
    "            if keyword in report.lower(): target = keyword; break\n",
    "            \n",
    "        predictions = detector(image_rgb, candidate_labels=[target])\n",
    "        \n",
    "        annotated = image_rgb.copy()\n",
    "        draw = ImageDraw.Draw(annotated)\n",
    "        for pred in predictions:\n",
    "            if pred['score'] > 0.05:\n",
    "                box = pred['box']\n",
    "                draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline='red', width=4)\n",
    "                draw.text((box['xmin'], box['ymin']-10), f\"{pred['label']} ({round(pred['score'],2)})\", fill='red')\n",
    "        \n",
    "        print(\"✨ Analysis COMPLETE\")\n",
    "        return annotated, report\n",
    "    except Exception as e:\n",
    "        err_msg = traceback.format_exc()\n",
    "        print(f\"❌ ERROR: {err_msg}\")\n",
    "        return None, f\"Backend Error: {str(e)}\"\n",
    "\n",
    "with gr.Blocks() as demo:\n",
    "    gr.Markdown(\"# MediGemma-X Base64 Backend\")\n",
    "    with gr.Row():\n",
    "        # CRITICAL: Receive Base64 as a Textbox so Gradio doesn't try to parse it as a file\n",
    "        img_in = gr.Textbox(label=\"Base64 Input (Hide in UI)\", visible=True)\n",
    "        txt_in = gr.Textbox()\n",
    "    with gr.Row():\n",
    "        img_out = gr.Image(type='pil')\n",
    "        txt_out = gr.Textbox()\n",
    "    btn = gr.Button(\"Run Analysis\")\n",
    "    btn.click(analyze_medical_scan, [img_in, txt_in], [img_out, txt_out], api_name='predict')\n",
    "\n",
    "demo.app.add_middleware(\n",
    "    CORSMiddleware,\n",
    "    allow_origins=['*'],\n",
    "    allow_credentials=True,\n",
    "    allow_methods=['*'],\n",
    "    allow_headers=['*'],\n",
    ")\n",
    "\n",
    "print(\"🔗 Starting Base64 Backend with full CORS...\")\n",
    "demo.launch(share=True, max_threads=1, debug=False, show_error=True)"
]

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Completely replace the final cell to enforce the new Base64 structure
    found_interface = False
    for cell in reversed(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if 'gr.Blocks' in source or 'interface.launch' in source or 'demo.launch' in source:
                cell['source'] = DEBUG_CELL_SOURCE
                found_interface = True
                break
    
    if not found_interface:
        nb.get('cells').append({
            "cell_type": "code",
            "metadata": {"id": "base64-interface"},
            "execution_count": None,
            "outputs": [],
            "source": DEBUG_CELL_SOURCE
        })

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully applied Base64 Bypass to {file_path}")
except Exception as e:
    print(f"Error modifying notebook: {e}")
