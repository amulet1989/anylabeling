import os
import json
import shutil
from pathlib import Path
from PIL import Image


def convert_to_yolov8_segmentation(input_dir, output_dir, val=True):
    # Crear las carpetas necesarias en el directorio de salida
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Subcarpetas de entrenamiento y validación
    if val:
        subfolders = ['Train', 'Validation']
    else:
        subfolders = ['Train']

    
    for subfolder in subfolders:
        input_images_path = os.path.join(input_dir, subfolder)
        output_images_path = os.path.join(images_dir, subfolder)
        output_labels_path = os.path.join(labels_dir, subfolder)
        os.makedirs(output_images_path, exist_ok=True)
        os.makedirs(output_labels_path, exist_ok=True)
        
        image_list = []
        
        for file_name in os.listdir(input_images_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                base_name = os.path.splitext(file_name)[0]
                image_path = os.path.join(input_images_path, file_name)
                json_path = os.path.join(input_images_path, f"{base_name}.json")
                
                # Copiar la imagen al directorio de salida
                shutil.copy(image_path, os.path.join(output_images_path, file_name))
                
                # Leer las anotaciones del archivo .json
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                height, width = data["imageHeight"], data["imageWidth"]
                
                # Convertir las anotaciones a formato YOLOv8
                label_file_path = os.path.join(output_labels_path, f"{base_name}.txt")
                with open(label_file_path, 'w') as label_file:
                    for shape in data['shapes']:
                        if shape['shape_type'] == 'polygon':
                            points = shape['points']
                            normalized_points = [(x / width, y / height) for x, y in points]
                            class_id = 0  # Asumimos que la clase es siempre "product"
                            line = f"{class_id} " + " ".join([f"{x} {y}" for x, y in normalized_points])
                            label_file.write(line + '\n')
                
                # Guardar la ruta de la imagen para el archivo Train.txt o Validation.txt
                image_list.append(f"data/images/{subfolder}/{file_name}")
        
        # Guardar el archivo Train.txt o Validation.txt
        with open(os.path.join(output_dir, f"{subfolder}.txt"), 'w') as f:
            f.write("\n".join(image_list))
    
    # Crear el archivo data.yaml
    data_yaml_content = f"""
Train: Train.txt
Validation: Validation.txt
names:
  0: product
path: .
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(data_yaml_content.strip())

# Ejemplo de uso:
# convert_to_yolov8_segmentation('ruta/a/dataset/original', 'ruta/a/dataset/salida')


def convert_yolov8_to_custom(image_dir, label_dir, num_classes, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            base_name = os.path.splitext(image_name)[0]
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                continue  # Saltar si no hay una etiqueta correspondiente

            # Leer la imagen para obtener sus dimensiones
            with Image.open(image_path) as img:
                image_width, image_height = img.size
            
            # Copiar la imagen a la carpeta de salida
            output_image_path = os.path.join(output_dir, image_name)
            shutil.copy(image_path, output_image_path)

            # Leer el archivo de etiquetas .txt
            shapes = []
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    points = list(map(float, parts[1:]))
                    
                    # Convertir los puntos a la escala original de la imagen
                    polygon_points = []
                    for i in range(0, len(points), 2):
                        x = points[i] * image_width
                        y = points[i + 1] * image_height
                        polygon_points.append([x, y])
                    
                    shape = {
                        "label": class_names[class_id],
                        "text": "",
                        "points": polygon_points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    shapes.append(shape)
            
            # Crear el contenido del archivo .json
            annotation = {
                "version": "0.4.10",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": image_height,
                "imageWidth": image_width,
                "text": ""
            }
            
            # Guardar el archivo .json en la carpeta de salida
            output_json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(output_json_path, 'w') as json_file:
                json.dump(annotation, json_file, indent=4)

# Ejemplo de uso:
# convert_yolov8_to_custom('ruta/a/imagenes', 'ruta/a/etiquetas', 1, ['product'], 'ruta/a/dataset/salida')



