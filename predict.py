from ultralytics import YOLO

model = YOLO('best.pt')
image_source = 'test_img5.jpg'
results = model.predict(source=image_source, stream=True, conf=0.25, iou=0.50)

print("The Following are results: ")
for result in results:
    class_thresholds = {
        'Acne': 0.25,
        'Rosacea': 0.50,
        'Melasma': 0.25
    }
    keep_indices = []

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        cls_name = result.names[cls_id]
        conf_val = float(box.conf[0].item())
        threshold = class_thresholds.get(cls_name, 0.25)

        if conf_val >= threshold:
            keep_indices.append(i)

    result.boxes = result.boxes[keep_indices]

    boxes = result.boxes
    total_detections = len(boxes)

    if total_detections > 0:
        print(f"Total number of existing symptoms: {total_detections} ")
        found_something = True
    else:
        print("No symptoms were found.")
        found_something = False

    if found_something:
        class_names = result.names
        class_ids = boxes.cls.tolist()

        counts_per_class = {}
        for cls_id in class_ids:
            cls_name = class_names[int(cls_id)]
            counts_per_class[cls_name] = counts_per_class.get(cls_name, 0) + 1
        print("\nThe number of each:")
        for name, count in counts_per_class.items():
            print(f"- {name}: {count} ")

    result.show()