import os
import random
import shutil

# defining original path & seperate path
source_dir = "data/classification"
train_dir = "data/train"
val_dir = "data/val"
split_ratio = 0.8

# make folder for result
for folder in [train_dir,val_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ၃။ classification အောက်က folder များကို ရှာဖွေခြင်း
categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

print("🚀 ဒေတာများကို Train (80%) နှင့် Val (20%) စတင်ခွဲဝေနေပါသည်...")

for category in categories:
    category_source = os.path.join(source_dir, category)
    all_images = [f for f in os.listdir(category_source) if os.path.isfile(os.path.join(category_source, f))]

    # ပုံများကို သတ်မှတ်ချက်မရှိ ရောသမမွှေခြင်း (ကျပန်းခွဲဝေနိုင်ရန်)
    random.seed(42)  # အမြဲတမ်း ပုံစံတူ ခွဲဝေမှုရစေရန် Seed သတ်မှတ်ခြင်း
    random.shuffle(all_images)

    # 80% နေရာကို တွက်ချက်ခြင်း
    split_idx = int(len(all_images) * split_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # ပစ်မှတ် Folder များ ဆောက်ခြင်း (ဥပမာ - data/train/acne)
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

    # ပုံများကို သက်ဆိုင်ရာ Folder များထဲသို့ ကူးထည့်ခြင်း
    for img in train_images:
        shutil.copy(os.path.join(category_source, img), os.path.join(train_dir, category, img))
    for img in val_images:
        shutil.copy(os.path.join(category_source, img), os.path.join(val_dir, category, img))

    print(f"✨ {category} ခွဲဝေမှု ပြီးပါပြီ -> Train: {len(train_images)} ပုံ | Val: {len(val_images)} ပုံ")

print("--------------------------------------------------")
print("🎯 ဒေတာခွဲဝေခြင်း လုပ်ငန်းစဉ် ရာနှုန်းပြည့် အောင်မြင်စွာ ပြီးဆုံးပါပြီ!")

