import os
from PIL import Image

# ၁။ စစ်ဆေးမယ့် Folder လမ်းကြောင်း
acne_dir = "data/classification/acne"

if not os.path.exists(acne_dir):
    print(f"❌ Error: {acne_dir} လမ်းကြောင်း ရှာမတွေ့ပါ။ Folder တည်ဆောက်ပုံကို ပြန်စစ်ပါ။")
else:
    all_files = os.listdir(acne_dir)
    total_files = len(all_files)

    print(f"📊 Folder ထဲမှာ ရှာတွေ့တဲ့ စုစုပေါင်းဖိုင်အရေအတွက်: {total_files} ဖိုင်")
    print("🔍 ဖိုင်အပျက်အစီး ရှိ၊ မရှိ စတင်စစ်ဆေးနေပါသည်...")

    corrupted_count = 0

    # ၃။ ပုံတစ်ပုံချင်းစီကို Python နဲ့ လှမ်းဖွင့်ပြီး စစ်မယ်
    for file_name in all_files:
        file_path = os.path.join(acne_dir, file_name)

        try:
            with Image.open(file_path) as img:
                img.verify()
        except (IOError, SyntaxError):
            print(f"❌ ဖိုင်အပျက် ရှာတွေ့ပါသည်: {file_name}")
            corrupted_count += 1

    print("--------------------------------------------------")
    print(f"✅ စစ်ဆေးခြင်း ပြီးဆုံးပါပြီ။")
    print(f"🎯 သုံးလို့ရတဲ့ ပုံကောင်း: {total_files - corrupted_count} ပုံ")
    print(f"⚠️ ဖိုင်အပျက်အစီး: {corrupted_count} ဖိုင်")