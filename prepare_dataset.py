import os, random, shutil

random.seed(42)

src_ambulance = 'dataset_all/ambulance'
src_firetruck = 'dataset_all/firetruck'

# create destination structure
os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/labels/train', exist_ok=True)
os.makedirs('dataset/labels/val', exist_ok=True)

def gather_files(src):
    if not os.path.isdir(src):
        return []
    return [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png','.heic','.webp'))]

def split_and_copy(src, cls_index):
    files = gather_files(src)
    random.shuffle(files)
    split_idx = int(0.8 * len(files))  # 80% train / 20% val
    for i, fname in enumerate(files):
        src_path = os.path.join(src, fname)
        dest_img_dir = 'dataset/images/train' if i < split_idx else 'dataset/images/val'
        dest_label_dir = 'dataset/labels/train' if i < split_idx else 'dataset/labels/val'
        # copy image (preserve original)
        shutil.copy2(src_path, os.path.join(dest_img_dir, fname))
        # create an empty YOLO .txt label file (to be filled by you/annotator)
        label_path = os.path.join(dest_label_dir, os.path.splitext(fname)[0] + '.txt')
        open(label_path, 'w').close()
        # NOTE: label files are empty placeholders. To add a full-image placeholder box:
        # open(label_path,'w').write(f"{cls_index} 0.5 0.5 1.0 1.0\n")

# copy for both classes
split_and_copy(src_ambulance, 0)
split_and_copy(src_firetruck, 1)

# report counts
def count_dir(d):
    return sum(1 for _ in os.listdir(d)) if os.path.isdir(d) else 0

print("counts:")
print("images/train:", count_dir('dataset/images/train'))
print("images/val:  ", count_dir('dataset/images/val'))
print("labels/train:", count_dir('dataset/labels/train'))
print("labels/val:  ", count_dir('dataset/labels/val'))
