file_path = "classes.names"   # replace with your file path

with open(file_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

print("Classes found:")
for i, name in enumerate(classes):
    print(f"{i}: {name}")