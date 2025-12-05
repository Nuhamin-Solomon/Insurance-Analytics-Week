input_path = "data/raw_data.txt"
output_path = "data/processed_data.txt"

with open(input_path, "r") as f:
    lines = f.readlines()

# Simple processing: convert to uppercase
processed = [line.upper() for line in lines]

with open(output_path, "w") as f:
    f.writelines(processed)

print("Processing complete! Saved processed_data.txt")
