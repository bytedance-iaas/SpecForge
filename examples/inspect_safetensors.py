from safetensors import safe_open

file_path = '/data01/qian_dev/SpecForge/outputs/qwen3-30b-a3b-instruct-eagle3-sharegpt/epoch_1_step_400/model.safetensors'

def inspect_safetensors(path, num_elements=5):
    with safe_open(path, framework="pt") as f:
        print("metadata:", f.metadata())
        print()

        for key in f.keys():
            tensor = f.get_tensor(key)

            # flatten tensor to print first few values
            flat = tensor.flatten()
            sample = flat[:num_elements]

            print(f"{key}")
            print(f"  shape: {tuple(tensor.shape)}")
            print(f"  dtype: {tensor.dtype}")
            print(f"  first {num_elements} values: {sample.tolist()}")
            print()

inspect_safetensors(file_path, num_elements=5)