file_path = "internals/document/store_policy.txt"

with open(file_path, "r", encoding="utf-8") as file:
    content = file.read().strip().split("\n\n")
    content = list(filter(None, content))

    print(len(content))

    for idx, value in enumerate(content, start = 1):
        print(idx, value)