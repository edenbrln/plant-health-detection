import os
import ujson as json
from PIL import Image
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

def process_file_metadata(file_path):
    """
    Extracts class, type, risk, and growth stage information from a single JSON annotation.
    """
    result = defaultdict(int)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data.get("type"), int):
                result[("type", data["type"])] += 1
            if isinstance(data.get("risk"), int):
                result[("risk", data["risk"])] += 1
            for obj in data.get("objects", []):
                if isinstance(obj.get("class"), int):
                    result[("class", obj["class"])] += 1
                if isinstance(obj.get("grow"), int):
                    result[("grow", obj["grow"])] += 1
    except Exception as e:
        result[("error", "parsing_error")] += 1
    return result

def plot_distribution(counter, title, xlabel):
    """
    Generates bar charts for data distribution analysis.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(counter.keys(), counter.values(), color='steelblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(list(counter.keys()))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    label_dir = r"path/to/Training/labels"
    
    json_files = [entry.path for entry in os.scandir(label_dir) if entry.name.endswith('.json')]
    print(f"Total JSON files found: {len(json_files)}")

    # Parallel processing for fast metadata extraction
    with Pool(cpu_count()) as pool:
        all_results = pool.map(process_file_metadata, json_files)

    merged = defaultdict(int)
    for result in all_results:
        for key, value in result.items():
            merged[key] += value

    class_counter = Counter({k[1]: v for k, v in merged.items() if k[0] == "class"})
    type_counter = Counter({k[1]: v for k, v in merged.items() if k[0] == "type"})
    risk_counter = Counter({k[1]: v for k, v in merged.items() if k[0] == "risk"})
    grow_counter = Counter({k[1]: v for k, v in merged.items() if k[0] == "grow"})

    # Visualization
    plot_distribution(class_counter, "Class Distribution", "Class ID")
    plot_distribution(type_counter, "Type Distribution", "Type Value")
    plot_distribution(risk_counter, "Risk Distribution", "Risk Level")
    plot_distribution(grow_counter, "Growth Stage Distribution", "Growth Stage")