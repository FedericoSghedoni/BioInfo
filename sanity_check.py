import re
from collections import defaultdict
import os

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[:94]
    return lines

def extract_parameters(lines):
    param_sections = []
    current_section = {}
    in_param_section = False

    for line in lines:
        if line.startswith("Dataset:"):
            if in_param_section:
                param_sections.append(current_section)
                current_section = {}
            in_param_section = True

        if in_param_section:
            if line.strip() == "":
                param_sections.append(current_section)
                current_section = {}
                in_param_section = False
            else:
                param_pattern = re.compile(r"^([\w\s]+): (.+)$")
                match = param_pattern.match(line)
                if match:
                    param, value = match.groups()
                    current_section[param.strip()] = value.strip()

    if current_section:
        param_sections.append(current_section)

    return param_sections

def compare_parameters(param_sections):
    discrepancies = defaultdict(list)
    
    if len(param_sections) < 2:
        return discrepancies

    first_section = param_sections[0]

    for section in param_sections[1:]:
        for param, value in first_section.items():
            if param in section and section[param] != value:
                discrepancies[param].append((first_section[param], section[param]))

    return discrepancies

def main():
    
    # Percorri tutte le cartelle di log
    logs_folders = ['logs_vit_mix', 'log_vit_mix_4']
    param_sections = []

    for folder in logs_folders:
        log_file = os.path.join(folder, 'log.txt')
        lines = read_log_file(log_file)
        param_sections.append(extract_parameters(lines)[0])
    discrepancies = compare_parameters(param_sections)

    if discrepancies:
        print(f"Discrepanze trovate in dir tra {logs_folders[0]} e {logs_folders[1]}:")
        for param, values in discrepancies.items():
            print(f"{param}: {values}")

if __name__ == "__main__":
    main()
