from collections import defaultdict
import uproot

class MassPointFileMapper:
    def __init__(self):
        self.cached_list_of_files = {}

    def map_masspoints(self, files, redirector=""):
        """
        Reads each ROOT file and maps GenModel_* branches to the file name.
        Returns a dictionary: {mass_point_str: [list of files]}
        """
        mass_point_map = defaultdict(list)

        for i, file in enumerate(files):
            full_path = redirector + file if redirector and not file.startswith("root://") else file
            print(f"[{i+1}/{len(files)}] Processing: {full_path}")

            try:
                with uproot.open(full_path + ":Events") as events_tree:
                    branches = events_tree.keys()
                    for b in branches:
                        if b.startswith("GenModel__"):
                            clean_mass_point = b.replace("GenModel__", "")
                            mass_point_map[clean_mass_point].append(full_path)
            except Exception as e:
                print(f"Failed to read {full_path}: {e}")

        print("\nSummary of Mass Points:")
        for mass_point, file_list in mass_point_map.items():
            print(f"{mass_point}: {len(file_list)} files")
            for f in file_list:
                print(f"  - {f}")
                
        return mass_point_map

