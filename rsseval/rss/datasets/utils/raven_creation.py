
import os
import glob
import numpy as np
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torchvision import transforms

class RAVEN_Dataset(Dataset):
    """
    RAVEN Dataset for RSBench.
    Loads RAVEN-10000 .npz files and filters them based on the 'Phase 1' curriculum:
    Phase 1 = Samples where exactly one attribute varies according to a non-Constant rule,
              and all other attributes are Constant.
    """
    def __init__(self, base_path, config="center_single", split="train"):
        self.base_path = base_path
        self.config = config
        self.split = split
        
        # Search pattern for files
        pattern = os.path.join(self.base_path, self.config, f"RAVEN_*_{self.split}.npz")
        self.all_files = sorted(glob.glob(pattern))
        
        if len(self.all_files) == 0:
            print(f"Warning: No files found for {pattern}")
            self.files = []
        else:
            self.files = self.all_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        
        # Images: [16, 160, 160]
        # Normalize to [0, 1] and add channel dimension: [16, 1, 160, 160]
        images = data['image'].astype(np.float32) / 255.0
        images = torch.from_numpy(images).unsqueeze(1)
        
        # Target: 0-7
        target = torch.tensor(data['target'], dtype=torch.long)
        
        # Concepts: Extract attributes for all 16 panels (8 context + 8 choices)
        # We need to parse XML again or trust meta_matrix? 
        # meta_matrix does NOT contain entity values (like Type=Triangle), only Rule activity.
        # So we must parse XML to get ground truth concepts for supervision.
        
        concepts = self._extract_concepts_from_xml(file_path.replace('.npz', '.xml'))
        
        return images, target, concepts

    def _extract_concepts_from_xml(self, xml_path):
        """
        Extract concept values for all 16 panels.
        Returns tensor of shape [16, 4] -> (Type, Size, Color, Number)
        Values are indices.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        panels_data = []
        all_panels = root.findall('.//Panel')
        # We expect 16 panels (8 context + 8 choices)
        # Note: XML might have more structure, but usually flattened order matches NPZ image stack.
        
        # Mappings based on RAVEN/src/dataset/const.py
        # Type: 0:none, 1:triangle, 2:square, 3:pentagon, 4:hexagon, 5:circle
        # Size: 0..5
        # Color: 0..9
        # Number: 0..8 (1-9) in const.py, but usually 0-indexed in XML attribute?
        
        for panel in all_panels[:16]:
            entity = panel.find('.//Entity')
            layout = panel.find('.//Layout')
            
            # Defaults
            p_c = [0, 0, 0, 0] # Type, Size, Color, Number
            
            if entity is not None:
                try: p_c[0] = int(entity.get('Type', 0)) - 1 # Shift 1-5 to 0-4
                except: pass
                try: p_c[1] = int(entity.get('Size', 0))
                except: pass
                try: p_c[2] = int(entity.get('Color', 0))
                except: pass
            
            if layout is not None:
                try: p_c[3] = int(layout.get('Number', 0))
                except: pass
            
            panels_data.append(p_c)
            
        return torch.tensor(panels_data, dtype=torch.long)



if __name__ == "__main__":
    # Initialize dataset without filtering to check a broad range of files
    dataset = RAVEN_Dataset(
        base_path="data/RAVEN-10000",
        config="center_single",
        split="train",
    )

    
    print(f"Analyzing first 1000 files out of {len(dataset.all_files)} for attribute value ranges...")
    
    attr_counts = {
        'Type': set(),
        'Size': set(),
        'Color': set(),
        'Number': set()
    }
    
    # Check first 1000 files to get a good sample
    for i, f in enumerate(dataset.all_files[:1000]):
        xml_path = f.replace('.npz', '.xml')
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for panel in root.findall('.//Panel'):
                entity = panel.find('.//Entity')
                layout = panel.find('.//Layout')
                
                if entity is not None:
                    if 'Type' in entity.attrib: attr_counts['Type'].add(int(entity.get('Type')))
                    if 'Size' in entity.attrib: attr_counts['Size'].add(int(entity.get('Size')))
                    if 'Color' in entity.attrib: attr_counts['Color'].add(int(entity.get('Color')))
                
                if layout is not None:
                    if 'Number' in layout.attrib: attr_counts['Number'].add(int(layout.get('Number')))

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")

    print("\nValue Ranges Found in XML:")
    for attr, values in attr_counts.items():
        sorted_vals = sorted(list(values))
        min_val = min(sorted_vals) if sorted_vals else 'N/A'
        max_val = max(sorted_vals) if sorted_vals else 'N/A'
        print(f"{attr}: {sorted_vals} (Min: {min_val}, Max: {max_val})")