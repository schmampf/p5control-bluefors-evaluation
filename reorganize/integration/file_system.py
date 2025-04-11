from dataclasses import dataclass

@dataclass
class CollectionConfig:
    root: str
    title: str
    
    figure_folder: str
    data_folder: str   