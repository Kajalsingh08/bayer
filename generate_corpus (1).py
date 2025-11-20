"""
Graph-to-Text Corpus Generator for Schema-Aware SLM

This script converts Cube.dev metadata and a business taxonomy into a rich, 
natural-language training corpus. It is designed to produce a high-quality dataset 
for fine-tuning Small Language Models (SLMs) to be schema-aware.

The script implements the strategy outlined in the CORPUS_CREATION_STRATEGY.md document,
which emphasizes the differential treatment of cubes, views, and the semantic catalog.
"""

import json
from typing import List, Dict, Set, Optional
from pathlib import Path
import logging

# --- Configuration ---
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Class for Corpus Generation ---

class GraphCorpusGenerator:
    """
    Generates a natural language corpus from Cube.dev metadata and business taxonomy.
    """
    
    def __init__(self, metadata_path: str, taxonomy_path: str, views_only_path: str):
        """
        Initializes the generator with paths to the data files.

        Args:
            metadata_path (str): Path to the metadata JSON file (e.g., test_meta.json).
            taxonomy_path (str): Path to the business taxonomy JSON file.
            views_only_path (str): Path to the views-only JSON file for additional context.
        """
        self.metadata = self._load_json(metadata_path)
        self.taxonomy = self._load_json(taxonomy_path)
        self.views_only = self._load_json(views_only_path)
        
        self.corpus_parts: List[str] = []
        self.seen_entities: Set[str] = set()

    def _load_json(self, file_path: str) -> Dict:
        """Loads a JSON file and returns its content."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {file_path}")
            return {}

    def generate_full_corpus(self) -> str:
        """
        Generates the complete training corpus by processing different parts of the metadata.

        Returns:
            str: The complete corpus as a single string.
        """
        logging.info("Starting corpus generation...")

        cubes_list = self.metadata.get('cubes', [])
        
        # Separate entities based on their type
        catalog_views = [c for c in cubes_list if c.get('name') == 'semantic_catalog']
        semantic_views = [c for c in cubes_list if c.get('type') == 'view' and c.get('name') != 'semantic_catalog']
        data_cubes = [c for c in cubes_list if c.get('type') == 'cube']

        # 1. Process the Semantic Catalog first for foundational context
        if catalog_views:
            logging.info("Processing semantic catalog...")
            for catalog in catalog_views:
                self.corpus_parts.append(self.generate_catalog_description(catalog))

        # 2. Process Data Cubes
        if data_cubes:
            logging.info(f"Processing {len(data_cubes)} data cubes...")
            for cube in data_cubes:
                self.corpus_parts.append(self.generate_cube_description(cube))

        # 3. Process Semantic Views
        if semantic_views:
            logging.info(f"Processing {len(semantic_views)} semantic views...")
            for view in semantic_views:
                self.corpus_parts.append(self.generate_view_description(view))

        # 4. Incorporate Business Taxonomy
        if self.taxonomy:
            logging.info("Generating business hierarchy description...")
            self.corpus_parts.append(self.generate_hierarchy_description())

        # 5. Add explicit relationship sentences
        logging.info("Generating relationship sentences...")
        self.corpus_parts.append(self.generate_relationship_sentences())

        # 6. Generate synthetic Q&A pairs for instruction-style training data
        logging.info("Generating query patterns...")
        self.corpus_parts.append(self.generate_query_patterns())

        # Combine all parts into a single corpus
        full_corpus = "\n\n---\n\n".join(self.corpus_parts)
        
        # --- Corpus Statistics ---
        stats = self._calculate_statistics(full_corpus)
        
        logging.info("Corpus generation complete.")
        for key, value in stats.items():
            logging.info(f"  - {key.replace('_', ' ').title()}: {value:,}")

        # Save statistics to a file
        self._save_statistics(stats)
        
        return full_corpus

    def generate_catalog_description(self, catalog: Dict) -> str:
        """
        Generates a detailed description for the semantic_catalog.
        This part is crucial as it describes the relationships between other entities.
        """
        if not catalog:
            return ""
        
        desc = "# Semantic Catalog - The Metadata Hub\n\n"
        desc += "The 'semantic_catalog' is a special view that acts as a metadata registry for the entire data schema. "
        desc += "It provides a complete picture of all available semantic views, cubes, and their relationships.\n\n"
        
        desc += "## Key Metadata Dimensions in the Catalog:\n\n"
        
        dimensions = catalog.get('dimensions', [])
        for dim in dimensions:
            dim_name = dim.get('name', 'unknown')
            dim_title = dim.get('title', '')
            dim_desc = dim.get('description', 'No description available.')
            
            # Highlight key relationship and context fields
            if any(keyword in dim_name for keyword in ['join', 'relationship', 'cube_', 'view_']):
                desc += f"- **{dim_title} (`{dim_name}`)**: {dim_desc}\n"
        
        return desc

    def generate_cube_description(self, cube: Dict) -> str:
        """
        Generates a detailed description for a data cube.
        """
        if not cube or cube.get('name') in self.seen_entities:
            return ""
        
        cube_name = cube.get('name', 'Unknown')
        self.seen_entities.add(cube_name)
        
        desc = f"# Data Cube: {cube.get('title', cube_name)}\n\n"
        desc += f"**Technical Name**: `{cube_name}`\n\n"
        desc += f"**Description**: {cube.get('description', 'No description available.')}\n\n"
        
        # Measures
        measures = cube.get('measures', [])
        if measures:
            desc += f"### Measures in {cube_name}:\n"
            for m in measures:
                desc += f"- **{m.get('title', m.get('name'))}** (`{m.get('name')}`): A `{m.get('aggType')}` aggregation. {m.get('description', '')}\n"
        
        # Dimensions
        dimensions = cube.get('dimensions', [])
        if dimensions:
            desc += f"\n### Dimensions in {cube_name}:\n"
            for d in dimensions:
                pk_info = " (Primary Key)" if d.get('primaryKey') else ""
                desc += f"- **{d.get('title', d.get('name'))}** (`{d.get('name')}`): Data type is `{d.get('type')}`.{pk_info} {d.get('description', '')}\n"
        
        # Detailed Field Descriptions
        desc += f"\n#### Detailed Fields for {cube_name}:\n"
        for m in measures:
            desc += self.generate_measure_description(m, cube_name) + "\n"
        for d in dimensions:
            desc += self.generate_dimension_description(d, cube_name) + "\n"

        return desc

    def generate_view_description(self, view: Dict) -> str:
        """
        Generates a detailed description for a semantic view.
        """
        if not view or view.get('name') in self.seen_entities:
            return ""
        
        view_name = view.get('name', 'Unknown')
        self.seen_entities.add(view_name)
        
        desc = f"# Semantic View: {view.get('title', view_name)}\n\n"
        desc += f"**Technical Name**: `{view_name}`\n\n"
        desc += f"**Description**: {view.get('description', 'No description available.')}\n\n"
        
        # Try to find business context from the taxonomy
        if self.taxonomy:
            for _, bu_data in self.taxonomy.get('hierarchy', {}).get('division', {}).get('business_units', {}).items():
                for _, subdiv_data in bu_data.get('subdivisions', {}).items():
                    for v in subdiv_data.get('views', []):
                        if v.get('name') == view_name:
                            desc += f"**Business Context**: Belongs to the '{subdiv_data.get('display_name')}' subdivision and is used for '{v.get('functional_area')}'.\n\n"
                            break
        
        # Measures in the view
        measures = view.get('measures', [])
        if measures:
            desc += f"### Key Metrics (Measures) in {view_name}:\n"
            for m in measures:
                desc += f"- **{m.get('title', m.get('name'))}** (`{m.get('name')}`): A `{m.get('aggType')}` aggregation.\n"
        
        # Dimensions in the view
        dimensions = view.get('dimensions', [])
        if dimensions:
            desc += f"\n### Attributes (Dimensions) in {view_name}:\n"
            for d in dimensions:
                desc += f"- **{d.get('title', d.get('name'))}** (`{d.get('name')}`): Data type is `{d.get('type')}`.\n"

        # Detailed Field Descriptions
        desc += f"\n#### Detailed Fields for {view_name}:\n"
        for m in measures:
            desc += self.generate_measure_description(m, view_name) + "\n"
        for d in dimensions:
            desc += self.generate_dimension_description(d, view_name) + "\n"
                
        return desc

    def generate_measure_description(self, measure: Dict, cube_name: str) -> str:
        """Generates a detailed, standalone description for a measure."""
        if not measure: return ""
        name = measure.get('name', 'unknown')
        title = measure.get('title', name)
        return (f"- **Measure**: The measure **{title}** (`{name}`) is used for `{measure.get('aggType')}` calculations. "
                f"Description: {measure.get('description', 'Not specified')}")

    def generate_dimension_description(self, dim: Dict, cube_name: str) -> str:
        """Generates a detailed, standalone description for a dimension."""
        if not dim: return ""
        name = dim.get('name', 'unknown')
        title = dim.get('title', name)
        pk_info = " It serves as a primary key." if dim.get('primaryKey') else ""
        return (f"- **Dimension**: The dimension **{title}** (`{name}`) is of type `{dim.get('type')}`.{pk_info} "
                f"Description: {dim.get('description', 'Not specified')}")

    def generate_hierarchy_description(self) -> str:
        """Generates a description of the business hierarchy from the taxonomy file."""
        if not self.taxonomy:
            return ""
        
        desc = "# Business Hierarchy and Context\n\n"
        desc += "This section describes the organizational structure and business context of the data.\n\n"
        
        hierarchy = self.taxonomy.get('hierarchy', {})
        division_data = hierarchy.get('division')

        if division_data:
            desc += f"## Division: {division_data.get('display_name', division_data.get('name'))}\n"
            
            business_units = division_data.get('business_units', {})
            for bu_name, bu_data in business_units.items():
                desc += f"### Business Unit: {bu_data.get('display_name', bu_name)}\n"
                desc += f"{bu_data.get('description', '')}\n\n"
        
        return desc

    def generate_relationship_sentences(self) -> str:
        """Generates explicit sentences describing view-cube relationships."""
        desc = "# View and Cube Relationships\n\n"
        semantic_views = [c for c in self.metadata.get('cubes', []) if c.get('type') == 'view' and c.get('name') != 'semantic_catalog']

        for view in semantic_views:
            view_name = view.get('title', view.get('name'))
            description = view.get('description', '')
            
            # Heuristic to find cube names in the description
            if 'A combined view of' in description:
                try:
                    # Extracts cube names like "A combined view of CUBE1, CUBE2, and CUBE3..."
                    cube_names_str = description.split('A combined view of')[1].split(' to provide')[0]
                    cube_names = [name.strip() for name in cube_names_str.replace('and ', '').split(',')]
                    if cube_names:
                        desc += f"The **{view_name}** view is constructed by combining data from the following cubes: **{', '.join(cube_names)}**.\n"
                except IndexError:
                    pass # Description format might not match
        return desc

    def generate_query_patterns(self) -> str:
        """
        Generates a diverse set of synthetic Question-Answer pairs for instruction tuning.
        """
        desc = "# Example Questions and Answers\n\n"
        
        cubes = self.metadata.get('cubes', [])
        for cube in cubes[:20]: # Process more cubes for variety
            cube_name = cube.get('name', 'Unknown')
            cube_title = cube.get('title', cube_name)
            
            # Question about purpose
            if cube.get('description'):
                desc += f"**Question**: What is the purpose of the '{cube_title}'?\n"
                desc += f"**Answer**: The '{cube_title}' (technical name: `{cube_name}`) is used for: {cube.get('description')}\n\n"
            
            # Question about measures
            measures = cube.get('measures', [])
            if measures:
                desc += f"**Question**: What metrics are available in '{cube_title}'?\n"
                measure_names = [f"'{m.get('title', m.get('name'))}'" for m in measures]
                desc += f"**Answer**: The '{cube_title}' provides the following metrics: {', '.join(measure_names)}.\n\n"

            # Question about a specific dimension's data type
            dimensions = cube.get('dimensions', [])
            if dimensions:
                dim = dimensions[0] # Pick the first one for an example
                dim_title = dim.get('title', dim.get('name'))
                dim_type = dim.get('type', 'unknown')
                desc += f"**Question**: What is the data type of '{dim_title}' in the '{cube_title}' view?\n"
                desc += f"**Answer**: In '{cube_title}', the data type for '{dim_title}' is `{dim_type}`.\n\n"

            # Question about field location
            if measures:
                measure = measures[0]
                measure_title = measure.get('title', measure.get('name'))
                desc += f"**Question**: Where can I find the '{measure_title}' metric?\n"
                desc += f"**Answer**: The metric '{measure_title}' is located in the **{cube_title}** cube/view.\n\n"

        return desc

    def _calculate_statistics(self, corpus: str) -> Dict:
        """Calculates various statistics about the generated corpus."""
        return {
            "total_parts": len(self.corpus_parts),
            "characters": len(corpus),
            "words": len(corpus.split()),
            "estimated_tokens": int(len(corpus.split()) * 1.3)
        }

    def _save_statistics(self, stats: Dict):
        """Saves the corpus statistics to a JSON file."""
        stats_path = Path("training_data/schema_corpus_stats.json")
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4)
            logging.info(f"Corpus statistics saved to: {stats_path}")
        except IOError as e:
            logging.error(f"Failed to save statistics file: {e}")

    def save_corpus(self, output_path: str):
        """
        Generates and saves the corpus and its statistics.
        """
        corpus = self.generate_full_corpus()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(corpus)
            logging.info(f"Corpus successfully saved to: {output_path}")
        except IOError as e:
            logging.error(f"Failed to save corpus file: {e}")

# --- Main Execution ---

def main():
    """Main function to run the corpus generation."""
    
    # --- TUNABLE PARAMETERS ---
    # You can change these paths to point to your actual data files.
    # For a full run, you might use 'full_meta.json'. For testing, 'test_meta.json' is faster.
    metadata_file = "data/full_meta.json"
    taxonomy_file = "data/business_taxonomy.json"
    views_only_file = "data/views_only.json"
    
    # The output file where the generated corpus will be saved.
    output_file = "training_data/schema_corpus.txt"
    
    logging.info("Initializing corpus generator...")
    generator = GraphCorpusGenerator(
        metadata_path=metadata_file,
        taxonomy_path=taxonomy_file,
        views_only_path=views_only_file
    )
    
    generator.save_corpus(output_file)

if __name__ == "__main__":
    main()