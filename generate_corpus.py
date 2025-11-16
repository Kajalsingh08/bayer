import json
from typing import List, Dict, Set
from pathlib import Path
import hashlib

class GraphCorpusGenerator:
    """Generate natural language corpus from graph metadata"""
    
    def __init__(self, metadata_path: str, taxonomy_path: str):
        """
        Args:
            metadata_path: Path to full_meat.json
            taxonomy_path: Path to business_taxonomy.json
        """
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        with open(taxonomy_path) as f:
            self.taxonomy = json.load(f)
        
        self.corpus_parts = []
        self.seen_entities = set()  # Prevent duplicates
    
    def generate_cube_description(self, cube):
        name = cube.get("name", "UnknownCube")
        title = cube.get("title", name)
        ctype = cube.get("type", "cube")
        is_visible = cube.get("isVisible", True)
        is_public = cube.get("public", True)
        description = cube.get("description", "used for data analysis.")
        conn_components = cube.get("connectedComponents", [])

        desc = []

        # -------------------------------
        #   CUBE HEADER DESCRIPTION
        # -------------------------------
        desc.append(f"### Cube: {title}\n")
        desc.append(f"The **{title}** cube is a data structure wiht the description:{description}.\n")
        desc.append(f"It has the following properties:")
        desc.append(f"- **Name:** {name}")
        desc.append(f"- **Title:** {title}")
        desc.append(f"- **Type:** {ctype.capitalize()}")
        desc.append(f"- **Visibility:** {'visible' if is_visible else 'not visible'}, {'public' if is_public else 'private'}")
        desc.append(f"- **Connected Components:** {len(conn_components)}\n")

        measures = cube.get("measures", [])
        dims = cube.get("dimensions", [])
        # ---------------------------------------------------------
        # BRIEF MEASURE SUMMARY
        # ---------------------------------------------------------
        desc.append("## Measures (Brief Summary)\n")

        if measures:
            desc.append(f"This cube contains **{len(measures)} measures**:\n")
            for m in measures:
                m_name = m.get("name", "unknown")
                m_title = m.get("title", m_name)
                m_type = m.get("type", "unknown")
                m_agg = m.get("aggType", "aggregation")
                m_desc_text = m.get("description", "").strip()

                brief_line = (
                    f"- **{m_name}**: A **{m_agg} ** measure with following description : {m_desc_text}."
                    f"\" This measure has the title {m_title}\" and is of type {m_type}"
                )
                desc.append(brief_line)
            desc.append("\n")

        # ---------------------------------------------------------
        # BRIEF DIMENSION SUMMARY
        # ---------------------------------------------------------
        desc.append("## Dimensions (Brief Summary)\n")

        if dims:
            desc.append(f"This cube contains **{len(dims)} dimensions**:\n")
            for d in dims:
                d_name = d.get("name", "unknown")
                d_title = d.get("title", d_name)
                d_type = d.get("type", "unknown")
                d_pk = d.get("primaryKey", False)
                d_vis = d.get("isVisible", False)

                line = (
                    f"- **{d_name}**: A **{d_type}** dimension "
                    f"{'that serves as the primary key' if d_pk else ''}. "
                    f"This dimension has the title \"{d_title}\" and is "
                    f"{'visible' if d_vis else 'not visible'}."
                )

                desc.append(line)
            desc.append("\n")

        # ---------------------------------------------------------
        # DETAILED MEASURE DESCRIPTIONS
        # ---------------------------------------------------------
        desc.append("## Detailed Measure Descriptions\n")
        desc.append(
            "Below is a detailed breakdown of each measure, including type, aggregation "
            "behavior, visibility, titles, and additional metadata:\n"
        )

        for m in measures:
            m_name = m.get("name", "unknown")
            m_title = m.get("title", m_name)
            m_short = m.get("shortTitle", "")
            m_desc_text = m.get("description", "No description provided.")
            m_type = m.get("type", "unknown")
            agg = m.get("aggType", "unknown")
            visible = m.get("isVisible", False)
            pub = m.get("public", False)
            cumulative = m.get("cumulative", False)

            # Build the paragraph in EXACT template format
            paragraph = []

            paragraph.append(f"The {m_name} is a measure in the {name} cube.")
            paragraph.append(f"It is a {agg} aggregation of type {m_type}.")
            paragraph.append(f"Its full name is {name}.{m_name}")
            paragraph.append(f'Its title is "{m_title}".')

            if m_short:
                paragraph.append(f'Its short title is "{m_short}".')

            paragraph.append(f"Description: {m_desc_text}.")

            paragraph.append(
                f"This measure is {'visible' if visible else 'not visible'} and "
                f"{'public' if pub else 'private'}."
            )

            paragraph.append(
                f"It is {'cumulative' if cumulative else 'not cumulative'}."
            )

            desc.append("\n".join(paragraph) + "\n")

        desc.append("\n")

        # ---------------------------------------------------------
        # DETAILED DIMENSION DESCRIPTIONS
        # ---------------------------------------------------------
        desc.append("## Detailed Dimension Descriptions\n")
        desc.append(
            "The following section provides detailed information for each dimension, "
            "including type, primary key status, visibility, titles, and other metadata:\n"
        )

        for d in dims:
            d_name = d.get("name", "unknown")
            d_title = d.get("title", d_name)
            d_type = d.get("type", "unknown")
            d_desc_text = d.get("description", "No description provided.")
            primary = d.get("primaryKey", False)
            visible = d.get("isVisible", False)
            pub = d.get("public", False)
            suggest = d.get("suggestFilterValues", False)
            alias = d.get("aliasMember", "")
            meta = d.get("meta", {})

            full_name = f"{name}.{d_name}"

            paragraph = []

            # Follow the EXACT template
            paragraph.append(f"The {d_name} is a dimension in the {name} cube.")
            paragraph.append(f"It is of type {d_type}.")
            paragraph.append(f"Its full name is {full_name}.")
            paragraph.append(f'Its title is "{d_title}".')
            paragraph.append(f"Description: {d_desc_text}.")

            paragraph.append(
                f"This dimension is {'visible' if visible else 'not visible'} and "
                f"{'public' if pub else 'private'}."
            )
            paragraph.append(f"It is {'a primary key' if primary else 'not a primary key'}.")
            if suggest:
                paragraph.append("It suggests filter values.")
            else:
                paragraph.append("It does not suggest filter values.")
            # alias usage
            if alias:
                paragraph.append(f"It has an alias member '{alias}', useful for joining across cubes.")
            # meta sub-entity
            if "subEntity" in meta:
                paragraph.append(f"It belongs to the sub-entity '{meta.get('subEntity')}'.")

            desc.append("\n".join(paragraph) + "\n")


        return "\n".join(desc)
  
    def generate_hierarchy_description(self) -> str:
        """Generate business hierarchy descriptions"""
        
        desc = "# Business Hierarchy\n\n"
        desc += "## Organizational Structure\n\n"
        
        org_name = self.taxonomy.get('organization', 'Organization').get("name","Unknown")
        org_code = self.taxonomy.get('organization', 'Organization').get("code","N/A")
        desc += f"The **{org_name}** is the top-level organization. The code is {org_code}\n\n"
        
        # TODO: Kajal The key in data base is "division" but will we add more divisions in future or any addition will be in the division key only
        division = self.taxonomy.get('hierarchy', {}).get('division', {}) 
        div_name = division.get('name','Unknown')
        desc += f"### Division: {div_name}\n\n"
        desc += f"The {org_name} has a division called **{div_name}**.\n\n"
        
        business_units = division.get('business_units', {})

        for bu_name, bu_data in business_units.items():
            desc += f"#### Business Unit: {bu_name}\n\n"
            desc += f"The {div_name} division contains the **{bu_name}** business unit.\n\n"
            display_name = bu_data.get("display_name","Unknown")
            description = bu_data.get("description","Unknown")
            desc+=  f"The division with name is known as '{display_name}' and  is user for : {description}.\n\n"
            subdivisions = bu_data.get('subdivisions', {})
            for subdiv_name, subdiv_data in subdivisions.items():
                subdiv_desc = subdiv_data.get("description", "N/A")
                desc += f"##### Subdivision: {subdiv_name}\n\n"
                desc += f"The {bu_name} business unit has a **{subdiv_name}** subdivision and is used for {subdiv_desc}\n\n"
                
                functional_areas = subdiv_data.get('functional_areas', [])
                if functional_areas:
                    desc += "**Functional Areas:**\n"
                    for area in functional_areas:
                        display_name = area.get("display_name", area.get("name", ""))
                        description = area.get("description", "")
                        desc += f"- {display_name}: {description}.\n"
                    desc += "\n"
                
                views = subdiv_data.get('views', [])
                if views:
                    desc += "**Views:**\n"
                    for view in views:
                        name = view.get("name", "")
                        view_type = view.get("type", "")
                        functional_area = view.get("functional_area", "")
                        tags = view.get("tags", [])

                        # Join tags nicely
                        tags_text = ", ".join(tags) if tags else "no associated tags"

                        desc += (
                            f"- **{name}**: This is a {view_type} view belonging to the "
                            f"{functional_area.replace('_', ' ')} functional area. "
                            f"It includes tags such as {tags_text}.\n"
                        )

                    desc += "\n"
        
        view_classifications = self.taxonomy.get("view_classifications", {})
        if view_classifications:
            desc += "### View Classifications\n\n"
            desc += (
                "The business unit includes a set of classified views. "
                "Each classification describes the purpose of the view, the data domains it covers, "
                "its primary users, and how frequently its data is updated.\n\n"
            )

            for vc_name, vc_data in view_classifications.items():
                purpose = vc_data.get("purpose", "No purpose provided")
                data_domains = vc_data.get("data_domains", [])
                primary_users = vc_data.get("primary_users", [])
                update_freq = vc_data.get("update_frequency", "unknown frequency")

                domains_text = ", ".join(data_domains) if data_domains else "no data domains"
                users_text = ", ".join(primary_users) if primary_users else "no defined users"

                desc += (
                    f"- **{vc_name}**: This classification is used for {purpose}. "
                    f"It covers data domains such as {domains_text}. "
                    f"The primary users of this view include {users_text}. "
                    f"The data for this classification is updated on a {update_freq} basis.\n"
                )

            desc += "\n"

        view_relationships = self.taxonomy.get("view_relationships", {})
        if view_relationships:
            desc += "### View Relationships\n\n"
            desc += (
                "This section describes how different views are connected to one another. "
                "Each entry lists related views, and when available, the shared measures, "
                "shared dimensions, or special relationship types that define how the views "
                "interact within the data ecosystem.\n\n"
            )

            for view_name, vr_data in view_relationships.items():
                related_views = vr_data.get("related_views", [])
                shared_measures = vr_data.get("shared_measures", [])
                shared_dimensions = vr_data.get("shared_dimensions", [])
                relationship_type = vr_data.get("relationship_type", None)

                # Formatting lists
                related_text = ", ".join(related_views) if related_views else "no directly related views"
                measures_text = ", ".join(shared_measures) if shared_measures else None
                dimensions_text = ", ".join(shared_dimensions) if shared_dimensions else None

                desc += f"- **{view_name}**:\n"
                desc += f"  - Related views: {related_text}.\n"

                if measures_text:
                    desc += f"  - Shared measures: {measures_text}.\n"
                if dimensions_text:
                    desc += f"  - Shared dimensions: {dimensions_text}.\n"
                if relationship_type:
                    desc += f"  - Relationship type: {relationship_type}.\n"

                desc += "\n"

        metadata = self.taxonomy.get("metadata", {})
        if metadata:
            desc += "### Metadata Summary\n\n"
            desc += (
                "The following metadata provides a high-level overview of the structure and "
                "composition of this business unit, including counts of views, view types, "
                "business units, subdivisions, and functional areas.\n\n"
            )

            total_views = metadata.get("total_views", "N/A")
            view_types = metadata.get("view_types", {})
            business_units = metadata.get("business_units", "N/A")
            subdivisions = metadata.get("subdivisions", "N/A")
            functional_areas_count = metadata.get("functional_areas", "N/A")

            # View types expansion
            view_type_lines = []
            for vt_name, vt_count in view_types.items():
                # format: business_application → business application
                formatted_name = vt_name.replace("_", " ")
                view_type_lines.append(f"    - {formatted_name}: {vt_count}")

            view_type_text = "\n".join(view_type_lines) if view_type_lines else "    - No detailed view types listed"

            desc += f"- **Total Views:** {total_views}\n"
            desc += f"- **View Types:**\n{view_type_text}\n"
            desc += f"- **Business Units:** {business_units}\n"
            desc += f"- **Subdivisions:** {subdivisions}\n"
            desc += f"- **Functional Areas:** {functional_areas_count}\n\n"
        
        desc += "---\n\n"
        return desc
    
    def generate_query_patterns(self) -> str:
        """Generate Q&A patterns for common queries"""
        
        desc = "# Common Query Patterns\n\n"
        
        # For each cube, generate Q&A pairs
        cubes = self.metadata.get('cubes', [])
        
        for cube in cubes[:10]:  # Limit for brevity
            cube_name = cube.get('name', 'Unknown')
            measures = cube.get('measures', [])
            dimensions = cube.get('dimensions', [])
            
            # Measure query
            if measures:
                desc += f"**Question:** What measures are in {cube_name}?\n\n"
                desc += f"**Answer:** The {cube_name} cube has {len(measures)} measures: "
                measure_names = [m.get('name', 'unknown') for m in measures]
                desc += ", ".join(measure_names) + ".\n\n"
            
            # Primary key query
            pk_dims = [d for d in dimensions if d.get('primaryKey')]
            if pk_dims:
                desc += f"**Question:** What is the primary key of {cube_name}?\n\n"
                desc += f"**Answer:** The primary key is {pk_dims[0].get('name')}, "
                desc += f"which is a {pk_dims[0].get('type', 'unknown')} dimension.\n\n"
            
            # Dimension query
            if dimensions:
                desc += f"**Question:** How many dimensions does {cube_name} have?\n\n"
                desc += f"**Answer:** {cube_name} has {len(dimensions)} dimensions.\n\n"
        
        desc += "---\n\n"
        return desc
    
    def generate_relationship_patterns(self) -> str:
        """Generate relationship descriptions"""
        
        desc = "# Cube Relationships\n\n"
        
        # Look for foreign key relationships in dimensions
        cubes = self.metadata.get('cubes', [])
        
        for cube in cubes:
            cube_name = cube.get('name', 'Unknown')
            dimensions = cube.get('dimensions', [])
            
            for dim in dimensions:
                dim_name = dim.get('name', '')
                
                # Heuristic: if dimension name ends with _id and isn't primary key
                if dim_name.endswith('_id') and not dim.get('primaryKey'):
                    # Infer relationship
                    target_cube = dim_name.replace('_id', '').title() + 'ID'
                    
                    desc += f"**Relationship:** {cube_name} → {target_cube}\n\n"
                    desc += f"The {cube_name} cube references the {target_cube} cube "
                    desc += f"through the {dim_name} dimension.\n\n"
                    desc += f"This is a many-to-one relationship where multiple records "
                    desc += f"in {cube_name} can reference a single record in {target_cube}.\n\n"
        
        desc += "---\n\n"
        return desc
    
    def generate_full_corpus(self) -> str:
        """Generate complete training corpus"""
        
        print("Generating corpus...")
        
        corpus_parts = []
        
        # 1. Business Hierarchy (10% of corpus)
        # print("  - Generating hierarchy descriptions...")
        # corpus_parts.append(self.generate_hierarchy_description())
        
        # 2. Cube Descriptions (40% of corpus)
        # print("  - Generating cube descriptions...")
        # cubes = self.metadata.get('cubes', [])
        # for i, cube in enumerate(cubes):
        #     if i % 10 == 0:
        #         print(f"    Processed {i}/{len(cubes)} cubes")
        #     corpus_parts.append(self.generate_cube_description(cube))
        
        # IMP : REMOVED THE MEASURE FUNCTION BECAUSE I HAVE INCLUDED THE MEASURES IN CUBE ONLY
        
        # 4. Query Patterns (20% of corpus)
        # print("  - Generating query patterns...")
        # corpus_parts.append(self.generate_query_patterns())
        
        # # 5. Relationships (10% of corpus)
        # print("  - Generating relationship patterns...")
        # corpus_parts.append(self.generate_relationship_patterns())
        
        # Combine all parts
        full_corpus = "\n".join(corpus_parts)
        
        # Calculate stats
        char_count = len(full_corpus)
        word_count = len(full_corpus.split())
        token_estimate = int(word_count * 1.3)  # Rough estimate
        
        print(f"\nCorpus Statistics:")
        print(f"  - Characters: {char_count:,}")
        print(f"  - Words: {word_count:,}")
        print(f"  - Estimated Tokens: {token_estimate:,}")
        
        return full_corpus
    
    def save_corpus(self, output_path: str):
        """Generate and save corpus to file"""
        
        corpus = self.generate_full_corpus()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(corpus)
        
        print(f"\n✓ Corpus saved to: {output_path}")
        return corpus
    
def main():
    """Main execution"""
    
    # Paths
    metadata_path = "./views_only.json"
    taxonomy_path = "./business_taxonomy.json"
    output_path = "./graph_corpus_v1.txt"
    
    # Generate corpus
    generator = GraphCorpusGenerator(metadata_path, taxonomy_path)
    generator.save_corpus(output_path)

if __name__ == "__main__":
    main()      