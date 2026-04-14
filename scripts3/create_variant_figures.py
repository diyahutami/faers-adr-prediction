"""
create_variant_figures.py
==========================
Create demographic visualization figures for XXX, XXX-Gender, and XXX-Age datasets.
Adapted for the current preprocessing pipeline structure.

Usage:
    python create_variant_figures.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_PATH, PREPROCESSED_PATH, OUTPUT_PATH

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VariantFigureCreator:
    """
    Create demographic visualization figures from XXX variant datasets
    """
    
    def __init__(self, data_path=DATA_PATH, preprocessed_path=PREPROCESSED_PATH, 
                 output_path=None):
        """
        Initialize with paths to data
        
        Parameters:
        -----------
        data_path : str
            Path to raw FAERS data
        preprocessed_path : str
            Path to preprocessed data with splits
        output_path : str
            Path to save output figures
        """
        self.data_path = data_path
        self.preprocessed_path = preprocessed_path
        self.output_path = output_path or os.path.join(OUTPUT_PATH, "variant_figures")
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load all necessary datasets"""
        print("="*80)
        print("LOADING DATA FOR FIGURE GENERATION")
        print("="*80)
        
        # Load main demographics and drug/ADR data
        print("\nLoading raw data...")
        self.demographics = pd.read_csv(f"{self.data_path}/DEMOGRAPHICS.csv")
        self.drugs = pd.read_csv(f"{self.data_path}/DRUGS_STANDARDIZED_DRUGBANK.csv")
        self.adverse_reactions = pd.read_csv(f"{self.data_path}/ADVERSE_REACTIONS.csv")
        
        print(f"  ✓ Demographics: {len(self.demographics):,} records")
        print(f"  ✓ Drugs: {len(self.drugs):,} records")
        print(f"  ✓ Adverse Reactions: {len(self.adverse_reactions):,} records")
        
        # Load dataset splits
        print("\nLoading dataset splits...")
        splits_file = f"{self.preprocessed_path}/splits.json"
        with open(splits_file, 'r') as f:
            self.splits = json.load(f)
        
        print(f"  ✓ Loaded splits for variants: {list(self.splits.keys())}")
        
        # Load association test results (if available)
        self.load_association_results()
        
        print("\n✓ Data loading complete!\n")
    
    def load_association_results(self):
        """Load gender/age association test results if available"""
        gender_file = f"{self.preprocessed_path}/gender_associations.csv"
        age_file = f"{self.preprocessed_path}/age_associations.csv"
        
        if os.path.exists(gender_file):
            self.gender_assoc = pd.read_csv(gender_file)
            print(f"  ✓ Gender associations: {len(self.gender_assoc):,} pairs")
        else:
            self.gender_assoc = None
            print(f"  ⚠ Gender associations not found: {gender_file}")
        
        if os.path.exists(age_file):
            self.age_assoc = pd.read_csv(age_file)
            print(f"  ✓ Age associations: {len(self.age_assoc):,} pairs")
        else:
            self.age_assoc = None
            print(f"  ⚠ Age associations not found: {age_file}")
    
    def get_variant_patients(self, variant):
        """Get all patient IDs for a specific variant (train + val + test)"""
        variant_splits = self.splits.get(variant, {})
        all_patients = set()
        
        for split in ['train', 'val', 'test']:
            if split in variant_splits:
                all_patients.update(variant_splits[split])
        
        return list(all_patients)
    
    def create_donut_chart(self, data, labels, colors, title, ax):
        """
        Create a donut chart with labels inside
        
        Parameters:
        -----------
        data : array-like
            Values for each category
        labels : list
            Labels for each category
        colors : list
            Colors for each category
        title : str
            Chart title
        ax : matplotlib axis
            Axis to plot on
        """
        # Create donut chart
        wedges, texts = ax.pie(
            data, 
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
        )
        
        # Add labels with percentages inside
        total = sum(data)
        for i, (wedge, label) in enumerate(zip(wedges, labels)):
            angle = (wedge.theta2 + wedge.theta1) / 2
            radius = 0.75
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            
            pct = data[i] / total * 100 if total > 0 else 0
            
            # Determine text color based on background
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(colors[i])
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = 'white' if luminance < 0.5 else 'black'
            
            ax.text(
                x, y,
                f"{label}\n{pct:.1f}%",
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                color=text_color
            )
        
        # Add title in center
        ax.text(0, 0, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.axis('equal')
    
    def create_gender_sankey_plotly(self, drug_name, adr_data_dict, filename):
        """
        Create Plotly Sankey diagram for gender demographics
        
        Parameters:
        -----------
        drug_name : str
            Drug name to display
        adr_data_dict : dict
            Dictionary with ADR names as keys and (female_count, male_count) as values
        filename : str
            Output filename
        """
        # Prepare data for Sankey
        adrs = list(adr_data_dict.keys())
        
        # Calculate totals
        total_female = sum([adr_data_dict[adr][0] for adr in adrs])
        total_male = sum([adr_data_dict[adr][1] for adr in adrs])
        total = total_female + total_male
        
        # Create node labels
        # Node 0: Drug
        # Node 1: Female group
        # Node 2: Male group
        # Nodes 3+: ADRs
        
        node_labels = [drug_name, f'Female<br>{total_female/total*100:.1f}%', f'Male<br>{total_male/total*100:.1f}%']
        node_labels.extend([f'{adr}<br>({adr_data_dict[adr][0]}:{adr_data_dict[adr][1]})' for adr in adrs])
        
        # Node colors
        node_colors = [
            'rgba(128, 128, 128, 0.8)',  # Drug (gray)
            'rgba(232, 165, 198, 0.8)',  # Female (pink)
            'rgba(136, 204, 232, 0.8)'   # Male (blue)
        ]
        # ADR colors (alternating pink/blue based on predominant gender)
        for adr in adrs:
            female_count, male_count = adr_data_dict[adr]
            if female_count > male_count:
                node_colors.append('rgba(232, 165, 198, 0.6)')  # Pink
            else:
                node_colors.append('rgba(136, 204, 232, 0.6)')  # Blue
        
        # Create links
        sources = []
        targets = []
        values = []
        link_colors = []
        
        # Links from drug to gender groups
        sources.append(0)  # Drug
        targets.append(1)  # Female
        values.append(total_female)
        link_colors.append('rgba(232, 165, 198, 0.4)')
        
        sources.append(0)  # Drug
        targets.append(2)  # Male
        values.append(total_male)
        link_colors.append('rgba(136, 204, 232, 0.4)')
        
        # Links from gender groups to ADRs
        for i, adr in enumerate(adrs):
            female_count, male_count = adr_data_dict[adr]
            adr_node_idx = 3 + i
            
            # Female to ADR
            if female_count > 0:
                sources.append(1)
                targets.append(adr_node_idx)
                values.append(female_count)
                link_colors.append('rgba(232, 165, 198, 0.3)')
            
            # Male to ADR
            if male_count > 0:
                sources.append(2)
                targets.append(adr_node_idx)
                values.append(male_count)
                link_colors.append('rgba(136, 204, 232, 0.3)')
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="white", width=2),
                label=node_labels,
                color=node_colors,
                x=[0.1, 0.4, 0.4] + [0.9]*len(adrs),  # X positions
                y=[0.5, 0.3, 0.7] + list(np.linspace(0, 1, len(adrs)))  # Y positions
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title={
                'text': f"<b>Gender Demographics for {drug_name}</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            font=dict(size=12, family="Arial"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        # Save as HTML and PNG
        html_path = f"{self.output_path}/{filename}.html"
        png_path = f"{self.output_path}/{filename}.png"
        
        fig.write_html(html_path)
        fig.write_image(png_path, width=1200, height=600, scale=2)
        
        print(f"  ✓ Saved interactive: {html_path}")
        print(f"  ✓ Saved static: {png_path}")
    
    def create_age_sankey_plotly(self, drug_name, adr_data_dict, filename):
        """
        Create Plotly Sankey diagram for age demographics
        
        Parameters:
        -----------
        drug_name : str
            Drug name to display
        adr_data_dict : dict
            Dictionary with ADR names as keys and (youth_count, adult_count, elderly_count) as values
        filename : str
            Output filename
        """
        # Prepare data for Sankey
        adrs = list(adr_data_dict.keys())
        
        # Calculate totals
        total_youth = sum([adr_data_dict[adr][0] for adr in adrs])
        total_adult = sum([adr_data_dict[adr][1] for adr in adrs])
        total_elderly = sum([adr_data_dict[adr][2] for adr in adrs])
        total = total_youth + total_adult + total_elderly
        
        # Create node labels
        # Node 0: Drug
        # Node 1: Youth
        # Node 2: Adult
        # Node 3: Elderly
        # Nodes 4+: ADRs
        
        node_labels = [
            drug_name,
            f'Youth<br>{total_youth/total*100:.1f}%',
            f'Adult<br>{total_adult/total*100:.1f}%',
            f'Elderly<br>{total_elderly/total*100:.1f}%'
        ]
        node_labels.extend([f'{adr}<br>({adr_data_dict[adr][0]}:{adr_data_dict[adr][1]}:{adr_data_dict[adr][2]})' 
                           for adr in adrs])
        
        # Node colors
        node_colors = [
            'rgba(128, 128, 128, 0.8)',  # Drug (gray)
            'rgba(168, 213, 186, 0.8)',  # Youth (green)
            'rgba(255, 184, 77, 0.8)',   # Adult (orange)
            'rgba(255, 107, 107, 0.8)'   # Elderly (red)
        ]
        
        # ADR colors (based on predominant age group)
        for adr in adrs:
            youth_count, adult_count, elderly_count = adr_data_dict[adr]
            max_count = max(youth_count, adult_count, elderly_count)
            if youth_count == max_count:
                node_colors.append('rgba(168, 213, 186, 0.6)')  # Green
            elif adult_count == max_count:
                node_colors.append('rgba(255, 184, 77, 0.6)')   # Orange
            else:
                node_colors.append('rgba(255, 107, 107, 0.6)')  # Red
        
        # Create links
        sources = []
        targets = []
        values = []
        link_colors = []
        
        # Links from drug to age groups
        if total_youth > 0:
            sources.append(0)  # Drug
            targets.append(1)  # Youth
            values.append(total_youth)
            link_colors.append('rgba(168, 213, 186, 0.4)')
        
        if total_adult > 0:
            sources.append(0)  # Drug
            targets.append(2)  # Adult
            values.append(total_adult)
            link_colors.append('rgba(255, 184, 77, 0.4)')
        
        if total_elderly > 0:
            sources.append(0)  # Drug
            targets.append(3)  # Elderly
            values.append(total_elderly)
            link_colors.append('rgba(255, 107, 107, 0.4)')
        
        # Links from age groups to ADRs
        for i, adr in enumerate(adrs):
            youth_count, adult_count, elderly_count = adr_data_dict[adr]
            adr_node_idx = 4 + i
            
            # Youth to ADR
            if youth_count > 0:
                sources.append(1)
                targets.append(adr_node_idx)
                values.append(youth_count)
                link_colors.append('rgba(168, 213, 186, 0.3)')
            
            # Adult to ADR
            if adult_count > 0:
                sources.append(2)
                targets.append(adr_node_idx)
                values.append(adult_count)
                link_colors.append('rgba(255, 184, 77, 0.3)')
            
            # Elderly to ADR
            if elderly_count > 0:
                sources.append(3)
                targets.append(adr_node_idx)
                values.append(elderly_count)
                link_colors.append('rgba(255, 107, 107, 0.3)')
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="white", width=2),
                label=node_labels,
                color=node_colors,
                x=[0.1, 0.4, 0.4, 0.4] + [0.9]*len(adrs),  # X positions
                y=[0.5, 0.2, 0.5, 0.8] + list(np.linspace(0, 1, len(adrs)))  # Y positions
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title={
                'text': f"<b>Age Demographics for {drug_name}</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            font=dict(size=12, family="Arial"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=700,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        # Save as HTML and PNG
        html_path = f"{self.output_path}/{filename}.html"
        png_path = f"{self.output_path}/{filename}.png"
        
        fig.write_html(html_path)
        fig.write_image(png_path, width=1400, height=700, scale=2)
        
        print(f"  ✓ Saved interactive: {html_path}")
        print(f"  ✓ Saved static: {png_path}")
    
    def create_variant_distribution_figure(self, variant):
        """
        Create distribution figures for a specific variant
        
        Parameters:
        -----------
        variant : str
            Variant name ('xxx', 'xxx_gender', 'xxx_age')
        """
        print(f"\n{'='*80}")
        print(f"CREATING DISTRIBUTION FIGURES FOR {variant.upper()}")
        print(f"{'='*80}")
        
        # Get patients for this variant
        patient_ids = self.get_variant_patients(variant)
        print(f"\nTotal patients in {variant}: {len(patient_ids):,}")
        
        if len(patient_ids) == 0:
            print(f"⚠ No patients found for variant {variant}")
            return
        
        # Filter demographics to this variant
        demo_variant = self.demographics[
            self.demographics['primaryid'].isin(patient_ids)
        ].copy()
        
        # Determine what to visualize based on variant
        if variant == 'xxx':
            # For base XXX, show overall statistics
            self.create_xxx_distribution(demo_variant)
        elif variant == 'xxx_gender':
            # For XXX-Gender, show gender distribution
            self.create_gender_distribution(demo_variant)
        elif variant == 'xxx_age':
            # For XXX-Age, show age distribution
            self.create_age_distribution(demo_variant)
    
    def create_xxx_distribution(self, demo_variant):
        """Create distribution figures for base XXX variant"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gender distribution
        gender_counts = demo_variant['Gender'].value_counts()
        gender_data = [gender_counts.get('F', 0), gender_counts.get('M', 0)]
        gender_labels = ['Female', 'Male']
        colors_gender = ['#E8A5C6', '#88CCE8']
        
        self.create_donut_chart(
            gender_data,
            gender_labels,
            colors_gender,
            'Gender',
            axes[0]
        )
        axes[0].set_title('Gender Distribution', fontsize=13, fontweight='bold', pad=20)
        
        # Age distribution
        demo_age = demo_variant[
            (demo_variant['AGE'] >= 0) & (demo_variant['AGE'] <= 120)
        ].copy()
        
        def categorize_age(age):
            if age < 18:
                return 'Youth'
            elif age < 65:
                return 'Adult'
            else:
                return 'Elderly'
        
        demo_age['age_group'] = demo_age['AGE'].apply(categorize_age)
        age_counts = demo_age['age_group'].value_counts()
        
        age_order = ['Youth', 'Adult', 'Elderly']
        age_data = [age_counts.get(g, 0) for g in age_order]
        colors_age = ['#A8D5BA', '#FFB84D', '#FF6B6B']
        
        self.create_donut_chart(
            age_data,
            age_order,
            colors_age,
            'Age Group',
            axes[1]
        )
        axes[1].set_title('Age Distribution', fontsize=13, fontweight='bold', pad=20)
        
        plt.suptitle('Dataset: XXX - Overall Distribution', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        filepath = f"{self.output_path}/xxx_overall_distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def create_gender_distribution(self, demo_variant):
        """Create distribution figures for XXX-Gender variant"""
        # Filter to valid genders
        demo_gender = demo_variant[demo_variant['Gender'].isin(['F', 'M'])].copy()
        
        # Get drugs and ADRs for these patients
        drugs_gender = self.drugs[
            self.drugs['primaryid'].isin(demo_gender['primaryid'])
        ].merge(demo_gender[['primaryid', 'Gender']], on='primaryid')
        
        adrs_gender = self.adverse_reactions[
            self.adverse_reactions['primaryid'].isin(demo_gender['primaryid'])
        ].merge(demo_gender[['primaryid', 'Gender']], on='primaryid')
        
        # Find most common drug
        top_drug = drugs_gender['DRUG'].value_counts().index[0]
        
        # Find most common ADR for that drug
        drug_patients = drugs_gender[drugs_gender['DRUG'] == top_drug]['primaryid'].unique()
        drug_adrs = adrs_gender[adrs_gender['primaryid'].isin(drug_patients)]
        top_adr = drug_adrs['ADVERSE_EVENT'].value_counts().index[0]
        
        print(f"\nTop drug: {top_drug}")
        print(f"Top ADR: {top_adr}")
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        
        # Drug gender distribution
        drug_gender = drugs_gender[drugs_gender['DRUG'] == top_drug]['Gender'].value_counts()
        drug_data = [drug_gender.get('F', 0), drug_gender.get('M', 0)]
        drug_labels = ['Female', 'Male']
        colors_gender = ['#E8A5C6', '#88CCE8']
        
        self.create_donut_chart(
            drug_data,
            drug_labels,
            colors_gender,
            'Gender',
            axes[0]
        )
        axes[0].set_title(f'Drug: {top_drug}', fontsize=13, fontweight='bold', pad=20)
        
        # ADR gender distribution
        adr_gender = drug_adrs[drug_adrs['ADVERSE_EVENT'] == top_adr]['Gender'].value_counts()
        adr_data = [adr_gender.get('F', 0), adr_gender.get('M', 0)]
        
        self.create_donut_chart(
            adr_data,
            drug_labels,
            colors_gender,
            'Gender',
            axes[1]
        )
        axes[1].set_title(f'ADR: {top_adr}', fontsize=13, fontweight='bold', pad=20)
        
        plt.suptitle('Dataset: XXX-Gender - Gender Distribution', 
                     fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filepath = f"{self.output_path}/xxx_gender_distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def create_age_distribution(self, demo_variant):
        """Create distribution figures for XXX-Age variant"""
        # Filter to valid ages
        demo_age = demo_variant[
            (demo_variant['AGE'] >= 0) & (demo_variant['AGE'] <= 120)
        ].copy()
        
        def categorize_age(age):
            if age < 18:
                return 'Youth'
            elif age < 65:
                return 'Adult'
            else:
                return 'Elderly'
        
        demo_age['age_group'] = demo_age['AGE'].apply(categorize_age)
        
        # Get drugs and ADRs for these patients
        drugs_age = self.drugs[
            self.drugs['primaryid'].isin(demo_age['primaryid'])
        ].merge(demo_age[['primaryid', 'age_group']], on='primaryid')
        
        adrs_age = self.adverse_reactions[
            self.adverse_reactions['primaryid'].isin(demo_age['primaryid'])
        ].merge(demo_age[['primaryid', 'age_group']], on='primaryid')
        
        # Find most common drug
        top_drug = drugs_age['DRUG'].value_counts().index[0]
        
        # Find most common ADR for that drug
        drug_patients = drugs_age[drugs_age['DRUG'] == top_drug]['primaryid'].unique()
        drug_adrs = adrs_age[adrs_age['primaryid'].isin(drug_patients)]
        top_adr = drug_adrs['ADVERSE_EVENT'].value_counts().index[0]
        
        print(f"\nTop drug: {top_drug}")
        print(f"Top ADR: {top_adr}")
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        
        age_order = ['Youth', 'Adult', 'Elderly']
        colors_age = ['#A8D5BA', '#FFB84D', '#FF6B6B']
        
        # Drug age distribution
        drug_age = drugs_age[drugs_age['DRUG'] == top_drug]['age_group'].value_counts()
        drug_data = [drug_age.get(g, 0) for g in age_order]
        
        self.create_donut_chart(
            drug_data,
            age_order,
            colors_age,
            'Age Group',
            axes[0]
        )
        axes[0].set_title(f'Drug: {top_drug}', fontsize=13, fontweight='bold', pad=20)
        
        # ADR age distribution
        adr_age = drug_adrs[drug_adrs['ADVERSE_EVENT'] == top_adr]['age_group'].value_counts()
        adr_data = [adr_age.get(g, 0) for g in age_order]
        
        self.create_donut_chart(
            adr_data,
            age_order,
            colors_age,
            'Age Group',
            axes[1]
        )
        axes[1].set_title(f'ADR: {top_adr}', fontsize=13, fontweight='bold', pad=20)
        
        plt.suptitle('Dataset: XXX-Age - Age Distribution', 
                     fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filepath = f"{self.output_path}/xxx_age_distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def create_dataset_statistics_table(self):
        """Create a comparison table of dataset statistics"""
        print(f"\n{'='*80}")
        print("CREATING DATASET STATISTICS TABLE")
        print(f"{'='*80}")
        
        stats = []
        
        for variant in ['xxx', 'xxx_gender', 'xxx_age']:
            patient_ids = self.get_variant_patients(variant)
            
            if len(patient_ids) == 0:
                continue
            
            # Get data for this variant
            demo = self.demographics[self.demographics['primaryid'].isin(patient_ids)]
            drugs_var = self.drugs[self.drugs['primaryid'].isin(patient_ids)]
            adrs_var = self.adverse_reactions[
                self.adverse_reactions['primaryid'].isin(patient_ids)
            ]
            
            # Calculate statistics
            n_patients = len(patient_ids)
            n_drugs = drugs_var['DRUG'].nunique()
            n_adrs = adrs_var['ADVERSE_EVENT'].nunique()
            
            # Gender distribution
            gender_counts = demo['Gender'].value_counts()
            pct_female = (gender_counts.get('F', 0) / n_patients * 100) if n_patients > 0 else 0
            pct_male = (gender_counts.get('M', 0) / n_patients * 100) if n_patients > 0 else 0
            
            # Age distribution
            demo_age = demo[(demo['AGE'] >= 0) & (demo['AGE'] <= 120)]
            avg_age = demo_age['AGE'].mean() if len(demo_age) > 0 else 0
            
            stats.append({
                'Variant': variant.upper(),
                'Patients': n_patients,
                'Drugs': n_drugs,
                'ADRs': n_adrs,
                'Female %': f"{pct_female:.1f}",
                'Male %': f"{pct_male:.1f}",
                'Avg Age': f"{avg_age:.1f}"
            })
        
        # Create DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Save as CSV
        csv_path = f"{self.output_path}/dataset_statistics.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved statistics table: {csv_path}")
        
        # Print table
        print("\nDataset Statistics:")
        print(stats_df.to_string(index=False))
        
        return stats_df
    
    def create_gender_sankey_for_variant(self, variant='xxx_gender', top_n_adrs=10):
        """
        Create gender Sankey diagram for XXX-Gender variant
        
        Parameters:
        -----------
        variant : str
            Variant name (default: 'xxx_gender')
        top_n_adrs : int
            Number of top ADRs to include (default: 10)
        """
        print(f"\n{'='*80}")
        print(f"CREATING GENDER SANKEY DIAGRAM FOR {variant.upper()}")
        print(f"{'='*80}")
        
        # Get patients for this variant
        patient_ids = self.get_variant_patients(variant)
        
        if len(patient_ids) == 0:
            print(f"⚠ No patients found for variant {variant}")
            return
        
        # Filter data to this variant with valid genders
        demo_variant = self.demographics[
            (self.demographics['primaryid'].isin(patient_ids)) &
            (self.demographics['Gender'].isin(['F', 'M']))
        ].copy()
        
        drugs_gender = self.drugs[
            self.drugs['primaryid'].isin(demo_variant['primaryid'])
        ].merge(demo_variant[['primaryid', 'Gender']], on='primaryid')
        
        adrs_gender = self.adverse_reactions[
            self.adverse_reactions['primaryid'].isin(demo_variant['primaryid'])
        ].merge(demo_variant[['primaryid', 'Gender']], on='primaryid')
        
        # Find top drug
        # top_drug = drugs_gender['DRUG'].value_counts().index[0]
        top_drug = "rifampin"  # Override for better visualization
        print(f"\nAnalyzing drug: {top_drug}")
        
        # Get top N ADRs for this drug
        drug_patients = set(drugs_gender[drugs_gender['DRUG'] == top_drug]['primaryid'])
        drug_adrs = adrs_gender[adrs_gender['primaryid'].isin(drug_patients)]
        
        top_adrs = drug_adrs['ADVERSE_EVENT'].value_counts().head(top_n_adrs).index.tolist()
        print(f"Including top {len(top_adrs)} ADRs")
        
        # Calculate gender counts for each ADR
        adr_gender_dict = {}
        
        for adr in top_adrs:
            adr_patients = set(drug_adrs[drug_adrs['ADVERSE_EVENT'] == adr]['primaryid'])
            combined_patients = drug_patients.intersection(adr_patients)
            
            if len(combined_patients) == 0:
                continue
            
            demo_combined = demo_variant[demo_variant['primaryid'].isin(combined_patients)]
            gender_counts = demo_combined['Gender'].value_counts()
            
            female_count = gender_counts.get('F', 0)
            male_count = gender_counts.get('M', 0)
            
            adr_gender_dict[adr] = (female_count, male_count)
        
        if len(adr_gender_dict) == 0:
            print("⚠ No valid ADRs found for Sankey diagram")
            return
        
        # Create Sankey diagram
        self.create_gender_sankey_plotly(
            top_drug,
            adr_gender_dict,
            f"{variant}_gender_sankey"
        )
    
    def create_age_sankey_for_variant(self, variant='xxx_age', top_n_adrs=10):
        """
        Create age Sankey diagram for XXX-Age variant
        
        Parameters:
        -----------
        variant : str
            Variant name (default: 'xxx_age')
        top_n_adrs : int
            Number of top ADRs to include (default: 10)
        """
        print(f"\n{'='*80}")
        print(f"CREATING AGE SANKEY DIAGRAM FOR {variant.upper()}")
        print(f"{'='*80}")
        
        # Get patients for this variant
        patient_ids = self.get_variant_patients(variant)
        
        if len(patient_ids) == 0:
            print(f"⚠ No patients found for variant {variant}")
            return
        
        # Filter data to this variant with valid ages
        demo_variant = self.demographics[
            (self.demographics['primaryid'].isin(patient_ids)) &
            (self.demographics['AGE'] >= 0) &
            (self.demographics['AGE'] <= 120)
        ].copy()
        
        def categorize_age(age):
            if age < 18:
                return 'Youth'
            elif age < 65:
                return 'Adult'
            else:
                return 'Elderly'
        
        demo_variant['age_group'] = demo_variant['AGE'].apply(categorize_age)
        
        drugs_age = self.drugs[
            self.drugs['primaryid'].isin(demo_variant['primaryid'])
        ].merge(demo_variant[['primaryid', 'age_group']], on='primaryid')
        
        adrs_age = self.adverse_reactions[
            self.adverse_reactions['primaryid'].isin(demo_variant['primaryid'])
        ].merge(demo_variant[['primaryid', 'age_group']], on='primaryid')
        
        # Find top drug
        # top_drug = drugs_age['DRUG'].value_counts().index[0]
        top_drug = "rifampin"
        print(f"\nAnalyzing drug: {top_drug}")
        
        # Get top N ADRs for this drug
        drug_patients = set(drugs_age[drugs_age['DRUG'] == top_drug]['primaryid'])
        drug_adrs = adrs_age[adrs_age['primaryid'].isin(drug_patients)]
        
        top_adrs = drug_adrs['ADVERSE_EVENT'].value_counts().head(top_n_adrs).index.tolist()
        print(f"Including top {len(top_adrs)} ADRs")
        
        # Calculate age counts for each ADR
        adr_age_dict = {}
        
        for adr in top_adrs:
            adr_patients = set(drug_adrs[drug_adrs['ADVERSE_EVENT'] == adr]['primaryid'])
            combined_patients = drug_patients.intersection(adr_patients)
            
            if len(combined_patients) == 0:
                continue
            
            demo_combined = demo_variant[demo_variant['primaryid'].isin(combined_patients)]
            age_counts = demo_combined['age_group'].value_counts()
            
            youth_count = age_counts.get('Youth', 0)
            adult_count = age_counts.get('Adult', 0)
            elderly_count = age_counts.get('Elderly', 0)
            
            adr_age_dict[adr] = (youth_count, adult_count, elderly_count)
        
        if len(adr_age_dict) == 0:
            print("⚠ No valid ADRs found for Sankey diagram")
            return
        
        # Create Sankey diagram
        self.create_age_sankey_plotly(
            top_drug,
            adr_age_dict,
            f"{variant}_age_sankey"
        )
    
    def create_all_figures(self, include_sankey=True):
        """
        Create all visualization figures
        
        Parameters:
        -----------
        include_sankey : bool
            Whether to include Sankey diagrams (default: True)
        """
        print("\n" + "="*80)
        print("CREATING ALL VARIANT FIGURES")
        print("="*80)
        
        # Create distribution figures for each variant
        for variant in ['xxx', 'xxx_gender', 'xxx_age']:
            if variant in self.splits:
                self.create_variant_distribution_figure(variant)
        
        # Create Sankey diagrams
        if include_sankey:
            if 'xxx_gender' in self.splits:
                self.create_gender_sankey_for_variant('xxx_gender')
            
            if 'xxx_age' in self.splits:
                self.create_age_sankey_for_variant('xxx_age')
        
        # Create statistics table
        self.create_dataset_statistics_table()
        
        print("\n" + "="*80)
        print("✓ ALL FIGURES CREATED SUCCESSFULLY!")
        print("="*80)
        print(f"\nFigures saved to: {self.output_path}")
        print("\nCreated files:")
        print("  • xxx_overall_distribution.png")
        print("  • xxx_gender_distribution.png")
        print("  • xxx_age_distribution.png")
        print("  • dataset_statistics.csv")
        
        if include_sankey:
            print("\n  Sankey Diagrams (interactive HTML + PNG):")
            print("  • xxx_gender_gender_sankey.html / .png")
            print("  • xxx_age_age_sankey.html / .png")
        print()


def main():
    """
    Main function to create all figures
    """
    print("\n" + "="*80)
    print("VARIANT FIGURE CREATOR")
    print("="*80)
    
    # Create figure generator
    creator = VariantFigureCreator()
    
    # Create all figures
    creator.create_all_figures()


if __name__ == "__main__":
    main()
