"""
Create demographic figures like Figure 2(d), (e), (f), (g) from the PreciseADR paper
Uses saved datasets and test results from preprocessing pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DemographicFigureCreator:
    """
    Create demographic visualization figures from FAERS processed data
    """
    
    def __init__(self, data_path, output_path="../output/demographic_figures"):
        """
        Initialize with paths to data
        
        Parameters:
        -----------
        data_path : str
            Path to preprocessed FAERS data directory
        output_path : str
            Path to save output figures
        """
        self.data_path = data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load all necessary datasets"""
        print("="*80)
        print("LOADING DATA FOR FIGURE GENERATION")
        print("="*80)
        
        # Load main XXX dataset
        print("\nLoading XXX dataset...")
        self.demographics = pd.read_csv(f"{self.data_path}/XXX/demographics.csv")
        self.drugs = pd.read_csv(f"{self.data_path}/XXX/drugs_standardized.csv")
        self.adverse_reactions = pd.read_csv(f"{self.data_path}/XXX/adverse_reactions.csv")
        
        print(f"  ✓ Demographics: {len(self.demographics):,} records")
        print(f"  ✓ Drugs: {len(self.drugs):,} records")
        print(f"  ✓ Adverse Reactions: {len(self.adverse_reactions):,} records")
        
        # Load test results
        print("\nLoading test results...")
        self.gender_test_results = pd.read_csv(f"{self.data_path}/gender_test_results.csv")
        self.age_test_results = pd.read_csv(f"{self.data_path}/age_test_results.csv")
        
        print(f"  ✓ Gender test results: {len(self.gender_test_results):,} pairs")
        print(f"  ✓ Age test results: {len(self.age_test_results):,} pairs")
        
        print("\n✓ Data loading complete!\n")
    
    def create_donut_chart(self, data, labels, colors, title, ax):
        """
        Create a donut chart with labels inside the chart (like panels d and f)
        
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
        
        # Add labels with percentages INSIDE the donut segments
        total = sum(data)
        for i, (wedge, label) in enumerate(zip(wedges, labels)):
            # Calculate the angle for positioning
            angle = (wedge.theta2 + wedge.theta1) / 2
            
            # Position on the outer part of the donut (between inner and outer radius)
            # radius = 0.75 means 75% from center (middle of the donut ring)
            radius = 0.75
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            
            # Calculate percentage
            pct = data[i] / total * 100
            
            # Determine text color (white or black) based on background color
            # Convert hex color to RGB and calculate luminance
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(colors[i])
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = 'white' if luminance < 0.5 else 'black'
            
            # Add text inside the segment
            ax.text(
                x, y,
                f"{label}\n{pct:.1f}%",
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                color=text_color
            )
        
        # Add title in center of donut
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
                'text': f"<b>Figure: Gender Demographics for {drug_name}</b>",
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
                'text': f"<b>Figure: Age Demographics for {drug_name}</b>",
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
    
    def create_figure_gender(self, drug_name=None, adr_name=None, top_n_adrs=10):
        """
        Create Figure - Gender-related demographics
        
        Parameters:
        -----------
        drug_name : str, optional
            Drug to analyze (if None, uses top result)
        adr_name : str, optional
            ADR to analyze (if None, uses top result)
        top_n_adrs : int
            Number of top ADRs to show in Sankey (default: 10)
        """
        print("="*80)
        print("CREATING FIGURE - GENDER DEMOGRAPHICS")
        print("="*80)
        
        # Find best gender-related drug-ADR pair from test results
        gender_significant = self.gender_test_results[
            self.gender_test_results['is_gender_related'] == True
        ].copy()
        
        # Filter out infinite odds ratios
        gender_significant = gender_significant[
            (gender_significant['odds_ratio'] != np.inf) &
            (gender_significant['odds_ratio'] > 0) &
            (gender_significant['odds_ratio'] < 100)
        ]
        
        if len(gender_significant) == 0:
            print("⚠ No valid gender-related pairs found")
            return
        
        # Try to find specified drug-ADR pair, otherwise use top result
        if drug_name and adr_name:
            specific_pair = gender_significant[
                (gender_significant['DRUG'].str.lower().str.contains(drug_name.lower())) &
                (gender_significant['ADVERSE_EVENT'].str.lower().str.contains(adr_name.lower()))
            ]
            
            if len(specific_pair) > 0:
                selected = specific_pair.iloc[0]
                print(f"\n✓ Found specified pair: {selected['DRUG']} - {selected['ADVERSE_EVENT']}")
            else:
                gender_significant_sorted = gender_significant.sort_values('odds_ratio', ascending=False)
                selected = gender_significant_sorted.iloc[0]
                print(f"\n⚠ Specified pair not found, using: {selected['DRUG']} - {selected['ADVERSE_EVENT']}")
        else:
            # Use top result
            gender_significant_sorted = gender_significant.sort_values('odds_ratio', ascending=False)
            selected = gender_significant_sorted.iloc[0]
            print(f"\n✓ Using top gender-related pair: {selected['DRUG']} - {selected['ADVERSE_EVENT']}")
        
        drug_name = selected['DRUG']
        adr_name = selected['ADVERSE_EVENT']
        
        # Prepare data with gender
        demo_with_gender = self.demographics[
            self.demographics['Gender'].isin(['M', 'F'])
        ].copy()
        
        drugs_gender = self.drugs.merge(
            demo_with_gender[['primaryid', 'Gender']], 
            on='primaryid'
        )
        
        adrs_gender = self.adverse_reactions.merge(
            demo_with_gender[['primaryid', 'Gender']], 
            on='primaryid'
        )
        
        # ========================================
        # Figure: Donut charts
        # ========================================
        
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        
        # Drug gender distribution
        drug_gender = drugs_gender[
            drugs_gender['DRUG'] == drug_name
        ]['Gender'].value_counts()
        
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
        axes[0].set_title(f'Drug: {drug_name}', fontsize=13, fontweight='bold', pad=20)
        
        # ADR gender distribution
        adr_gender = adrs_gender[
            adrs_gender['ADVERSE_EVENT'] == adr_name
        ]['Gender'].value_counts()
        
        adr_data = [adr_gender.get('F', 0), adr_gender.get('M', 0)]
        
        self.create_donut_chart(
            adr_data,
            drug_labels,
            colors_gender,
            'Gender',
            axes[1]
        )
        axes[1].set_title(f'ADR: {adr_name}', fontsize=13, fontweight='bold', pad=20)
        
        plt.suptitle('Figure: Gender Distribution', fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filepath = f"{self.output_path}/figure_gender_distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: {filepath}")
        plt.close()
        
        # ========================================
        # Figure: Plotly Sankey diagram
        # ========================================
        
        print("\nCreating Figure: Gender Sankey diagram...")
        
        # Get top N ADRs for this drug by gender association
        drug_adrs = gender_significant[
            gender_significant['DRUG'] == drug_name
        ].sort_values('odds_ratio', ascending=False).head(top_n_adrs)
        
        if len(drug_adrs) == 0:
            print(f"⚠ No significant ADRs found for drug {drug_name}")
            return
        
        # Calculate actual counts for each ADR with this drug
        adr_gender_dict = {}
        
        for _, row in drug_adrs.iterrows():
            adr = row['ADVERSE_EVENT']
            
            # Get primaryids with this drug AND this ADR
            drug_ids = set(drugs_gender[drugs_gender['DRUG'] == drug_name]['primaryid'])
            adr_ids = set(adrs_gender[adrs_gender['ADVERSE_EVENT'] == adr]['primaryid'])
            combined_ids = drug_ids.intersection(adr_ids)
            
            if len(combined_ids) == 0:
                continue
            
            # Get gender counts
            combined_demo = demo_with_gender[
                demo_with_gender['primaryid'].isin(combined_ids)
            ]
            
            gender_counts = combined_demo['Gender'].value_counts()
            
            female_count = gender_counts.get('F', 0)
            male_count = gender_counts.get('M', 0)
            
            adr_gender_dict[adr] = (female_count, male_count)
        
        # Create Plotly Sankey
        self.create_gender_sankey_plotly(
            drug_name,
            adr_gender_dict,
            "figure_gender_sankey"
        )
    
    def create_figure_age(self, drug_name=None, adr_name=None, top_n_adrs=10):
        """
        Create Figure - Age-related demographics
        
        Parameters:
        -----------
        drug_name : str, optional
            Drug to analyze (if None, uses top result)
        adr_name : str, optional
            ADR to analyze (if None, uses top result)
        top_n_adrs : int
            Number of top ADRs to show in Sankey (default: 10)
        """
        print("\n" + "="*80)
        print("CREATING FIGURE - AGE DEMOGRAPHICS")
        print("="*80)
        
        # Find best age-related drug-ADR pair
        age_significant = self.age_test_results[
            self.age_test_results['is_age_related'] == True
        ].copy()
        
        # Filter out infinite/extreme risk ratios
        age_significant = age_significant[
            (age_significant['risk_ratio'] != np.inf) &
            (age_significant['risk_ratio'] > 0) &
            (age_significant['risk_ratio'] < 20)
        ]
        
        if len(age_significant) == 0:
            print("⚠ No valid age-related pairs found")
            return
        
        # Try to find specified drug-ADR pair
        if drug_name and adr_name:
            specific_pair = age_significant[
                (age_significant['DRUG'].str.lower().str.contains(drug_name.lower())) &
                (age_significant['ADVERSE_EVENT'].str.lower().str.contains(adr_name.lower()))
            ]
            
            if len(specific_pair) > 0:
                selected = specific_pair.iloc[0]
                print(f"\n✓ Found specified pair: {selected['DRUG']} - {selected['ADVERSE_EVENT']}")
            else:
                age_significant_sorted = age_significant.sort_values('risk_ratio', ascending=False)
                selected = age_significant_sorted.iloc[0]
                print(f"\n⚠ Specified pair not found, using: {selected['DRUG']} - {selected['ADVERSE_EVENT']}")
        else:
            # Use top result
            age_significant_sorted = age_significant.sort_values('risk_ratio', ascending=False)
            selected = age_significant_sorted.iloc[0]
            print(f"\n✓ Using top age-related pair: {selected['DRUG']} - {selected['ADVERSE_EVENT']}")
        
        drug_name = selected['DRUG']
        adr_name = selected['ADVERSE_EVENT']
        
        # Prepare data with age
        demo_with_age = self.demographics[
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
        
        demo_with_age['age_group'] = demo_with_age['AGE'].apply(categorize_age)
        
        drugs_age = self.drugs.merge(
            demo_with_age[['primaryid', 'age_group']], 
            on='primaryid'
        )
        
        adrs_age = self.adverse_reactions.merge(
            demo_with_age[['primaryid', 'age_group']], 
            on='primaryid'
        )
        
        # ========================================
        # Figure: Donut charts
        # ========================================
        
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        
        # Drug age distribution
        drug_age = drugs_age[
            drugs_age['DRUG'] == drug_name
        ]['age_group'].value_counts()
        
        age_order = ['Youth', 'Adult', 'Elderly']
        drug_data = [drug_age.get(g, 0) for g in age_order]
        colors_age = ['#A8D5BA', '#FFB84D', '#FF6B6B']
        
        self.create_donut_chart(
            drug_data,
            age_order,
            colors_age,
            'Age',
            axes[0]
        )
        axes[0].set_title(f'Drug: {drug_name}', fontsize=13, fontweight='bold', pad=20)
        
        # ADR age distribution
        adr_age = adrs_age[
            adrs_age['ADVERSE_EVENT'] == adr_name
        ]['age_group'].value_counts()
        
        adr_data = [adr_age.get(g, 0) for g in age_order]
        
        self.create_donut_chart(
            adr_data,
            age_order,
            colors_age,
            'Age',
            axes[1]
        )
        axes[1].set_title(f'ADR: {adr_name}', fontsize=13, fontweight='bold', pad=20)
        
        plt.suptitle('Figure: Age Distribution', fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filepath = f"{self.output_path}/figure_age_distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: {filepath}")
        plt.close()
        
        # ========================================
        # Figure: Plotly Sankey diagram
        # ========================================
        
        print("\nCreating Figure: Age Sankey diagram...")
        
        # Get top N ADRs for this drug by age association
        drug_adrs = age_significant[
            age_significant['DRUG'] == drug_name
        ].sort_values('risk_ratio', ascending=False).head(top_n_adrs)
        
        if len(drug_adrs) == 0:
            print(f"⚠ No significant ADRs found for drug {drug_name}")
            return
        
        # Calculate actual counts for each ADR with this drug
        adr_age_dict = {}
        
        for _, row in drug_adrs.iterrows():
            adr = row['ADVERSE_EVENT']
            
            # Get primaryids with this drug AND this ADR
            drug_ids = set(drugs_age[drugs_age['DRUG'] == drug_name]['primaryid'])
            adr_ids = set(adrs_age[adrs_age['ADVERSE_EVENT'] == adr]['primaryid'])
            combined_ids = drug_ids.intersection(adr_ids)
            
            if len(combined_ids) == 0:
                continue
            
            # Get age counts
            combined_demo = demo_with_age[
                demo_with_age['primaryid'].isin(combined_ids)
            ]
            
            age_counts = combined_demo['age_group'].value_counts()
            
            youth_count = age_counts.get('Youth', 0)
            adult_count = age_counts.get('Adult', 0)
            elderly_count = age_counts.get('Elderly', 0)
            
            adr_age_dict[adr] = (youth_count, adult_count, elderly_count)
        
        # Create Plotly Sankey
        self.create_age_sankey_plotly(
            drug_name,
            adr_age_dict,
            "figure_age_sankey"
        )

    def create_all_figures(self, gender_drug=None, gender_adr=None, age_drug=None, age_adr=None):
        """
        Create all demographic figures
        
        Parameters:
        -----------
        gender_drug : str, optional
            Specific drug for gender analysis
        gender_adr : str, optional
            Specific ADR for gender analysis
        age_drug : str, optional
            Specific drug for age analysis
        age_adr : str, optional
            Specific ADR for age analysis
        """
        print("\n" + "="*80)
        print("CREATING ALL DEMOGRAPHIC FIGURES")
        print("="*80 + "\n")
        
        # Gender figures
        self.create_figure_gender(drug_name=gender_drug, adr_name=gender_adr)
        
        # Age figures
        self.create_figure_age(drug_name=age_drug, adr_name=age_adr)
        
        print("\n" + "="*80)
        print("✓ ALL FIGURES CREATED SUCCESSFULLY!")
        print("="*80)
        print(f"\nFigures saved to: {self.output_path}")
        print("\nCreated files:")
        print("  • figure_gender_distribution.png")
        print("  • figure_gender_sankey.html (interactive)")
        print("  • figure_gender_sankey.png (static)")
        print("  • figure_age_distribution.png")
        print("  • figure_age_sankey.html (interactive)")
        print("  • figure_age_sankey.png (static)")
        print("\n")

def main():
    """
    Main function to create all figures
    """
    # Configuration
    DATA_PATH = "../data/preprocessed_faers_optimized"  # Update this path
    OUTPUT_PATH = "../output/demographic_figures"  # Update this path
    
    # Create figure generator
    creator = DemographicFigureCreator(DATA_PATH, OUTPUT_PATH)
    
    # Create all figures
    # creator.create_all_figures()

    # Option 2: Create figures with specific drugs/ADRs
    creator.create_all_figures(
        gender_drug='gemcitabine',
        gender_adr='vomiting',
        age_drug='cyclophosphamide”',
        age_adr='vomiting'
    )


if __name__ == "__main__":
    main()