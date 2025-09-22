import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import re
import pytz
# --- Ajuste global de fontes ---
plt.rcParams.update({
    'font.size': 20,          # tamanho geral do texto
    'axes.titlesize': 20,     # título dos gráficos
    'axes.labelsize': 20,     # labels dos eixos
    'xtick.labelsize': 20,    # rótulos do eixo X
    'ytick.labelsize': 20,    # rótulos do eixo Y
    'legend.fontsize': 20     # legenda
  })  
# --- Configuration ---
# Excel file names
EXCEL_FILE_WATER = 'Physicochemical Parameters table for water.xlsx'

# Output directories for plots
OUTPUT_DIR = 'Plots_Analysis_Final'
WATER_PLOTS_DIR = os.path.join(OUTPUT_DIR, 'Water_Plots')
PARAM_BY_LOCAL_WATER_PLOTS_DIR = os.path.join(OUTPUT_DIR, 'Parameter_by_Local_Water_Plots')
CORRELATION_MATRICES_DIR = os.path.join(OUTPUT_DIR, 'Correlation_Matrices')
DETAILED_TIME_SERIES_PLOTS_DIR = os.path.join(OUTPUT_DIR, 'Detailed_Time_Series_Plots')

# Column name mapping: (Excel_Column_Name: Standardized_Script_Name)
COL_NAME_MAPPING = {
    'Temperature (°C)': 'Temperature (C°)',
    'Temp (C°)': 'Temperature (C°)',
    'Temperature (C)': 'Temperature (C°)',
    'RDO in %': 'RDO (%)',
    'RDO%': 'RDO (%)',
    'EL.Cond.(µS/cm)': 'El. Cond. (µS/cm)',
    'El. Cond. (µs/cm)': 'El. Cond. (µS/cm)',
    'TDS (ppt)': 'TDS (ppm)',
    'TDS (ppm)': 'TDS (ppm)',
    'Salinity (psu)': 'Salinity (psu)',
    'Salinity in %': 'Salinity in %',
    'pH': 'pH',
    'RDO (mg/L)': 'RDO (mg/L)',
    'Samples': 'Samples',
    'Time': 'Time',
    'Date': 'Data'  # Corrigido: Mapeamento da coluna 'Date' para 'Data'
}

# Parameters for analysis and plotting (using standardized script names)
PARAMS_TO_ANALYZE = [
    'pH', 'Temperature (C°)', 'RDO (mg/L)', 'RDO (%)',
    'El. Cond. (µS/cm)', 'Salinity (psu)', 'TDS (ppm)'
]

# Color palette for saltworks
SALTWORK_PALETTE = {
    '1': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), # Blue
    '2': (1.0, 0.4980392156862745, 0.054901960784313725),             # Orange
}

# Create output directories if they don't exist
os.makedirs(WATER_PLOTS_DIR, exist_ok=True)
os.makedirs(PARAM_BY_LOCAL_WATER_PLOTS_DIR, exist_ok=True)
os.makedirs(CORRELATION_MATRICES_DIR, exist_ok=True)
os.makedirs(DETAILED_TIME_SERIES_PLOTS_DIR, exist_ok=True)

# --- Data Processing Functions ---

def extract_saltwork_stage(sample_name):
    """Extracts Saltwork and Stage from a sample name."""
    saltwork = None
    stage = None

    # Patterns to extract Saltwork, Stage, and Date
    match_cm = re.match(r'(\d)([A-D])CM', sample_name)
    match_wred_cm = re.match(r'(WRED)CM', sample_name)
    match_simple_saltwork_stage = re.match(r'(\d)([A-D])_', sample_name)
    match_wred_simple = re.match(r'(WRED)_', sample_name)

    if match_cm:
        saltwork = match_cm.group(1)
        stage = match_cm.group(2)
    elif match_wred_cm:
        saltwork = match_wred_cm.group(1)
    elif match_simple_saltwork_stage:
        saltwork = match_simple_saltwork_stage.group(1)
        stage = match_simple_saltwork_stage.group(2)
    elif match_wred_simple:
        saltwork = match_wred_simple.group(1)

    if not saltwork:
        if sample_name.startswith('WRED'):
            saltwork = 'WRED'
        elif sample_name and sample_name[0].isdigit():
            saltwork = sample_name[0]
            if len(sample_name) > 1 and sample_name[1].isalpha():
                stage = sample_name[1]

    return saltwork, stage

def load_and_process_file(file_path, file_type):
    """Loads and processes all sheets from an Excel file."""
    print(f"Loading {file_type} data from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Skipping.")
        return pd.DataFrame()

    xls = pd.ExcelFile(file_path)
    all_dataframes = []

    for sheet_name in xls.sheet_names:
        if (file_type == 'Water' and sheet_name.startswith('W ')):
            
            df = pd.read_excel(xls, sheet_name=sheet_name)
            print(f"  Loaded sheet '{sheet_name}': {len(df)} rows")

            df.rename(columns={excel_col: script_col for excel_col, script_col in COL_NAME_MAPPING.items() if excel_col in df.columns}, inplace=True)
            
            df['Type'] = file_type

            df[['Saltwork', 'Stage']] = df['Samples'].apply(
                lambda x: pd.Series(extract_saltwork_stage(x))
            )
            
            # Reads the now-renamed 'Data' column
            if 'Data' in df.columns and 'Time' in df.columns:
                df['Full_DateTime'] = df.apply(
                    lambda row: pd.NaT if pd.isna(row['Data']) or pd.isna(row['Time']) else
                                pd.to_datetime(f"{row['Data']} {row['Time']}", errors='coerce'),
                    axis=1
                )
                df['Data'] = df['Full_DateTime'].dt.normalize()
                df = df.drop(columns=['Full_DateTime'], errors='ignore')
            elif 'Data' in df.columns:
                 df['Data'] = pd.to_datetime(df['Data'], errors='coerce')


            for col in PARAMS_TO_ANALYZE:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    print(f"  Warning: Column '{col}' (standardized name) not found in sheet '{sheet_name}'. It will be skipped for this sheet.")

            desired_cols_order = ['Samples', 'Type', 'Saltwork', 'Stage', 'Data', 'Time'] + [p for p in PARAMS_TO_ANALYZE if p in df.columns]
            df = df[[col for col in desired_cols_order if col in df.columns] + 
                    [col for col in df.columns if col not in desired_cols_order and col not in ['Time']]]

            all_dataframes.append(df)
        else:
            print(f"  Skipping sheet '{sheet_name}' (does not match {file_type} pattern).")

    if not all_dataframes:
        print(f"No relevant sheets found for {file_type} in {file_path}. Returning empty DataFrame.")
        return pd.DataFrame() 
    
    return pd.concat(all_dataframes, ignore_index=True)


# --- Plotting Functions ---

def generate_time_series_plots(df_filtered, plot_type):
    """Generates time series plots for selected parameters."""
    print(f"\nGenerating Time Series Plots for {plot_type}...")
    output_path = WATER_PLOTS_DIR

    df_filtered['Data'] = pd.to_datetime(df_filtered['Data'], errors='coerce')

    local_timezone = pytz.timezone('Europe/Lisbon')
    df_filtered['Data'] = df_filtered['Data'].dt.tz_localize(None).dt.tz_localize(local_timezone, ambiguous='NaT', nonexistent='NaT')
    df_filtered['Data_UTC'] = df_filtered['Data']

    for param in PARAMS_TO_ANALYZE:
        fig, ax = plt.subplots(figsize=(14, 7))
        df_plot = df_filtered[(df_filtered['Type'] == plot_type) & 
                              (df_filtered['Saltwork'].isin(['1', '2']))].copy()
        
        df_plot_clean = df_plot.dropna(subset=[param, 'Data_UTC'])

        if df_plot_clean.empty:
            print(f"  Skipping Time Series plot for '{param}' ({plot_type}): No valid data points found for plotting.")
            plt.close(fig)
            continue
        
        sns.lineplot(data=df_plot_clean, x='Data_UTC', y=param, hue='Saltwork', palette=SALTWORK_PALETTE, marker='o', errorbar=None, ax=ax)
        ax.set_title(f'{param} Over Time by Saltwork ({plot_type})')
        ax.set_xlabel('Date')
        ax.set_ylabel(param)
        ax.grid(True)
        
        unique_dates_utc = sorted(df_plot_clean['Data_UTC'].unique())
        
        if len(unique_dates_utc) > 0:
            ax.set_xticks(unique_dates_utc)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{param.replace("/", "").replace("°", "").replace("(","").replace(")","").replace(".","")}_TimeSeries{plot_type.replace(" ", "_")}.png'))
        plt.close(fig)
    print(f"  Finished Time Series Plots for {plot_type}.")


def generate_time_series_by_stage_plots(df_filtered, plot_type):
    """Generates time series plots for each parameter, for each stage, for each saltwork."""
    print(f"\nGenerating Detailed Time Series Plots (by Stage and Saltwork) for {plot_type}...")
    output_path = DETAILED_TIME_SERIES_PLOTS_DIR

    df_filtered['Data'] = pd.to_datetime(df_filtered['Data'], errors='coerce')

    local_timezone = pytz.timezone('Europe/Lisbon')
    df_filtered['Data'] = df_filtered['Data'].dt.tz_localize(None).dt.tz_localize(local_timezone, ambiguous='NaT', nonexistent='NaT')
    df_filtered['Data_UTC'] = df_filtered['Data']

    df_plot_base = df_filtered[(df_filtered['Type'] == plot_type) &
                               (df_filtered['Saltwork'].isin(['1', '2',])) &
                               (df_filtered['Stage'].isin(['A', 'B', 'C', 'D']))].copy()

    if df_plot_base.empty:
        print(f"  Skipping Detailed Time Series plots for {plot_type}: No valid data found for Saltwork (1,2,WRED) and Stages (A-D).")
        return

    unique_combinations = df_plot_base[['Saltwork', 'Stage']].drop_duplicates().sort_values(by=['Saltwork', 'Stage'])

    for param in PARAMS_TO_ANALYZE:
        for index, row in unique_combinations.iterrows():
            saltwork = row['Saltwork']
            stage = row['Stage']

            fig, ax = plt.subplots(figsize=(14, 7))
            df_subset = df_plot_base[(df_plot_base['Saltwork'] == saltwork) &
                                     (df_plot_base['Stage'] == stage)].dropna(subset=[param, 'Data_UTC'])

            if df_subset.empty:
                print(f"  Skipping Detailed Time Series plot for '{param}' - Saltwork {saltwork}, Stage {stage}: No valid data points found.")
                plt.close(fig)
                continue
            
            sns.lineplot(data=df_subset, x='Data_UTC', y=param, marker='o', color=SALTWORK_PALETTE.get(saltwork, 'gray'), ax=ax)
            
            ax.set_title(f'{param} Over Time - Saltwork {saltwork}, Stage {stage} ({plot_type})')
            ax.set_xlabel('Date')
            ax.set_ylabel(param)
            ax.grid(True)

            unique_dates_utc = sorted(df_subset['Data_UTC'].unique())
            
            if len(unique_dates_utc) > 0:
                ax.set_xticks(unique_dates_utc)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            fig.autofmt_xdate()

            filename = f'{param.replace("/", "").replace("°", "").replace("(","").replace(")","").replace(".","")}_TimeSeries_Saltwork{saltwork}_Stage{stage}{plot_type.replace(" ", "_")}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, filename))
            plt.close(fig)
        print(f"  Finished Detailed Time Series Plots for '{param}'.")
    print(f"Finished generating Detailed Time Series Plots (by Stage and Saltwork) for {plot_type}.")


def generate_comparison_bar_plots(df_filtered, plot_type):
    """Generates comparison bar plots of the mean by saltwork."""
    print(f"\nGenerating Comparison Bar Plots for {plot_type}...")
    output_path = WATER_PLOTS_DIR

    for param in PARAMS_TO_ANALYZE:
        plt.figure(figsize=(10, 6))
        df_plot = df_filtered[df_filtered['Type'] == plot_type].copy()

        if df_plot[param].dropna().empty:
            print(f"  Skipping Comparison Bar plot for '{param}' ({plot_type}): No valid data found for plotting.")
            plt.close()
            continue

        sns.barplot(data=df_plot, x='Saltwork', y=param, hue='Saltwork', errorbar='sd', palette=SALTWORK_PALETTE, legend=False)
        plt.title(f'Mean of {param} by Saltwork ({plot_type})')
        plt.xlabel('Saltwork')
        plt.ylabel(f'Mean of {param}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'Mean_{param.replace("/", "").replace("°", "").replace("(","").replace(")","").replace(".","")}_by_Saltwork{plot_type.replace(" ", "_")}.png'))
        plt.close()
    print(f"  Finished Comparison Bar Plots for {plot_type}.")


def generate_parameter_by_stage_line_plots(df_filtered, plot_type):
    """Generates line plots of parameter variation by Stage (A-D) per Saltwork."""
    print(f"\nGenerating 'Parameter by Stage (A-D) by Saltwork' line plots for {plot_type}...")
    output_path = PARAM_BY_LOCAL_WATER_PLOTS_DIR 

    if plot_type == 'Water':
        df_plot = df_filtered[(df_filtered['Type'] == 'Water') & 
                              (df_filtered['Saltwork'].isin(['1', '2'])) &
                              (df_filtered['Stage'].isin(['A', 'B', 'C', 'D']))].copy()
        
        if df_plot.empty:
            print(f"  Skipping 'Parameter by Stage' plots for {plot_type}: No valid samples with Stage (A-D) found for Saltwork 1 and 2.")
            return

        stage_order = ['A', 'B', 'C', 'D']
        df_plot['Stage'] = pd.Categorical(df_plot['Stage'], categories=stage_order, ordered=True)


        for param in PARAMS_TO_ANALYZE:
            plt.figure(figsize=(12, 7))
            
            df_param_agg = df_plot.groupby(['Stage', 'Saltwork'])[param].agg(['mean', 'std']).reset_index()
            
            df_param_agg['Stage'] = pd.Categorical(df_param_agg['Stage'], categories=stage_order, ordered=True)
            df_param_agg = df_param_agg.sort_values(by='Stage')

            if df_param_agg['mean'].dropna().empty:
                print(f"  Skipping 'Parameter by Stage' line plot for '{param}': No valid data to aggregate.")
                plt.close()
                continue
            
            sns.lineplot(data=df_param_agg, x='Stage', y='mean', hue='Saltwork', 
                         palette=SALTWORK_PALETTE, marker='o', errorbar=None)

            for saltwork in df_param_agg['Saltwork'].unique():
                subset = df_param_agg[df_param_agg['Saltwork'] == saltwork]
                color = SALTWORK_PALETTE.get(saltwork, 'gray')
                plt.fill_between(x=subset['Stage'],
                                 y1=subset['mean'] - subset['std'],
                                 y2=subset['mean'] + subset['std'],
                                 color=color, alpha=0.2, label='nolegend')

            plt.title(f'Mean {param} by Stage for Saltwork ({plot_type})')
            plt.xlabel('Stage')
            plt.ylabel(f'Mean {param}')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'Mean_{param.replace("/", "").replace("°", "").replace("(","").replace(")","").replace(".","")}_by_Stage_Saltwork_Line{plot_type.replace(" ", "_").upper()}.png'))
            plt.close()
    print(f"  Finished 'Parameter by Stage (A-D) by Saltwork' line plots for {plot_type}.")


def generate_correlation_matrices(df_filtered):
    """Generates and saves correlation matrices by Type, Saltwork, and Stage."""
    print("\nGenerating Correlation Matrices by Type, Saltwork, and Stage...")
    
    unique_saltwork_type_combinations = df_filtered[['Type', 'Saltwork']].dropna().drop_duplicates()
    
    for index, row in unique_saltwork_type_combinations.iterrows():
        current_type = row['Type']
        current_saltwork = row['Saltwork']

        df_subset = df_filtered[(df_filtered['Type'] == current_type) & 
                                (df_filtered['Saltwork'] == current_saltwork)].copy()
        
        numeric_subset = df_subset[[col for col in PARAMS_TO_ANALYZE if col in df_subset.columns]].dropna(axis=1, how='all')
        
        if numeric_subset.empty or len(numeric_subset.columns) < 2:
            print(f"  Skipping correlation for {current_type}_{current_saltwork}: Insufficient data or numeric columns.")
            continue

        corr_matrix = numeric_subset.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'Correlation Matrix - {current_type} - Saltwork {current_saltwork}')
        plt.tight_layout()
        plt.savefig(os.path.join(CORRELATION_MATRICES_DIR, f'Correlation_Matrix_{current_type}_{current_saltwork}.png'))
        plt.close()
        print(f"  Generated correlation matrix for {current_type} - Saltwork {current_saltwork}")

    unique_water_stage_combinations = df_filtered[(df_filtered['Type'] == 'Water') & 
                                                    (df_filtered['Stage'].isin(['A', 'B', 'C', 'D']))][['Saltwork', 'Stage']].dropna().drop_duplicates()
    
    for index, row in unique_water_stage_combinations.iterrows():
        current_saltwork = row['Saltwork']
        current_stage = row['Stage']

        df_subset = df_filtered[(df_filtered['Type'] == 'Water') & 
                                (df_filtered['Saltwork'] == current_saltwork) &
                                (df_filtered['Stage'] == current_stage)].copy()
        
        numeric_subset = df_subset[[col for col in PARAMS_TO_ANALYZE if col in df_subset.columns]].dropna(axis=1, how='all')
        
        if numeric_subset.empty or len(numeric_subset.columns) < 2:
            print(f"  Skipping correlation for Water_Saltwork_{current_saltwork}Stage{current_stage}: Insufficient data or numeric columns.")
            continue

        corr_matrix = numeric_subset.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'Correlation Matrix - Water - Saltwork {current_saltwork} - Stage {current_stage}')
        plt.tight_layout()
        plt.savefig(os.path.join(CORRELATION_MATRICES_DIR, f'Correlation_Matrix_Water_Saltwork_{current_saltwork}Stage{current_stage}.png'))
        plt.close()
        print(f"  Generated correlation matrix for Water - Saltwork {current_saltwork} - Stage {current_stage}")

    print("Finished generating Correlation Matrices.")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting data loading and processing ---")

    df_water = load_and_process_file(EXCEL_FILE_WATER, 'Water')

    dfs_coleta = []
    if not df_water.empty:
        dfs_coleta.append(df_water)

    if not dfs_coleta:
        print("No data loaded. Exiting.")
    else:
        combined_df = pd.concat(dfs_coleta, ignore_index=True)
        print(f"\n--- End of file loading ---")
        print(f"Dimension of combined dataframe: {combined_df.shape}")

        saltworks_to_keep = ['1', '2',]
        filtered_df = combined_df[combined_df['Saltwork'].isin(saltworks_to_keep)].copy()
        
        filtered_out_rows = combined_df.shape[0] - filtered_df.shape[0]
        print(f"Filtered out {filtered_out_rows} rows. Keeping only saltworks {saltworks_to_keep}.")

        print("\n--- Filtered Data (Saltworks 1 and 2 only) ---")
        
        display_cols = ['Samples', 'Saltwork', 'Stage', 'Type']
        if 'Data' in filtered_df.columns:
            display_cols.append('Data')
        if 'Temperature (C°)' in filtered_df.columns:
            display_cols.append('Temperature (C°)')
        if 'Salinity (psu)' in filtered_df.columns:
            display_cols.append('Salinity (psu)')
        if 'TDS (ppm)' in filtered_df.columns:
            display_cols.append('TDS (ppm)')
        if 'El. Cond. (µS/cm)' in filtered_df.columns:
            display_cols.append('El. Cond. (µS/cm)')

        print(filtered_df[display_cols].head(20))
        print("-------------------------------------------------------")

        print("\n--- Generating plots ---")
        print(f"Using Saltwork color palette: {SALTWORK_PALETTE}")

        generate_time_series_plots(filtered_df, 'Water')
        generate_comparison_bar_plots(filtered_df, 'Water')

        generate_parameter_by_stage_line_plots(filtered_df, 'Water')

        generate_time_series_by_stage_plots(filtered_df, 'Water')

        generate_correlation_matrices(filtered_df)

        print("\n--- Analysis Completed ---")
        print(f"Plots saved in the '{OUTPUT_DIR}' directory.")