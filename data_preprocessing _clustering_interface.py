# streamlit_app.py
#PS C:\Users\dferf> python -m streamlit run d:\Bioinfo\TPs\tempCodeRunnerFile.py
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO # Added BytesIO for download
import base64 # For download links (alternative method)
import collections # For easier frequency counting in mode
import time 
# --- Sklearn / Standard Libraries (Keep for Clustering part) ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import make_blobs, make_moons, make_circles

# --- Plotting Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from mpl_toolkits.mplot3d import Axes3D

# --- Other necessary libraries (from user functions) ---
import re # For is_invalid_value function

# =============================================================
# START: Helper Functions (like CSV download)
# =============================================================

# Helper function to create a download link for DataFrames
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

# =============================================================
# START: User Provided Functions from TP1 (Modified where needed)
# =============================================================

# 1.Fonctions Manuelles (Keep as is)
def manual_len(lst):
    count = 0
    for _ in lst:
        count += 1
    return count

def manual_sum(lst):
    total = 0
    for num in lst:
        # Basic check for numeric before adding
        if isinstance(num, (int, float)) and num == num: # Check for NaN
           total += num
        # Optional: Try converting if string looks numeric? (Adds complexity)
        # elif isinstance(num, str):
        #    try: total += float(num)
        #    except ValueError: pass # Ignore non-convertible strings
    return total

def max_fun(lst):
    if not lst: return None
    # Filter for comparable values (e.g., numbers)
    comparable_lst = [v for v in lst if isinstance(v, (int, float)) and v==v]
    if not comparable_lst: return None
    max_value = comparable_lst[0]
    for value in comparable_lst[1:]:
        if value > max_value:
            max_value = value
    return max_value

def min_fun(lst):
    if not lst: return None
    comparable_lst = [v for v in lst if isinstance(v, (int, float)) and v==v]
    if not comparable_lst: return None
    min_value = comparable_lst[0]
    for value in comparable_lst[1:]:
        if value < min_value:
            min_value = value
    return min_value

def manual_std(data, column):
    if column not in data.columns:
        st.error(f"Column '{column}' not found in DataFrame for std dev calculation.")
        return None
    values = data[column]
    valid_values = []
    for val in values:
        if val is not None and val == val: # Check for NaN
            if is_numeric(val):
                valid_values.append(float(val)) # Use append for clarity
    if not valid_values or manual_len(valid_values) < 2:
        return None

    mean = manual_mean(data, column)
    if mean is None:
        return None

    variance = 0
    n = manual_len(valid_values)
    for val in valid_values:
        variance += (val - mean) ** 2
    variance = variance / (n - 1) # Sample standard deviation
    std_dev = variance ** 0.5
    return std_dev

def is_numeric(val):
    if isinstance(val, (int, float, np.number)) and val == val: # Check NaN
        return True
    # Allow strings that represent numbers? Be careful with this.
    # if isinstance(val, str):
    #    try:
    #        float(val)
    #        return True
    #    except ValueError:
    #        return False
    return False


def is_invalid_value(val):
    # Check for actual None
    if val is None:
        return True
    # Check for float NaN
    if isinstance(val, float) and val != val:
        return True
    # Check for empty or whitespace-only strings
    if isinstance(val, str):
        val = val.strip()
        if val == "":
            return True
        # Optional: Check for strings containing only special characters
        # if re.fullmatch(r'[^\w\s]+', val): # Check if it contains ONLY non-alphanumeric/non-whitespace
        #    return True
    # Add other potential invalid checks if needed (e.g., specific placeholders like "?")
    # if val == "?": return True
    return False


# 2.Exploration des Données
# CHANGE 1: Modified to show info for ONE selected attribute at a time
def informations_de_base_des_donnees_st(data):
    st.subheader("Informations Générales")
    nombre_colonnes = data.shape[1]
    st.write(f"Nombre d'attributs (colonnes) : {nombre_colonnes}")
    nombre_lignes = data.shape[0]
    st.write(f"Nombre d'instances (lignes) : {nombre_lignes}")

    st.subheader("Détails par Attribut")
    if data.empty:
        st.warning("Le jeu de données est vide.")
        return

    # Allow user to select which attribute to inspect
    attribute_list = data.columns.tolist()
    selected_column = st.selectbox("Choisir un attribut pour voir les détails:", attribute_list, key="base_info_select_col")

    if selected_column:
        st.markdown(f"**Détails pour : `{selected_column}`**")
        col_data = data[selected_column]
        st.write(f"- Type de Données : `{col_data.dtype}`")

        try:
            # Get all unique values, handle potential errors
            unique_vals = col_data.unique()
            num_unique = manual_len(unique_vals) # Use manual_len
            st.write(f"- Nombre de valeurs distinctes : {num_unique}")

            # Display ALL unique values in an expander for tidiness
            with st.expander(f"Afficher les {num_unique} valeurs distinctes"):
                # Convert to list for display, handle potential large lists
                unique_list = list(unique_vals)
                # To prevent huge outputs in streamlit, maybe limit display or use dataframe
                if num_unique > 1000: # Example limit
                     st.warning(f"Affichage limité aux 1000 premières valeurs uniques sur {num_unique}.")
                     st.dataframe(pd.Series(unique_list[:1000]), width=400) # Display as a series/dataframe
                elif num_unique > 50: # Use dataframe for medium lists too
                    st.dataframe(pd.Series(unique_list), width=400)
                else: # Show as simple list for short ones
                     st.write(unique_list)

        except Exception as e:
            st.error(f"Impossible d'analyser les valeurs uniques pour `{selected_column}`. Erreur: {e}")
            # Attempt to show some raw values anyway
            try:
                st.write("Quelques valeurs brutes :")
                st.dataframe(col_data.head(10))
            except: pass # Ignore if even head fails

# 3.Mesures de dispersion et de tendance centrale (Functions seem okay, ensure they handle non-numeric gracefully)
# Make sure the manual functions correctly filter non-numeric before calculation
def manual_min(data, column=None):
    if column is None:
        numeric_cols = data.select_dtypes(include=np.number).columns
        results = {}
        for col in numeric_cols:
            results[col] = manual_min(data, col)
        return results
    if column not in data.columns:
        return f"Erreur: Colonne '{column}' inexistante."
    values = data[column]
    valid_values = [float(v) for v in values if is_numeric(v)] # is_numeric checks NaN too
    if not valid_values: return None
    return min_fun(valid_values)

def manual_max(data, column=None):
    if column is None:
        numeric_cols = data.select_dtypes(include=np.number).columns
        results = {}
        for col in numeric_cols:
            results[col] = manual_max(data, col)
        return results
    if column not in data.columns:
        return f"Erreur: Colonne '{column}' inexistante."
    values = data[column]
    valid_values = [float(v) for v in values if is_numeric(v)]
    if not valid_values: return None
    return max_fun(valid_values)

def manual_median(data, column=None):
    if column is None:
        numeric_cols = data.select_dtypes(include=np.number).columns
        results = {}
        for col in numeric_cols:
            results[col] = manual_median(data, col)
        return results
    if column not in data.columns:
        return f"Erreur : La colonne '{column}' n'existe pas."
    values = data[column]
    valid_values = sorted([float(v) for v in values if is_numeric(v)])
    if not valid_values: return None
    n = manual_len(valid_values)
    if n == 0: return None
    if n % 2 == 1:
        return valid_values[n // 2]
    else:
        mid1 = valid_values[n // 2 - 1]
        mid2 = valid_values[n // 2]
        return (mid1 + mid2) / 2

def manual_q1(data, column=None):
     if column is None:
         numeric_cols = data.select_dtypes(include=np.number).columns
         return {col: manual_q1(data, col) for col in numeric_cols}
     if column not in data.columns:
         return f"Erreur: Colonne '{column}' inexistante."
     values = data[column]
     valid_values = sorted([float(v) for v in values if is_numeric(v)])
     if not valid_values: return None
     n = manual_len(valid_values)
     if n == 0: return None
     # Simple percentile calculation (method varies, this is one way)
     index = 0.25 * (n - 1) # 0-based index
     if index == int(index):
         return valid_values[int(index)]
     else:
         lower_idx = int(index)
         upper_idx = lower_idx + 1
         fraction = index - lower_idx
         if upper_idx < n: # Ensure upper index is within bounds
            return valid_values[lower_idx] + (valid_values[upper_idx] - valid_values[lower_idx]) * fraction
         else: # Handle edge case where upper index is out of bounds
            return valid_values[lower_idx] # Or handle as per specific definition


def manual_q3(data, column=None):
    if column is None:
        numeric_cols = data.select_dtypes(include=np.number).columns
        return {col: manual_q3(data, col) for col in numeric_cols}
    if column not in data.columns:
        return f"Erreur: Colonne '{column}' inexistante."
    values = data[column]
    valid_values = sorted([float(v) for v in values if is_numeric(v)])
    if not valid_values: return None
    n = manual_len(valid_values)
    if n == 0: return None
    # Simple percentile calculation
    index = 0.75 * (n - 1) # 0-based index
    if index == int(index):
        return valid_values[int(index)]
    else:
        lower_idx = int(index)
        upper_idx = lower_idx + 1
        fraction = index - lower_idx
        if upper_idx < n:
            return valid_values[lower_idx] + (valid_values[upper_idx] - valid_values[lower_idx]) * fraction
        else:
            return valid_values[lower_idx]

def manual_mean(data, column=None):
    if column is None:
        numeric_cols = data.select_dtypes(include=np.number).columns
        results = {}
        for col in numeric_cols:
            results[col] = manual_mean(data, col)
        return results
    if column not in data.columns:
        return f"Erreur: Colonne '{column}' inexistante."
    values = data[column]
    total_sum = 0
    count = 0
    for val in values:
        if is_numeric(val): # is_numeric checks NaN
            num_val = float(val) # Convert to float
            total_sum += num_val
            count += 1
    if count == 0: return None
    mean = total_sum / count
    return mean


# 4.Statistiques Descriptives
# CHANGE 3: Modified manual_mode to return frequency, count, and handle uniform case
def manual_mode(data, column=None):
    if column is None:
        results = {}
        for col in data.columns:
            results[col] = manual_mode(data, col) # Recursive call
        return results # Returns dict of results per column

    if column not in data.columns:
        return {"error": f"Erreur: Colonne '{column}' inexistante."}

    values = data[column]
    # Filter out only actual None and NaN, keep other types (strings, numbers etc.)
    # Use is_invalid_value OR just check for None/NaN depending on desired behavior
    # Let's filter strictly None/NaN for mode calculation
    valid_values = [v for v in values if not (v is None or (isinstance(v, float) and pd.isna(v)))] # Using pd.isna for robustness

    if not valid_values:
        return {"modes": [], "frequency": 0, "count": 0, "message": "Pas de valeurs valides pour calculer le mode."}

    # Use collections.Counter for efficient frequency calculation
    # This is much more reliable than manual counting, especially with mixed/unhashable types
    try:
        # Convert unhashable types (like lists/dicts if they exist in column) to strings for counting
        hashable_values = [str(v) if not isinstance(v, (int, float, str, bool, tuple)) else v for v in valid_values]
        freq_counter = collections.Counter(hashable_values)
        if not freq_counter: # Should not happen if valid_values is not empty
             return {"modes": [], "frequency": 0, "count": 0, "message": "Erreur de comptage de fréquence."}

        # Find the maximum frequency
        max_freq = max(freq_counter.values())

        # Check if all values have the same frequency
        all_same_freq = all(f == max_freq for f in freq_counter.values())

        # Find modes (keys with max frequency)
        modes_raw = [val for val, freq in freq_counter.items() if freq == max_freq]

        # Attempt to map back to original types if possible (best effort)
        original_modes = []
        original_value_map = {str(v) if not isinstance(v, (int, float, str, bool, tuple)) else v : v for v in set(valid_values)}
        for mode_raw in modes_raw:
            original_modes.append(original_value_map.get(mode_raw, mode_raw)) # Fallback to raw if not found

        num_modes = manual_len(original_modes)

        # CHANGE 3: Handle uniform distribution message
        if all_same_freq and manual_len(freq_counter) > 1: # More than one unique value, all same freq
            message = f"Distribution uniforme : Toutes les {manual_len(freq_counter)} valeurs apparaissent {max_freq} fois."
            # Return all values as modes in this case, or indicate no specific mode? Let's return all.
            return {"modes": original_modes, "frequency": max_freq, "count": num_modes, "message": message}
        elif manual_len(freq_counter) == 1: # Only one unique value
             message = f"Une seule valeur unique ('{original_modes[0]}') présente {max_freq} fois."
             return {"modes": original_modes, "frequency": max_freq, "count": num_modes, "message": message}
        else:
            # Normal case: return modes, frequency, and count
            return {"modes": original_modes, "frequency": max_freq, "count": num_modes, "message": None}

    except TypeError as e:
        # Handle cases where Counter fails (e.g., extremely complex unhashable types)
        return {"modes": [], "frequency": 0, "count": 0, "message": f"Erreur calcul mode (type non hashable?): {e}"}
    except Exception as e:
        return {"modes": [], "frequency": 0, "count": 0, "message": f"Erreur inattendue calcul mode: {e}"}


# 5.Visualisation des Données (Keep as is)
def box_plot_st(data):
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("Aucune colonne numérique valide pour le boxplot.")
        return None
    # Drop NaNs for plotting
    numeric_data_dict = {col: data[col].dropna().tolist() for col in numeric_cols}
    # Filter out empty lists after dropna
    numeric_data_dict = {k: v for k, v in numeric_data_dict.items() if v}
    if not numeric_data_dict:
        st.warning("Aucune donnée numérique valide restante après suppression des NaN pour le boxplot.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        ax.boxplot(numeric_data_dict.values(), labels=numeric_data_dict.keys(), patch_artist=True)
        ax.tick_params(axis='x', rotation=45)
        ax.set_title("Distribution des Attributs Numériques (Boxplot)")
        ax.set_xlabel("Attributs")
        ax.set_ylabel("Valeurs")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        return fig
    except Exception as e:
        st.error(f"Erreur lors de la génération du boxplot: {e}")
        return None

def scatter_plot_all_st(data, col1=None, col2=None):
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("Au moins deux colonnes numériques sont nécessaires pour un scatter plot.")
        return None

    if col1 and col2:
        if col1 not in numeric_columns or col2 not in numeric_columns:
            st.error(f"'{col1}' ou '{col2}' n'est pas une colonne numérique valide.")
            return None
        plot_data = data[[col1, col2]].dropna()
        if plot_data.empty:
            st.warning(f"Aucune paire de valeurs valides trouvée entre '{col1}' et '{col2}'.")
            return None
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(plot_data[col1], plot_data[col2], alpha=0.6)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"Scatter Plot : {col1} vs {col2}")
        ax.grid(True, linestyle='--', alpha=0.6)
        return fig
    else:
        st.info("Génération des scatter plots pour toutes les paires numériques (Pair Plot)...")
        try:
            # Filter only numeric columns for pairplot
            numeric_data_for_plot = data[numeric_columns]
            # Drop rows with ANY NaN in the numeric columns to avoid pairplot errors
            numeric_data_for_plot = numeric_data_for_plot.dropna()
            if numeric_data_for_plot.empty or numeric_data_for_plot.shape[0] < 2 :
                 st.warning("Pas assez de données valides (sans NaN) pour générer le pair plot.")
                 return None
            if numeric_data_for_plot.shape[1] < 2:
                 st.warning("Moins de deux colonnes numériques valides (sans NaN) pour générer le pair plot.")
                 return None

            pair_plot_fig = sns.pairplot(numeric_data_for_plot)
            pair_plot_fig.fig.suptitle("Scatter Plots pour les Paires Numériques", y=1.02)
            return pair_plot_fig.fig
        except Exception as e:
            st.error(f"Erreur durant la génération du Pair Plot: {e}")
            return None


# 6. Gestion des Valeurs Manquantes / Invalides
def find_missing_and_invalid_values_st(data):
    st.subheader("Identification des Valeurs Problématiques")
    if data.empty:
        st.warning("Le jeu de données est vide.")
        return {'missing': {}, 'invalid': {}}

    # Let user choose which columns to analyze
    all_cols = data.columns.tolist()
    cols_to_check = st.multiselect(
        "Sélectionnez les colonnes à analyser:",
        options=all_cols,
        default=all_cols,
        key="missing_invalid_cols_select"
    )

    # Separate missing and invalid values
    missing_stats = {}
    invalid_stats = {}

    for col in cols_to_check:
        col_series = data[col]
        
        # 1. Missing values (NaN/None)
        na_mask = col_series.isna()
        missing_indices = col_series[na_mask].index.tolist()
        missing_count = len(missing_indices)
        
        # 2. Invalid values (using is_invalid_value function)
        invalid_mask = col_series[~na_mask].apply(is_invalid_value)
        invalid_indices = col_series[~na_mask][invalid_mask].index.tolist()
        invalid_count = len(invalid_indices)
        
        # Store stats
        if missing_count > 0:
            missing_stats[col] = {
                'count': missing_count,
                'indices': missing_indices,
                'values': [np.nan]*missing_count
            }
        
        if invalid_count > 0:
            invalid_stats[col] = {
                'count': invalid_count,
                'indices': invalid_indices,
                'values': col_series[invalid_indices].tolist()
            }

    # Display results in tabs
    tab1, tab2 = st.tabs(["Valeurs Manquantes", "Valeurs Invalides"])

    with tab1:
        if missing_stats:
            st.markdown("### Valeurs Manquantes par Colonne")
            for col, stats in missing_stats.items():
                with st.expander(f"**{col}** - {stats['count']} valeurs manquantes"):
                    st.write(f"Positions (premières 10): {stats['indices']}")
                    st.write(f"Valeurs: {stats['values']}")
        else:
            st.success("Aucune valeur manquante détectée dans les colonnes sélectionnées.")

    with tab2:
        if invalid_stats:
            st.markdown("### Valeurs Invalides par Colonne")
            for col, stats in invalid_stats.items():
                with st.expander(f"**{col}** - {stats['count']} valeurs invalides"):
                    st.write(f"Positions (premières 10): {stats['indices']}")
                    st.write(f"Valeurs (premières 10): {stats['values']}")
        else:
            st.success("Aucune valeur invalide détectée dans les colonnes sélectionnées.")

    return {'missing': missing_stats, 'invalid': invalid_stats}
# CHANGE 2: NEW Function for Interactive Non-Numeric Handling
def handle_non_numeric_interactive(df):
    st.subheader("2.2a Gestion Interactive des Colonnes Non Numériques")
    data_copy = df.copy() # Work on a copy
    
    # Use session state to store decisions across reruns within this step
    if 'non_numeric_decisions' not in st.session_state:
        st.session_state.non_numeric_decisions = {}

    # Identify non-numeric columns (object, string, category, boolean potentially)
    non_numeric_cols = data_copy.select_dtypes(exclude=[np.number]).columns.tolist()

    if not non_numeric_cols:
        st.info("Aucune colonne non numérique détectée nécessitant une action manuelle.")
        st.session_state.non_numeric_decisions = {} # Clear decisions if no cols
        return data_copy # Return original copy

    st.warning("Les colonnes suivantes ne sont pas purement numériques. Choisissez une action pour chacune.")

    cols_to_delete = []
    cols_to_encode = {} # Store encoding maps like {'col_name': {'value1': replacement1, ...}}

    for col in non_numeric_cols:
        # Retrieve previous decision for this column if exists
        default_action_index = 0 # Default to 'Keep (No Change)'
        previous_decision = st.session_state.non_numeric_decisions.get(col, {})
        action_options = ["Conserver (Pas de Changement)", "Supprimer la Colonne", "Encoder/Remplacer Valeurs"]
        if previous_decision.get('action') == 'delete':
            default_action_index = 1
        elif previous_decision.get('action') == 'encode':
            default_action_index = 2

        with st.expander(f"Attribut : `{col}` (Type: {data_copy[col].dtype})", expanded=(default_action_index != 0)):
            unique_vals = data_copy[col].unique()
            st.write(f"Valeurs distinctes ({len(unique_vals)}):")
            # Display unique values (use dataframe for many)
            if len(unique_vals) > 50:
                 st.dataframe(pd.Series(unique_vals), height=150)
            else:
                 st.write(list(unique_vals))

            action = st.radio(
                "Action:",
                action_options,
                key=f"action_{col}",
                index=default_action_index,
                horizontal=True
            )

            # Store the chosen action immediately
            st.session_state.non_numeric_decisions[col] = {'action': action.split(' ')[0].lower()} # store 'conserver', 'supprimer', 'encoder/remplacer'

            if action == "Supprimer la Colonne":
                cols_to_delete.append(col)
                st.session_state.non_numeric_decisions[col]['action'] = 'delete' # update state

            elif action == "Encoder/Remplacer Valeurs":
                st.markdown("**Définir les remplacements :**")
                encoding_map = previous_decision.get('map', {}) # Load previous map if exists
                
                # Use columns for better layout
                col1, col2 = st.columns([2,1])
                with col1:
                    st.write("Valeur Originale")
                with col2:
                    st.write("Remplacer par")

                new_encoding_map = {}
                # Sort unique values for consistent order, handle NaN/None representation
                display_vals = sorted([v for v in unique_vals if v is not None and v==v] + ["<NaN/None>"])

                for i, val in enumerate(display_vals):
                    original_val_key = val if val != "<NaN/None>" else None # Key for dictionary

                    # Get previous value or default to original
                    default_replace_val = encoding_map.get(original_val_key, "")

                    col1_form, col2_form = st.columns([2,1])
                    with col1_form:
                        st.text(f"`{str(val)}`") # Display original value clearly
                    with col2_form:
                        # Use text_input allowing any replacement (numbers or strings)
                        # If you STRICTLY want numeric replacements, use st.number_input
                        replace_val_str = st.text_input(
                            f"Remplacement pour `{str(val)}`",
                            value=str(default_replace_val), # Work with strings in input
                            key=f"replace_{col}_{i}",
                            label_visibility="collapsed"
                        )
                        # Try to convert back to numeric if possible, otherwise keep as string
                        try:
                            replace_val = float(replace_val_str)
                            # Handle integer conversion if no decimal part
                            if replace_val.is_integer():
                                replace_val = int(replace_val)
                        except ValueError:
                             # Allow empty string to mean "no replacement / keep original type if possible"
                             # or map to NaN? Let's map empty string input to NaN.
                             if replace_val_str.strip() == "":
                                 replace_val = np.nan # Or None, depending on desired outcome
                             else:
                                 replace_val = replace_val_str # Keep as string if not numeric


                    # Store the mapping only if a replacement value is provided
                    # Use the actual value (handling None) as the key
                    current_original_val = None if val == "<NaN/None>" else val
                    if replace_val_str.strip() != "": # Only add to map if user entered something
                        new_encoding_map[current_original_val] = replace_val

                cols_to_encode[col] = new_encoding_map
                st.session_state.non_numeric_decisions[col]['action'] = 'encode' # update state
                st.session_state.non_numeric_decisions[col]['map'] = new_encoding_map # store map


            elif action == "Conserver (Pas de Changement)":
                 st.session_state.non_numeric_decisions[col]['action'] = 'conserver' # update state
                 # Remove map if action changed back to conserver
                 if 'map' in st.session_state.non_numeric_decisions[col]:
                     del st.session_state.non_numeric_decisions[col]['map']

    # Button to apply all chosen actions
    if st.button("Appliquer les Actions sur Colonnes Non Numériques", key="apply_non_numeric"):
        final_df = df.copy() # Start fresh from original df passed to function

        # Retrieve decisions from session state
        decisions = st.session_state.non_numeric_decisions
        final_cols_to_delete = []
        final_cols_to_encode = {}

        for col, decision_data in decisions.items():
             action = decision_data.get('action')
             if action == 'delete':
                 final_cols_to_delete.append(col)
             elif action == 'encode':
                 encoding_map = decision_data.get('map')
                 if encoding_map is not None: # Ensure map exists
                    final_cols_to_encode[col] = encoding_map

        # 1. Apply Deletions
        if final_cols_to_delete:
            st.write(f"Suppression des colonnes: {final_cols_to_delete}")
            final_df.drop(columns=final_cols_to_delete, inplace=True)

        # 2. Apply Encoding/Replacement
        if final_cols_to_encode:
            st.write(f"Encodage/Remplacement pour les colonnes: {list(final_cols_to_encode.keys())}")
            for col, mapping in final_cols_to_encode.items():
                if col in final_df.columns: # Check if column still exists (wasn't deleted)
                    st.write(f" - Remplacement pour `{col}` avec map: {mapping}")
                    # Use replace for robust mapping (handles values not explicitly in map if needed - defaults to no change)
                    # Or use map for strict mapping (values not in map become NaN) - Let's use map for explicit control
                    final_df[col] = final_df[col].map(mapping)
                    # Optional: Convert column type if all replacements were numeric
                    try:
                        final_df[col] = pd.to_numeric(final_df[col])
                        st.write(f"   - Colonne `{col}` convertie en type numérique.")
                    except ValueError:
                        st.write(f"   - Colonne `{col}` contient des valeurs non-numériques après remplacement.")


        st.success("Actions appliquées sur les colonnes non numériques.")
        st.dataframe(final_df.head())
        # Clear decisions after applying
        # st.session_state.non_numeric_decisions = {} # Optional: clear state after apply? Or keep for review? Let's keep.
        return final_df # Return the modified dataframe

    else:
        # If button not pressed, return the *original* df copy passed to the function
        # Or potentially return a preview based on current selections without applying?
        # For simplicity, let's only return modified df when button is pressed.
        st.info("Appuyez sur le bouton 'Appliquer les Actions...' pour modifier le jeu de données.")
        return df.copy() # Return the original copy if not applied


# Modified replacement function to handle potential type errors during replacement
def replace_missing_or_invalid(df, missing_strategy=None, invalid_strategy=None, 
                             missing_cols=None, invalid_cols=None):
    """Process DataFrame to handle missing/invalid values separately"""
    data_copy = df.copy()
    
    # Handle missing values
    if missing_strategy and missing_strategy != "Aucune" and missing_cols:
        st.info(f"Traitement des valeurs manquantes avec stratégie: '{missing_strategy}'")
        for col in missing_cols:
            if col not in data_copy.columns:
                continue
                
            na_mask = data_copy[col].isna()
            if not na_mask.any():
                continue
                
            replacement_value = None
            if missing_strategy == "mean":
                replacement_value = manual_mean(data_copy, col)
            elif missing_strategy == "median":
                replacement_value = manual_median(data_copy, col)
            elif missing_strategy == "mode":
                mode_result = manual_mode(data_copy, col)
                if mode_result and mode_result.get("modes"):
                    replacement_value = mode_result["modes"][0]
            
            if replacement_value is not None:
                data_copy.loc[na_mask, col] = replacement_value
                st.write(f"- Colonne '{col}': {na_mask.sum()} valeurs manquantes remplacées par {replacement_value}")

    # Handle invalid values
    if invalid_strategy and invalid_strategy != "Aucune" and invalid_cols:
        st.info(f"Traitement des valeurs invalides avec stratégie: '{invalid_strategy}'")
        for col in invalid_cols:
            if col not in data_copy.columns:
                continue
                
            invalid_mask = data_copy[col].apply(is_invalid_value) & ~data_copy[col].isna()
            if not invalid_mask.any():
                continue
                
            replacement_value = None
            if invalid_strategy == "mean":
                replacement_value = manual_mean(data_copy, col)
            elif invalid_strategy == "median":
                replacement_value = manual_median(data_copy, col)
            elif invalid_strategy == "mode":
                mode_result = manual_mode(data_copy, col)
                if mode_result and mode_result.get("modes"):
                    replacement_value = mode_result["modes"][0]
            
            if replacement_value is not None:
                data_copy.loc[invalid_mask, col] = replacement_value
                st.write(f"- Colonne '{col}': {invalid_mask.sum()} valeurs invalides remplacées par {replacement_value}")

    return data_copy

# Placeholder Normalization functions (Replace with your TP1 functions)
# These should ideally work on a copy and return the modified df
def min_max_normalization_manual(df, columns):
    df_copy = df.copy()
    st.markdown("--- \n **Normalisation Min-Max (Manuelle - Placeholder)**")
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            min_val = manual_min(df_copy, col)
            max_val = manual_max(df_copy, col)
            if min_val is not None and max_val is not None and max_val > min_val:
                # Apply normalization: (value - min) / (max - min)
                # Handle potential division by zero if min == max
                range_val = max_val - min_val
                if range_val == 0:
                     st.warning(f"Colonne '{col}': Min et Max sont égaux ({min_val}). Normalisation Min-Max non appliquée (division par zéro).")
                     df_copy[col] = 0 # Or 0.5? Or keep original? Setting to 0.
                else:
                    # Iterate manually (less efficient but follows manual principle)
                    normalized_col = []
                    for val in df_copy[col]:
                        if is_numeric(val):
                            normalized = (float(val) - min_val) / range_val
                            normalized_col.append(normalized)
                        else:
                            normalized_col.append(val) # Keep non-numeric as is or NaN? Let's use NaN
                            # normalized_col.append(np.nan)
                    df_copy[col] = normalized_col
                    st.write(f"- Colonne '{col}' normalisée (Min-Max).")

            elif min_val is not None and max_val is not None and min_val == max_val:
                 st.warning(f"Colonne '{col}': Toutes les valeurs sont identiques ({min_val}). Normalisation Min-Max donne 0.")
                 df_copy[col] = 0.0 # Assign 0 or 0.5? Setting to 0.
            else:
                st.warning(f"Impossible de calculer Min/Max pour la colonne '{col}'. Normalisation sautée.")
        elif col in df_copy.columns:
             st.warning(f"Colonne '{col}' n'est pas numérique. Normalisation Min-Max sautée.")

    return df_copy

def z_score_normalization_manual(df, columns):
    df_copy = df.copy()
    st.markdown("--- \n **Normalisation Z-Score (Manuelle - Placeholder)**")
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            mean_val = manual_mean(df_copy, col)
            std_dev_val = manual_std(df_copy, col) # Uses manual_mean inside
            if mean_val is not None and std_dev_val is not None:
                # Apply normalization: (value - mean) / std_dev
                # Handle potential division by zero if std_dev is 0
                if std_dev_val == 0:
                    st.warning(f"Colonne '{col}': Écart-type est zéro. Normalisation Z-Score non appliquée (division par zéro). Valeurs mises à 0.")
                    df_copy[col] = 0.0 # Set standardized score to 0 if std dev is 0
                else:
                     # Iterate manually
                    normalized_col = []
                    for val in df_copy[col]:
                        if is_numeric(val):
                            normalized = (float(val) - mean_val) / std_dev_val
                            normalized_col.append(normalized)
                        else:
                            normalized_col.append(np.nan) # Keep non-numeric as NaN
                    df_copy[col] = normalized_col
                    st.write(f"- Colonne '{col}' normalisée (Z-Score).")
            else:
                st.warning(f"Impossible de calculer Moyenne/Écart-type pour '{col}'. Normalisation Z-Score sautée.")
        elif col in df_copy.columns:
             st.warning(f"Colonne '{col}' n'est pas numérique. Normalisation Z-Score sautée.")
    return df_copy


# =============================================================
# END: User Provided Functions
# =============================================================


# --- Configuration and Page Setup ---
st.set_page_config(layout="wide", page_title="Exploration & Clustering")
st.title("Exploration & Clustering ")

# --- Session State Initialization ---
# Store data at different stages
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'data_after_non_numeric_handling' not in st.session_state:
    st.session_state.data_after_non_numeric_handling = None
if 'data_after_missing_handling' not in st.session_state:
     st.session_state.data_after_missing_handling = None # Renamed from processed_data_tp1
if 'data_after_normalization' not in st.session_state:
    st.session_state.data_after_normalization = None
if 'data_for_clustering' not in st.session_state:
    st.session_state.data_for_clustering = None # Final data prepped for Sklearn
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
# Add state for non-numeric handling choices
if 'non_numeric_decisions' not in st.session_state:
    st.session_state.non_numeric_decisions = {}

# --- Sidebar for Controls ---
st.sidebar.header("Configuration")

# ================================================
# SECTION 1: DATA LOADING
# ================================================
st.sidebar.subheader("1. Chargement des Données")
# Clear previous data if source changes
st.sidebar.radio("Source des données:",
                 ('Uploader CSV', 'Données Exemple'),
                 key="data_source",
                 on_change=lambda: [s.clear() for s in [ # Reset states on source change
                    'raw_data', 'data_after_non_numeric_handling',
                    'data_after_missing_handling', 'data_after_normalization',
                    'data_for_clustering', 'non_numeric_decisions', 'results_history']])

uploaded_file = None # Define outside the if block

if st.session_state.data_source == 'Uploader CSV':
    uploaded_file = st.sidebar.file_uploader("Sélectionnez CSV", type=["csv"], key="uploader")
    if uploaded_file and st.session_state.raw_data is None: # Load only once or if cleared
        try:
            df_loaded = pd.read_csv(uploaded_file)
            st.session_state.raw_data = df_loaded
            # Reset downstream data states when new file is loaded
            st.session_state.data_after_non_numeric_handling = None
            st.session_state.data_after_missing_handling = None
            st.session_state.data_after_normalization = None
            st.session_state.data_for_clustering = None
            st.session_state.non_numeric_decisions = {}
            st.rerun() # Rerun to reflect loaded data
        except Exception as e:
            st.sidebar.error(f"Erreur lecture CSV: {e}")
            st.session_state.raw_data = None # Ensure it's None on error
            st.stop()

elif st.session_state.data_source == 'Données Exemple':
     if st.session_state.raw_data is None: # Generate only once or if cleared
        dataset_name = st.sidebar.selectbox("Jeu de données:", ('Blobs', 'Moons', 'Circles'), key="sample_dataset")
        n_samples = st.sidebar.slider("Nb Échantillons", 100, 1000, 200, key="sample_n")
        noise = st.sidebar.slider("Bruit", 0.0, 0.2, 0.05, key="sample_noise")
        if dataset_name == 'Blobs': X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=42)
        elif dataset_name == 'Moons': X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        else: X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        df_generated = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
        # Add a categorical column for testing non-numeric handling
        if dataset_name == 'Blobs': # Add based on blob label 'y'
            df_generated['Category'] = pd.Series(y).map({0:'TypeA', 1:'TypeB', 2:'TypeC', 3:'TypeD'}).fillna('Other')
            # Introduce some missing/invalid values for testing processing steps
            df_generated.iloc[5:10, 0] = np.nan # Add some NaNs
            if 'Category' in df_generated.columns:
                 df_generated.iloc[15:20, df_generated.columns.get_loc('Category')] = "" # Add empty strings
                 df_generated.iloc[25:30, df_generated.columns.get_loc('Category')] = None # Add Nones

        st.session_state.raw_data = df_generated
        # Reset downstream states
        st.session_state.data_after_non_numeric_handling = None
        st.session_state.data_after_missing_handling = None
        st.session_state.data_after_normalization = None
        st.session_state.data_for_clustering = None
        st.session_state.non_numeric_decisions = {}
        st.rerun() # Rerun to reflect generated data

# Display Raw Data if loaded/generated
st.header("1. Données Chargées (Brutes)")
if st.session_state.raw_data is not None:
    st.dataframe(st.session_state.raw_data.head())
    st.write(f"Dimensions Brutes: {st.session_state.raw_data.shape}")
else:
    st.info("Chargez un fichier CSV ou sélectionnez des données d'exemple depuis la barre latérale.")
    st.stop() # Stop execution if no data


# ================================================
# SECTION 2: PREPROCESSING (TP1 Integration)
# ================================================
st.header("2. Exploration & Prétraitement (Basé sur TP1)")

# Determine the starting point for this section's operations
# Should cascade: start from raw, then non-numeric handled, then missing handled etc.
current_data_stage = st.session_state.raw_data # Start with raw

# --- 2.1 Informations Générales ---
st.markdown("---")
st.subheader("2.1 Informations Générales & Statistiques (TP1)")
info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("**Informations de Base**")
    # Pass the current data stage to the info function
    informations_de_base_des_donnees_st(current_data_stage)

with info_col2:
    st.markdown("**Statistiques Descriptives (Manuelles - TP1)**")
    numeric_cols_tp1 = current_data_stage.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols_tp1:
        selected_col_stat = st.selectbox("Choisir colonne numérique pour stats:", numeric_cols_tp1, key="stat_col_select")
        if selected_col_stat:
            # Calculate stats using manual functions
            stats_data = {}
            stats_data["Moyenne"] = manual_mean(current_data_stage, selected_col_stat)
            stats_data["Médiane"] = manual_median(current_data_stage, selected_col_stat)
            stats_data["Min"] = manual_min(current_data_stage, selected_col_stat)
            stats_data["Max"] = manual_max(current_data_stage, selected_col_stat)
            stats_data["Q1"] = manual_q1(current_data_stage, selected_col_stat)
            stats_data["Q3"] = manual_q3(current_data_stage, selected_col_stat)
            stats_data["Écart-type"] = manual_std(current_data_stage, selected_col_stat)

            # Handle Mode Separately (returns dict)
            mode_result = manual_mode(current_data_stage, selected_col_stat)
            mode_val = "N/A"
            mode_freq = "N/A"
            mode_count = "N/A"
            mode_msg = None

            if isinstance(mode_result, dict):
                if "error" in mode_result:
                    mode_val = mode_result["error"]
                elif "message" in mode_result and mode_result["message"]:
                    # Display special message (e.g., uniform distribution)
                    mode_msg = mode_result["message"]
                    mode_val = f"({mode_msg})" # Indicate message in value field
                    if "modes" in mode_result: mode_val = f"{mode_result['modes']}" # Still show modes if available
                    if "frequency" in mode_result: mode_freq = mode_result['frequency']
                    if "count" in mode_result: mode_count = mode_result['count']
                elif "modes" in mode_result:
                     mode_val = mode_result["modes"]
                     mode_freq = mode_result.get("frequency", "N/A")
                     mode_count = mode_result.get("count", "N/A")

            # Format results for display
            stats_display = {}
            for k, v in stats_data.items():
                stats_display[k] = (f"{v:.3f}" if isinstance(v, (int, float)) else str(v)) if v is not None else "N/A"

            # Add Mode info
            stats_display["Mode(s)"] = str(mode_val) # Ensure mode list is stringified
            # CHANGE 3: Add option to show mode frequency and count
            show_mode_details = st.checkbox("Afficher détails du mode (fréq, nb)", key="show_mode_freq")
            if show_mode_details:
                 stats_display["Mode Fréquence"] = str(mode_freq)
                 stats_display["Nombre de Modes"] = str(mode_count)
                 if mode_msg: st.info(f"Note sur le Mode: {mode_msg}") # Display message if exists

            st.json(stats_display) # Display as JSON
    else:
        st.info("Aucune colonne numérique détectée pour les statistiques dans les données actuelles.")


# --- 2.2 Interactive Non-Numeric Handling ---
st.markdown("---")
# Only run this step if the previous step produced data
if current_data_stage is not None:
    # Perform interactive handling
    # The function returns the modified df *only when 'Apply' is clicked*
    data_after_nn_handling_temp = handle_non_numeric_interactive(current_data_stage)

    # Update session state ONLY if the data returned is different (meaning apply was clicked and changes were made)
    # Comparing DataFrames directly can be tricky, rely on the button press logic within the function.
    # A better way: the function could return a tuple (df, applied_changes_flag)
    # For now, assume if the function modifies, it returns a *different* object (due to copy)
    # Or better: Check if the button "apply_non_numeric" exists in st.query_params or was clicked (requires more complex state handling)
    # Let's update the state if the function returns something seemingly processed
    if data_after_nn_handling_temp is not None: # Check if function returned a df
         st.session_state.data_after_non_numeric_handling = data_after_nn_handling_temp
         current_data_stage = st.session_state.data_after_non_numeric_handling # Update current stage

         # CHANGE 2: Add Download button after this step
         st.download_button(
            label="📥 Télécharger Données après Gestion Non-Numérique (.csv)",
            data=convert_df_to_csv(current_data_stage),
            file_name='data_after_non_numeric.csv',
            mime='text/csv',
            key='download_non_numeric'
         )

# --- 2.3 Missing/Invalid Value Handling ---
st.markdown("---")
st.subheader("2.3 Gestion des Valeurs Manquantes/Invalides (TP1)")

if current_data_stage is not None:
    # Initialize variables
    missing_strategy = 'Aucune'
    invalid_strategy = 'Aucune'
    selected_missing_cols = []
    selected_invalid_cols = []
    
    # Run detection
    if st.checkbox("Identifier valeurs problématiques", key="find_missing_check"):
        problem_values = find_missing_and_invalid_values_st(current_data_stage)
        
        # Get columns with missing and invalid values
        missing_cols = list(problem_values['missing'].keys())
        invalid_cols = list(problem_values['invalid'].keys())

        # Missing values treatment
        if missing_cols:
            st.markdown("### Traitement des Valeurs Manquantes")
            missing_strategy = st.selectbox(
                "Stratégie pour valeurs manquantes:",
                ['Aucune', 'Supprimer Lignes', 'Remplacer par Moyenne', 
                 'Remplacer par Médiane', 'Remplacer par Mode'],
                key="missing_strategy_select"
            )
            
            # Let user select which missing columns to treat
            selected_missing_cols = st.multiselect(
                "Colonnes avec valeurs manquantes à traiter:",
                options=missing_cols,
                default=missing_cols,
                key="missing_cols_select"
            )

        # Invalid values treatment
        if invalid_cols:
            st.markdown("### Traitement des Valeurs Invalides")
            invalid_strategy = st.selectbox(
                "Stratégie pour valeurs invalides:",
                ['Aucune', 'Supprimer Lignes', 'Remplacer par Moyenne', 
                 'Remplacer par Médiane', 'Remplacer par Mode'],
                key="invalid_strategy_select"
            )
            
            # Let user select which invalid columns to treat
            selected_invalid_cols = st.multiselect(
                "Colonnes avec valeurs invalides à traiter:",
                options=invalid_cols,
                default=invalid_cols,
                key="invalid_cols_select"
            )

        if st.button("Appliquer Traitement", key="apply_missing_treat"):
            # Convert strategy names to keywords
            missing_keyword = None
            if missing_strategy != 'Aucune' and selected_missing_cols:
                missing_keyword = {
                    'Supprimer Lignes': 'remove',
                    'Remplacer par Moyenne': 'mean',
                    'Remplacer par Médiane': 'median',
                    'Remplacer par Mode': 'mode'
                }[missing_strategy]
            
            invalid_keyword = None
            if invalid_strategy != 'Aucune' and selected_invalid_cols:
                invalid_keyword = {
                    'Supprimer Lignes': 'remove',
                    'Remplacer par Moyenne': 'mean',
                    'Remplacer par Médiane': 'median',
                    'Remplacer par Mode': 'mode'
                }[invalid_strategy]
            
            # Apply treatment
            data_after_missing_handling_temp = replace_missing_or_invalid(
                current_data_stage,
                missing_strategy=missing_keyword,
                invalid_strategy=invalid_keyword,
                missing_cols=selected_missing_cols,
                invalid_cols=selected_invalid_cols
            )
            
            st.session_state.data_after_missing_handling = data_after_missing_handling_temp
            st.success("Traitement appliqué.")
            st.dataframe(st.session_state.data_after_missing_handling.head())
            current_data_stage = st.session_state.data_after_missing_handling
            st.rerun()

   # Display data after treatment
if st.session_state.data_after_missing_handling is not None:
    st.markdown("**Aperçu après traitement:**")
    st.dataframe(st.session_state.data_after_missing_handling.head())
    current_data_stage = st.session_state.data_after_missing_handling
    
    # Add download button
    st.download_button(
        label="📥 Télécharger Données après Gestion Manquantes/Invalides (.csv)",
        data=convert_df_to_csv(current_data_stage),
        file_name='data_after_missing_invalid_handling.csv',
        mime='text/csv',
        key='download_missing_invalid'
    )
# --- 2.4 Normalisation ---
st.markdown("---")
st.subheader("2.4 Normalisation (Manuelle - Optionnelle)")
# Use data potentially modified by the previous step
if current_data_stage is not None:
    data_to_normalize = current_data_stage

    # Check for numeric columns AFTER potential previous steps
    numeric_cols_norm = data_to_normalize.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols_norm:
        normalization_method = st.selectbox(
            "Choisir méthode de normalisation:",
            ["Aucune", "Min-Max", "Z-Score"],
            key="norm_select",
            index=0 # Default to 'Aucune'
        )
        if normalization_method != "Aucune":
            cols_to_normalize = st.multiselect(
                "Colonnes numériques à normaliser:",
                numeric_cols_norm,
                default=numeric_cols_norm, # Default to all numeric
                key="norm_cols"
            )
            if st.button(f"Appliquer Normalisation {normalization_method}", key="apply_norm"):
                if not cols_to_normalize:
                     st.warning("Veuillez sélectionner au moins une colonne à normaliser.")
                else:
                    if normalization_method == "Min-Max":
                        data_after_norm_temp = min_max_normalization_manual(data_to_normalize, cols_to_normalize)
                    elif normalization_method == "Z-Score":
                        data_after_norm_temp = z_score_normalization_manual(data_to_normalize, cols_to_normalize)
                    else: # Should not happen with selectbox
                        data_after_norm_temp = data_to_normalize # No change

                    st.session_state.data_after_normalization = data_after_norm_temp
                    st.success(f"Normalisation {normalization_method} appliquée.")
                    st.dataframe(st.session_state.data_after_normalization.head())
                    current_data_stage = st.session_state.data_after_normalization # Update current stage
                    st.rerun() # Rerun to reflect changes

    else:
        st.info("Aucune colonne numérique détectée dans les données actuelles pour la normalisation.")

    # Display data after potential normalization
    if st.session_state.data_after_normalization is not None:
         st.markdown("**Aperçu après normalisation:**")
         st.dataframe(st.session_state.data_after_normalization.head())
         st.write(f"Dimensions: {st.session_state.data_after_normalization.shape}")
         current_data_stage = st.session_state.data_after_normalization # Ensure current stage updated

         # CHANGE 2: Add Download button after this step
         st.download_button(
            label="📥 Télécharger Données après Normalisation (.csv)",
            data=convert_df_to_csv(current_data_stage),
            file_name='data_after_normalization.csv',
            mime='text/csv',
            key='download_normalized'
         )
    elif normalization_method == 'Aucune' and 'norm_select' in st.session_state: # Check if selectbox exists
         # If no strategy applied, the data for the next step is the same as before this section
         st.session_state.data_after_normalization = current_data_stage


# --- 2.5 Visualisation (Post-Processing) ---
st.markdown("---")
st.subheader("2.5 Visualisation (Après Prétraitement)")
# Use the latest available data stage
if current_data_stage is not None:
    viz_col1, viz_col2 = st.columns(2)
    data_for_viz = current_data_stage

    with viz_col1:
        if st.button("Afficher Box Plot (Après Prétraitement)", key="show_boxplot_post"):
            fig_bp = box_plot_st(data_for_viz)
            if fig_bp:
                st.pyplot(fig_bp)
            else:
                st.warning("Impossible de générer le Box Plot avec les données actuelles.")

    with viz_col2:
        if st.button("Afficher Scatter Plots (Après Prétraitement)", key="show_scatterplot_post"):
            st.write("Affichage du Pair Plot (toutes paires numériques)")
            fig_sp = scatter_plot_all_st(data_for_viz) # Default pairplot
            if fig_sp:
                st.pyplot(fig_sp)
            else:
                 st.warning("Impossible de générer les Scatter Plots avec les données actuelles.")


# ================================================
# SECTION 3: CLUSTERING 
# ================================================
from scipy.spatial.distance import pdist, squareform
# ================================================
# SECTION 3: CLUSTERING ANALYSIS
# ================================================

import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.kmedoids import kmedoids
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from mpl_toolkits.mplot3d import Axes3D

def calculate_distances(X, labels):
    """Calculate intra-class and inter-class distances (for K-Means/K-Medoids only)"""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or -1 in unique_labels:
        return None, None
    
    centroids = []
    intra_dists = []
    
    # Calculate intra-cluster distances
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        intra_dists.extend(np.linalg.norm(cluster_points - centroid, axis=1))
    
    # Calculate inter-cluster distances
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    
    return np.mean(intra_dists), np.mean(inter_dists)

# Initialize results history in session state if not exists
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = []

st.header("3. Clustering Analysis")
    
if current_data_stage is not None:
    # Prepare data - use only numeric columns and drop NA
    numeric_cols = current_data_stage.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found for clustering!")
        st.stop()
    
    # Let user select columns for clustering
    selected_cols = st.multiselect(
        "Select columns for clustering",
        options=numeric_cols,
        default=numeric_cols,
        key='clustering_columns'
    )
    
    if not selected_cols:
        st.warning("Please select at least one column for clustering.")
        st.stop()
    
    data_for_clustering = current_data_stage[selected_cols].dropna()
    if len(data_for_clustering) < 2:
        st.warning("Not enough data points for clustering after dropping NA values!")
        st.stop()
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_clustering)
    max_clusters = len(data_scaled)  # Max clusters based on number of data points
    
    # Elbow Method with adjustable K range
    st.subheader("Elbow Method for K Selection")
    col1, col2 = st.columns(2)
    with col1:
        min_k = st.number_input("Minimum K", 2, max_clusters, 2, key='elbow_min')
    with col2:
        max_k = st.number_input("Maximum K", min_k, max_clusters, min(5, max_clusters), key='elbow_max')
    
    if st.button("Run Elbow Analysis"):
        wcss = []
        k_range = range(min_k, max_k+1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data_scaled)
            wcss.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(k_range, wcss, 'bo-', markersize=8)
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        ax.grid(True)
        st.pyplot(fig)

    # Clustering algorithm selection
    st.subheader("Clustering Configuration")
    algorithm = st.selectbox(
        "Select Algorithm",
        ["K-Means", "K-Medoids", "AGNES", "DIANA", "DBSCAN"],
        key='algo_select'
    )
    
    # Algorithm-specific parameters
    if algorithm in ["K-Means", "K-Medoids", "AGNES", "DIANA"]:
        k = st.slider("Number of clusters", 2, max_clusters, min(3, max_clusters), key='n_clusters')
    else:  # DBSCAN parameters
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("Epsilon (ε)", 0.1, 2.0, 0.5, step=0.1)
        with col2:
            min_samples = st.slider("Minimum samples", 2, 20, 5)

    # Visualization type selection
    viz_type = st.radio(
        "Visualization Type",
        ["2D", "3D"],
        index=0,
        key='viz_type'
    )
    
    if st.button("Run Clustering Algorithm"):
        start_time = time.time()
        results = {
            "Algorithm": algorithm,
            "Time (s)": None,
            "Silhouette": None,
            "Intra-dist": None,
            "Inter-dist": None,
            "Clusters": None,
            "Noise": None,
            "Parameters": ""
        }
        
        clusters = None
        
        try:
            if algorithm == "K-Means":
                model = KMeans(n_clusters=k, random_state=42)
                clusters = model.fit_predict(data_scaled)
                intra, inter = calculate_distances(data_scaled, clusters)
                results["Parameters"] = f"k={k}"
                results["Intra-dist"] = intra
                results["Inter-dist"] = inter

            elif algorithm == "K-Medoids":
                initial_medoids = np.random.choice(data_scaled.shape[0], k, replace=False)
                model = kmedoids(data_scaled, initial_medoids)
                model.process()
                cluster_labels = np.full(data_scaled.shape[0], -1)
                for idx, cluster in enumerate(model.get_clusters()):
                    cluster_labels[cluster] = idx
                clusters = cluster_labels
                intra, inter = calculate_distances(data_scaled, clusters)
                results["Parameters"] = f"k={k}"
                results["Intra-dist"] = intra
                results["Inter-dist"] = inter

            elif algorithm == "AGNES":
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                clusters = model.fit_predict(data_scaled)
                results["Parameters"] = f"k={k}, linkage=ward"

            elif algorithm == "DIANA":
                Z = linkage(data_scaled, 'ward')
                clusters = fcluster(Z, k, criterion='maxclust') - 1
                results["Parameters"] = f"k={k}"

            elif algorithm == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(data_scaled)
                results["Parameters"] = f"ε={eps}, min_samples={min_samples}"

            # Calculate metrics
            valid_mask = clusters != -1
            n_clusters = len(np.unique(clusters[valid_mask]))
            
            if n_clusters > 1:
                results["Silhouette"] = silhouette_score(
                    data_scaled[valid_mask], 
                    clusters[valid_mask]
                )
                
            results["Time (s)"] = time.time() - start_time
            results["Clusters"] = n_clusters
            results["Noise"] = sum(clusters == -1)

            # Add to results history
            st.session_state.clustering_results.append(results)

        except Exception as e:
            st.error(f"Clustering failed: {str(e)}")
            st.stop()

        # Display current results
        st.subheader("Current Results")
        current_results_df = pd.DataFrame([results])
        st.dataframe(current_results_df)

        # Visualization
        st.subheader("Cluster Visualization")
        
        if viz_type == "2D":
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(data_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                                c=clusters, cmap='viridis', alpha=0.7)
            
            if algorithm in ["K-Means", "K-Medoids"]:
                centers = pca.transform(model.cluster_centers_) if algorithm == "K-Means" \
                          else pca.transform(data_scaled[initial_medoids])
                marker = 'X' if algorithm == "K-Means" else 'D'
                ax.scatter(centers[:, 0], centers[:, 1],
                          marker=marker, s=200, c='red',
                          edgecolor='black', label='Centers')
            
            ax.set_title(f"{algorithm} Clustering Result (2D)")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            plt.colorbar(scatter)
            st.pyplot(fig)
            
        else:  # 3D visualization
            pca = PCA(n_components=3)
            reduced_data = pca.fit_transform(data_scaled)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                                c=clusters, cmap='viridis', alpha=0.7)
            
            if algorithm in ["K-Means", "K-Medoids"]:
                centers = pca.transform(model.cluster_centers_) if algorithm == "K-Means" \
                          else pca.transform(data_scaled[initial_medoids])
                marker = 'X' if algorithm == "K-Means" else 'D'
                ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                          marker=marker, s=200, c='red',
                          edgecolor='black', label='Centers')
            
            ax.set_title(f"{algorithm} Clustering Result (3D)")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_zlabel("PCA Component 3")
            fig.colorbar(scatter, ax=ax, pad=0.1)
            st.pyplot(fig)

    # Results history and comparison
    if st.session_state.clustering_results:
        st.subheader("Results Comparison")
        
        # Create dataframe from results history
        results_df = pd.DataFrame(st.session_state.clustering_results)
        
        # Format the dataframe for display
        display_df = results_df.copy()
        display_df["Silhouette"] = display_df["Silhouette"].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
        display_df["Intra-dist"] = display_df["Intra-dist"].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
        display_df["Inter-dist"] = display_df["Inter-dist"].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
        display_df["Time (s)"] = display_df["Time (s)"].apply(lambda x: f"{x:.3f}")
        
        # Display the table with checkboxes for removal
        cols = st.columns([5,1])
        with cols[0]:
            st.dataframe(display_df)
        
        with cols[1]:
            st.write("Remove:")
            for i in range(len(results_df)):
                if st.button(f"❌ {i}", key=f"remove_{i}"):
                    st.session_state.clustering_results.pop(i)
                    st.rerun()
        
        # Plot metrics comparison
        if len(results_df) > 1:
            st.subheader("Metrics Comparison")
            
            # Filter only algorithms that have silhouette scores
            valid_results = results_df[results_df["Silhouette"].notna()]
            
            if len(valid_results) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = range(len(valid_results))
                
                # Plot silhouette scores
                ax.bar(x, valid_results["Silhouette"], width=0.4, label='Silhouette')
                
                # Plot intra/inter distances if available
                if "Intra-dist" in valid_results.columns and "Inter-dist" in valid_results.columns:
                    intra_valid = valid_results[valid_results["Intra-dist"].notna()]
                    if len(intra_valid) > 0:
                        ax2 = ax.twinx()
                        ax2.plot(x, valid_results["Intra-dist"], 'r--', marker='o', label='Intra-dist')
                        ax2.plot(x, valid_results["Inter-dist"], 'g--', marker='s', label='Inter-dist')
                        ax2.set_ylabel('Distance')
                        ax2.legend(loc='upper right')
                
                ax.set_xticks(x)
                ax.set_xticklabels(valid_results["Algorithm"], rotation=45)
                ax.set_xlabel('Algorithm')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Algorithm Performance Comparison')
                ax.legend(loc='upper left')
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            else:
                st.info("Not enough algorithms with valid silhouette scores for comparison.")