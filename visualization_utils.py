import pandas as pd
import numpy as np
from itertools import chain, combinations
from matplotlib import pyplot as plt
import seaborn as sns
import os

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class StepFunction:
    """
    Represents a step function defined by a set of x and y values.
    Allows adding "cuts" to add a step to the function at specific x values.
    """
    def __init__(self, x_list=[0, 1], y_list=[0, 0]):
        self.x_list = x_list
        self.y_list = y_list

    def __str__(self):
        return f"X: {self.x_list}, Y: {self.y_list}"

    def __eq__(self, other):
        """
        Checks equality between two step functions.
        Two step functions are equivalent if they both
        represent a flat line at 0, or they have the exact same
        set of steps.
        """
        return (
            np.array_equal(self.x_list, other.x_list) and np.array_equal(self.y_list, other.y_list)
            or (not np.any(self.y_list) and not np.any(other.y_list))
        )

    def add_cut(self, new_x, new_y):
        """Adds a step change to the function at a given x value."""
        if new_x in self.x_list:
            x_arr = np.array(self.x_list)
            y_arr = np.array(self.y_list)

            y_arr[x_arr <= new_x] += new_y

            self.y_list = list(y_arr)
        else:
            x_argsort = np.array(self.x_list + [new_x]).argsort()
            x_arr = np.array(self.x_list + [new_x])[x_argsort]

            if np.array(self.y_list)[np.array(self.x_list) >= new_x].shape[0] > 0:
                new_y_adjusted = np.array(self.y_list)[np.array(self.x_list) >= new_x][0] + new_y
            else:
                new_y_adjusted = new_y
            y_arr = np.array(self.y_list + [new_y_adjusted])[x_argsort]
            y_arr[x_arr < new_x] += new_y

            self.x_list = list(np.array(self.x_list + [new_x])[x_argsort])
            self.y_list = list(y_arr)

def get_curve(dataset_structure_map, thresh_vals, coefs, range_by_var, target_var='PercentTradesWBalance', missing_vars=([], []), inters=True):
    """
    Constructs a step function representing the effect of a variable on the model.
    """
    sf = StepFunction(range_by_var[target_var], [0, 0])
    
    if target_var in missing_vars[0]:
        target_type_ind = missing_vars[0].index(target_var)
        target_type = missing_vars[1][target_type_ind]
        ind_of_interest = dataset_structure_map[target_var]['intercepts'][target_type]
        return StepFunction(range_by_var[target_var], [coefs[ind_of_interest]] * 2)
    
    for b in dataset_structure_map[target_var]['bins']:
        sf.add_cut(thresh_vals[b], coefs[b])
    
    if inters:
        for other_var, interactions in dataset_structure_map[target_var]['interactions'].items():
            if other_var in missing_vars[0]:
                target_type_ind = missing_vars[0].index(other_var)
                target_type = missing_vars[1][target_type_ind]
                for b in interactions[target_type]:
                    sf.add_cut(thresh_vals[b], coefs[b])
    
    return sf

def check_interesting_missingness(dataset_structure_map, coefs, inters=True):
    """
    Identifies variables where missing values contribute to model behavior.
    """
    miss_cols, miss_types = [], []
    
    for var, details in dataset_structure_map.items():
        if inters:
            for v_inter, inters_dict in details['interactions'].items():
                for m_type, indices in inters_dict.items():
                    if any(coefs[i] != 0 for i in indices):
                        if (v_inter not in miss_cols) or (miss_types[miss_cols.index(v_inter)] != m_type):
                            miss_cols.append(v_inter)
                            miss_types.append(m_type)
        
        for m_type, coef_idx in details['intercepts'].items():
            if coefs[coef_idx] != 0:
                if (var not in miss_cols) or (miss_types[miss_cols.index(v)] != m_type):
                    miss_cols.append(var)
                    miss_types.append(m_type)
    
    return miss_cols, miss_types


def get_missingness_sets_for_model(dataset_structure_map, model_coef, inters=True):
    """
    Generates all possible sets of missingness scenarios affecting the model.
    """
    possible_missing_vars = check_interesting_missingness(dataset_structure_map, model_coef, inters=inters)
    
    missingness_sets = [p for p in powerset(zip(possible_missing_vars[0], possible_missing_vars[1]))]
    shaped_missingness_sets = []
    for s in missingness_sets:
        names = [cur[0] for cur in s]
        vals = [cur[1] for cur in s]
        shaped_missingness_sets.append((names, vals))

    return shaped_missingness_sets

def plot_shape_functions(
    train_df,
    binarizer,
    fit_model,
    target_lambda,
    save_loc,
    figsize_height_scale=4,
    figsize_width_scale=5.5
):
    assert len(binarizer.categorical_cols) == 0, \
        "Error: Visualization of categorical variables is not yet supported. Will be added soon!"
    coefs = fit_model.coeff(lambda_0=target_lambda).toarray().flatten()
    inter = coefs[0]
    coefs = coefs[1:]
    dataset_structure_map = binarizer.dataset_structure_map
    thresh_vals = binarizer.thresh_vals
    range_by_var = {
       v: [train_df[v].min(), train_df[v].max()] for v in train_df.columns[:-1]
    }

    possible_missing_vars = check_interesting_missingness(dataset_structure_map, coefs, inters=True)
    shaped_missingness_sets = get_missingness_sets_for_model(dataset_structure_map, coefs)

    os.makedirs("/".join(save_loc.split("/")[:-1]), exist_ok=True)
    sns.set(font_scale=2.0)
    sns.set_style(style='white')
    inters = True

    n_interesting = 0
    possible_active_vars = []

    # Used to set the shared y axis across curves
    max_y = 0
    min_y = 0

    # ====== First, lets figure out which variables are different for some missingness case
    complete_case_curves = {
        var: get_curve(dataset_structure_map, thresh_vals, coefs, range_by_var=range_by_var, target_var=var, missing_vars=([], []), inters=False)
        for var in dataset_structure_map
    }

    all_vars = {var for var in dataset_structure_map}
    vars_that_change = set()
    for m_set in shaped_missingness_sets:
        curves_for_cur_case = {
            var: get_curve(dataset_structure_map, thresh_vals, coefs, range_by_var=range_by_var, target_var=var, missing_vars=m_set, inters=False)
            for var in dataset_structure_map
        }
        for var in curves_for_cur_case:
            if not (curves_for_cur_case[var] == complete_case_curves[var]):
                vars_that_change.add(var)

    vars_that_dont_change = list(all_vars - vars_that_change)
    vars_that_dont_change.sort()
    vars_that_change = list(vars_that_change)
    vars_that_change.sort()
    vars_in_order = vars_that_change + vars_that_dont_change


    # ====== This loop fiugres out which variables have a non-zero shape function in some
    # ====== missingness setting
    for missing_vals in shaped_missingness_sets:
        for var in vars_in_order:
            c = get_curve(dataset_structure_map, thresh_vals, coefs, range_by_var=range_by_var,target_var=var, missing_vars=missing_vals, inters=inters)

            # If this variable is used at all, record the extrema of its curve
            if np.any(abs(np.array(c.y_list)) > 0):
                n_interesting += 1
                max_y = max(max_y, max(c.y_list))
                min_y = min(min_y, min(c.y_list))

            for m_set in shaped_missingness_sets:
                c = get_curve(dataset_structure_map, thresh_vals, coefs, range_by_var=range_by_var,target_var=var, missing_vars=m_set, inters=inters)
                if np.any(abs(np.array(c.y_list)) > 0):
                    max_y = max(max_y, max(c.y_list))
                    min_y = min(min_y, min(c.y_list))

                if np.any(abs(np.array(c.y_list)) > 0) and (var not in possible_active_vars):
                    possible_active_vars.append(var)
                elif (var not in possible_active_vars) and (var in vars_that_change):
                    possible_active_vars.append(var)

    # ====== Now, plot each curve in some missingness case
    n_miss_configs = len(shaped_missingness_sets)
    fig, ax = plt.subplots(
        n_miss_configs, len(possible_active_vars), 
        figsize=(len(possible_active_vars)*figsize_height_scale, n_miss_configs*figsize_width_scale)
    )
    if len(ax.shape) == 1:
        ax = ax.reshape((1, -1))

    for row in range(n_miss_configs):
        missing_vals = shaped_missingness_sets[row]
        if len(missing_vals[0]) == 0:
            missingness_label = "Nothing Missing"
        else:
            missingness_label = "\n".join(
                ["Missing: "] + 
                [f"{missing_vals[0][mv_ind]}={missing_vals[1][mv_ind]}" for mv_ind in range(len(missing_vals[0]))]
            )
        for cur_i, var in enumerate(possible_active_vars, start=0):
            if row == 0:
                ax[row, cur_i].set_title(var)
            c = get_curve(dataset_structure_map, thresh_vals, coefs, range_by_var=range_by_var, target_var=var, missing_vars=missing_vals, inters=inters)
            if complete_case_curves[var] == c:
                curve_color = 'blue'
            else:
                curve_color = 'orange'
            
            if (row > 0) and (var not in vars_that_change):
                ax[row, cur_i].set_visible(False)
                continue

            if np.any(abs(np.array(c.y_list)) > 0):
                ax[row, cur_i].step(
                    c.x_list,
                    c.y_list, 
                    where='pre',
                    color=curve_color
                )
            else:
                ax[row, cur_i].step(
                    range_by_var[var], 
                    [0, 0], 
                    where='pre',
                    color=curve_color
                )
            
            cur_range = max_y - min_y
            ax[row, cur_i].set_ylim(min_y - 0.1 * cur_range, max_y + 0.1 * cur_range)
            if cur_i > 0:
                ax[row, cur_i].set_yticks([])
            else:
                ax[row, cur_i].set_ylabel(missingness_label)

    fig.tight_layout(pad=4.0)
    plt.savefig(save_loc)