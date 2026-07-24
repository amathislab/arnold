import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import os
import numpy as np
import matplotlib.pyplot as plt
from definitions import ROOT_DIR
import seaborn as sns
import json

def load_results(results_dir):
    """Load all PCA inactivation results from directory"""
    individual_results = {}
    combined_results = {}

    for filename in os.listdir(results_dir):
        if not filename.endswith('.pkl'):
            continue
            
        filepath = os.path.join(results_dir, filename)
        results = np.load(filepath, allow_pickle=True).item()
        task = results['task']
        if 'combined' in filename:
            combined_results[task] = results
        else:
            individual_results[task] = results
            
    return individual_results, combined_results

def load_benchmark_results(benchmark_path):
    """Load benchmark results to get full task performance"""
    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)
    return {task: data['avg_solved_steps'] for task, data in benchmark_data.items()}

def plot_task_comparison(individual_results, combined_results, benchmark_results, output_dir='figures'):
    """Create summary plots comparing all tasks with normalized performance and error bars"""
    # Set font sizes
    SMALL_SIZE = 16    # was 8
    MEDIUM_SIZE = 20   # was 10
    BIGGER_SIZE = 24   # was 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    tasks = sorted(individual_results.keys())
    n_components = 39
    
    fig_titles = ['Individual PCA Performance', 'Combined PCA Performance']
    data_sets = [individual_results, combined_results]
    filenames = ['individual_pca_summary.png', 'combined_pca_summary.png']
    
    # Set up colors using a combination of color palettes
    colors = (sns.color_palette("husl", 8) +  # Bright distinct colors
             sns.color_palette("Set2", 8) +    # Softer colors
             sns.color_palette("Dark2", 8))    # Darker colors
    
    for title, results, filename in zip(fig_titles, data_sets, filenames):
        plt.figure(figsize=(12, 8))
        
        for task_idx, task in enumerate(tasks):
            if task in results:
                # Get performance values and errors in reverse order (1 to 39 components)
                solved_fractions = [d['solved_mean'] for d in results[task]['performance']][::-1]
                solved_errors = [d['solved_sem'] for d in results[task]['performance']][::-1]
                component_nums = range(1, n_components + 1)
                
                # Normalize by benchmark performance
                benchmark_perf = benchmark_results[task]
                normalized_fractions = np.array(solved_fractions) / benchmark_perf
                normalized_errors = np.array(solved_errors) / benchmark_perf
                
                # Plot line and error bars
                plt.errorbar(component_nums, normalized_fractions, 
                           yerr=normalized_errors,
                           label=task, 
                           color=colors[task_idx],
                           alpha=0.7,
                           marker='o',
                           markersize=3,
                           capsize=2,
                           capthick=1,
                           elinewidth=1)
        
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Normalized Performance (relative to full performance)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Full Performance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def plot_individual_tasks(individual_results, combined_results, benchmark_results, output_dir='figures'):
    """Create individual plots for each task with normalized performance and error bars"""
    # Use the same font sizes as above
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    
    tasks = sorted(individual_results.keys())
    n_components = 39
    
    for task in tasks:
        # Get performance values and errors in reverse order
        ind_perf = [d['solved_mean'] for d in individual_results[task]['performance']][::-1]
        ind_err = [d['solved_sem'] for d in individual_results[task]['performance']][::-1]
        com_perf = [d['solved_mean'] for d in combined_results[task]['performance']][::-1]
        com_err = [d['solved_sem'] for d in combined_results[task]['performance']][::-1]
        
        # Normalize by benchmark performance
        benchmark_perf = benchmark_results[task]
        ind_norm = np.array(ind_perf) / benchmark_perf
        ind_err_norm = np.array(ind_err) / benchmark_perf
        com_norm = np.array(com_perf) / benchmark_perf
        com_err_norm = np.array(com_err) / benchmark_perf
        
        component_nums = range(1, n_components + 1)
        
        plt.figure(figsize=(10, 6))
        # Plot individual PCA performance with error bands
        plt.plot(component_nums, ind_norm, 
                label='Task-specific PCA', color='blue', alpha=0.7,
                marker='o', markersize=3)
        plt.fill_between(component_nums,
                        ind_norm - ind_err_norm,
                        ind_norm + ind_err_norm,
                        color='blue', alpha=0.2)
        
        # Plot combined PCA performance with error bands
        plt.plot(component_nums, com_norm, 
                label='Combined PCA', color='red', alpha=0.7,
                marker='o', markersize=3)
        plt.fill_between(component_nums,
                        com_norm - com_err_norm,
                        com_norm + com_err_norm,
                        color='red', alpha=0.2)
        
        plt.axhline(y=1.0, color='gray', linestyle='--', 
                   alpha=0.5, label='Full Performance')
        
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Normalized Performance (relative to full performance)')
        plt.title(f'PCA Component Analysis - {task}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pca_comparison_{task}.png'), dpi=300)
        plt.close()

def main():
    # Setup
    results_dir = os.path.join(ROOT_DIR, "data/pca_analysis/285_64670238")
    output_dir = os.path.join(ROOT_DIR, 'data/figures/pca_inactivation/285_64670238')
    benchmark_path = os.path.join(ROOT_DIR, "data/final_benchmarks/arnold/seed_0/seed_0_rl_model_64670238_steps_results.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    sns.set_palette('husl')
    
    # Load results
    print("Loading results...")
    individual_results, combined_results = load_results(results_dir)
    benchmark_results = load_benchmark_results(benchmark_path)
    
    # Create plots
    print("Creating summary plots...")
    plot_task_comparison(individual_results, combined_results, benchmark_results, output_dir=output_dir)
    
    print("Creating individual task plots...")
    plot_individual_tasks(individual_results, combined_results, benchmark_results, output_dir=output_dir)
    
    print("Done! Plots saved to:", output_dir)

if __name__ == "__main__":
    main()
