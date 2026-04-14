import matplotlib.pyplot as plt
import numpy as np

def plot_environment(env):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

    # Draw obstacles
    for obs in env.obstacles:
        x, y = zip(*obs.to_tuples())
        ax.fill(x, y, color='gray', alpha=0.5, label='Obstacle', zorder=10)

    # Draw sensors
    for s in env.sensors:
        ax.add_patch(
            plt.Circle(
                (s.position.x, s.position.y),
                s.radius,
                facecolor=(0.4, 0.7, 1.0, 0.25),  
                edgecolor=(0.4, 0.7, 1.0, 0.4), 
                linewidth=0.7,
                zorder=1 
            )
        )
        ax.plot(s.position.x, s.position.y, 'o',
                color=(0.4, 0.7, 1.0, 0.7),
                markersize=3,
                zorder=2)

    plt.title("Environment with Sensors and Obstacles")
    plt.show()

def plot_environment_image(env, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')

    # Draw obstacles (top layer)
    for obs in env.obstacles:
        x, y = zip(*obs.to_tuples())
        ax.fill(x, y, color='gray', alpha=0.5, zorder=10)

    # Draw sensors (more visible, below obstacles layer)
    for s in env.sensors:
        ax.add_patch(
            plt.Circle(
                (s.position.x, s.position.y),
                s.radius,
                facecolor=(0.4, 0.7, 1.0, 0.25),
                edgecolor=(0.4, 0.7, 1.0, 0.4),
                linewidth=0.7,
                zorder=1
            )
        )
        ax.plot(s.position.x, s.position.y, 'o',
                color=(0.4, 0.7, 1.0, 0.7),
                markersize=3,
                zorder=2)

    plt.title("Environment")
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_hypervolume_history(hypervolume_history, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = range(len(hypervolume_history))
    ax.plot(generations, hypervolume_history, 'b-', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Hypervolume', fontsize=12)
    ax.set_title('Hypervolume over Generations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pareto_fronts_by_generation(pareto_front_history, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select a subset of generations to display
    selected_gens = []
    n_gens = len(pareto_front_history)
    step = max(1, n_gens // 6)  # Display ~6 generations
    for i in range(0, n_gens, step):
        selected_gens.append(i)
    # Ensure the final generation is included
    if n_gens - 1 not in selected_gens:
        selected_gens.append(n_gens - 1)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_gens)))
    
    for idx, gen in enumerate(selected_gens):
        pareto_front = pareto_front_history[gen]
        if pareto_front:
            f1_values = [obj[0] for obj in pareto_front]
            f2_values = [obj[1] for obj in pareto_front]
            # Sort to draw a continuous line
            sorted_indices = sorted(range(len(f1_values)), key=lambda i: f1_values[i])
            f1_sorted = [f1_values[i] for i in sorted_indices]
            f2_sorted = [f2_values[i] for i in sorted_indices]
            
            marker = 'o' if gen == n_gens - 1 else None
            linewidth = 2.5 if gen == n_gens - 1 else 1.5
            linestyle = '--' if gen == n_gens - 1 else '-'
            label = f"Gen {gen}" if gen != n_gens - 1 else f"Final Gen {gen}"
            
            ax.plot(f1_sorted, f2_sorted, color=colors[idx], marker=marker, 
                   markersize=5, linewidth=linewidth, linestyle=linestyle, label=label)
            if gen != n_gens - 1:
                ax.scatter(f1_values, f2_values, color=colors[idx], s=20, alpha=0.6)
    
    ax.set_xlabel('f1 (negative exposure)', fontsize=12)
    ax.set_ylabel('f2 (length)', fontsize=12)
    ax.set_title('Pareto Front by Generation', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pareto_size_history(pareto_size_history, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = range(len(pareto_size_history))
    ax.plot(generations, pareto_size_history, 'g-', linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Pareto Front Size', fontsize=12)
    ax.set_title('Pareto Front Size over Generations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
