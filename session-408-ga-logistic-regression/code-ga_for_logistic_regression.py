import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import os
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")

# ==================== Part 1: 遗传算法框架 ====================
class GeneticAlgorithm:
    def __init__(
            self,
            n_population=100,
            n_generations=100,
            crossover_rate=0.8,
            mutation_rate=0.2,
            n_elite=5,
            tournament_size=3,
    ):
        self.n_pop = n_population
        self.n_gen = n_generations
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.n_elite = n_elite
        self.tournament_size = tournament_size

        # 记录变量
        self.best_individual = None
        self.best_fitness = float("inf")
        self.fitness_history = []
        self.avg_fitness_history = []
        self.best_individuals_history = []

    def run(self, problem):
        """运行遗传算法主循环"""
        # 1. 初始化种群
        population = [problem.create_individual() for _ in range(self.n_pop)]

        for gen in range(self.n_gen):
            # 2. 评估种群适应度
            fitness = []
            for ind in population:
                fit = problem.evaluate_fitness(ind)
                fitness.append(fit)
            fitness = np.array(fitness)

            # 3. 记录并更新全局最优
            min_fit_idx = np.argmin(fitness)
            current_best_fitness = fitness[min_fit_idx]
            current_best_individual = population[min_fit_idx].copy()

            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = current_best_individual.copy()

            # 4. 记录历史
            self.fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(np.mean(fitness))
            self.best_individuals_history.append(current_best_individual)

            # 5. 选择
            selected = self._tournament_selection(population, fitness)

            # 6. 交叉和变异生成下一代
            next_population = []

            # 6.1 精英保留
            elite_indices = np.argsort(fitness)[: self.n_elite]
            for idx in elite_indices:
                next_population.append(population[idx].copy())

            # 6.2 生成后代
            while len(next_population) < self.n_pop:
                parent_indices = np.random.choice(len(selected), 2, replace=False)
                p1 = selected[parent_indices[0]]
                p2 = selected[parent_indices[1]]
                child1, child2 = p1.copy(), p2.copy()

                if np.random.rand() < self.cr:
                    child1, child2 = self._uniform_crossover(child1, child2)

                child1 = self._gaussian_mutation(child1)
                child2 = self._gaussian_mutation(child2)

                next_population.extend([child1, child2])

            population = next_population[: self.n_pop]

        return self.best_individual

    def _tournament_selection(self, population, fitness):
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            contenders = np.random.choice(
                len(population), self.tournament_size, replace=False
            )
            best_idx = contenders[np.argmin(fitness[contenders])]
            selected.append(population[best_idx])
        return selected

    def _uniform_crossover(self, parent1, parent2):
        mask = np.random.rand(len(parent1)) > 0.5
        child1 = parent1.copy()
        child2 = parent2.copy()
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
        return child1, child2

    def _gaussian_mutation(self, individual, scale=0.1):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.rand() < self.mr:
                mutated[i] += np.random.randn() * scale
        return mutated


# ==================== Part 2: 逻辑回归问题定义 ====================
class LogisticRegressionProblem:
    """定义逻辑回归问题，供遗传算法优化"""

    def __init__(self, X, y):
        self.X = np.c_[X, np.ones((X.shape[0], 1))]  # 添加偏置项
        self.y = y

    def create_individual(self):
        """随机生成个体 - 从标准正态分布中采样"""
        n_params = self.X.shape[1]
        # 从标准正态分布中采样，不限制范围
        return np.random.randn(n_params)

    def evaluate_fitness(self, individual):
        """适应度函数：1 - 准确率"""
        logits = self.X @ individual
        y_pred_prob = 1 / (1 + np.exp(-logits))
        y_pred = (y_pred_prob > 0.5).astype(int)
        acc = accuracy_score(self.y, y_pred)
        return 1.0 - acc


# ==================== Part 3: 可视化函数 ====================
def plot_fitness_history(ga_instance, filename="fitness_history.png"):
    """绘制适应度历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(ga_instance.fitness_history, label='Best Fitness', linewidth=2, color='red')
    plt.plot(ga_instance.avg_fitness_history, label='Average Fitness',
             alpha=0.7, color='blue', linewidth=1.5)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness (1 - Accuracy)', fontsize=12)
    plt.title('GA Fitness History (100 Generations)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'code-plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_evolution(ga_instance, filename="parameter_evolution.png"):
    """绘制参数进化过程"""
    if not ga_instance.best_individuals_history:
        return

    params_history = np.array(ga_instance.best_individuals_history)
    n_params = min(6, params_history.shape[1])  # 最多显示6个参数

    plt.figure(figsize=(12, 8))
    for i in range(n_params):
        label = f'w{i + 1}' if i < n_params - 1 else 'bias'
        plt.plot(params_history[:, i], label=label, alpha=0.7, linewidth=1.5)

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Parameter Value', fontsize=12)
    plt.title('Parameter Evolution of Best Individuals', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'code-plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


def plot_decision_boundary_single(X, y, ga_params, sklearn_params, f1, f2, filename):
    """
    绘制单个决策边界图，只显示决策边界线和数据点
    """
    # 提取这两个特征的数据
    X_2d = X[:, [f1, f2]]

    # 创建网格
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 创建包含偏置的网格点
    grid_points = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape[0])]

    plt.figure(figsize=(10, 8))

    # GA模型预测
    ga_features = np.array([ga_params[f1], ga_params[f2], ga_params[-1]])  # 使用对应的权重和偏置
    ga_logits = grid_points @ ga_features
    ga_Z = 1 / (1 + np.exp(-ga_logits))
    ga_Z = ga_Z.reshape(xx.shape)

    # Sklearn模型预测
    sk_features = np.array([sklearn_params[f1], sklearn_params[f2], sklearn_params[-1]])
    sk_logits = grid_points @ sk_features
    sk_Z = 1 / (1 + np.exp(-sk_logits))
    sk_Z = sk_Z.reshape(xx.shape)

    # 绘制决策边界（只绘制轮廓线，不填充）
    # GA模型 - 红色轮廓
    plt.contour(xx, yy, ga_Z, levels=[0.5], colors='red', linewidths=3, linestyles='solid')
    # Sklearn模型 - 蓝色轮廓
    plt.contour(xx, yy, sk_Z, levels=[0.5], colors='blue', linewidths=3, linestyles='dashed')

    # 绘制数据点
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm',
                edgecolor='k', s=50, alpha=0.8)

    plt.xlabel(f'Feature {f1 + 1}', fontsize=12)
    plt.ylabel(f'Feature {f2 + 1}', fontsize=12)
    plt.title(f'Decision Boundary: Feature {f1 + 1} vs Feature {f2 + 1}\n'
              f'Red: GA Model, Blue: Sklearn Model', fontsize=14, fontweight='bold')

    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='GA Decision Boundary'),
        Line2D([0], [0], color='blue', lw=3, linestyle='dashed', label='Sklearn Decision Boundary'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'code-plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


# ==================== Part 4: 主程序 ====================
def main():
    # 创建code-plots文件夹
    os.makedirs("code-plots", exist_ok=True)

    # 设置随机种子
    np.random.seed(42)

    # 1. 生成分类数据
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42
    )

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 使用GA优化逻辑回归
    print("\n>>> 使用遗传算法优化逻辑回归 (100代)...")
    print("   参数配置: crossover_rate=0.85, mutation_rate=0.15, n_elite=5, tournament_size=3")
    logr_problem = LogisticRegressionProblem(X_train_scaled, y_train)

    ga = GeneticAlgorithm(
        n_population=100,
        n_generations=100,
        crossover_rate=0.85,
        mutation_rate=0.15,
        n_elite=5,
        tournament_size=3
    )

    best_params_ga = ga.run(logr_problem)
    print(f"GA找到的最优参数: {best_params_ga}")

    # 5. 在测试集上评估GA模型
    X_test_with_bias = np.c_[X_test_scaled, np.ones((X_test_scaled.shape[0], 1))]
    logits_ga = X_test_with_bias @ best_params_ga
    y_pred_prob_ga = 1 / (1 + np.exp(-logits_ga))
    y_pred_ga = (y_pred_prob_ga > 0.5).astype(int)
    ga_accuracy = accuracy_score(y_test, y_pred_ga)
    print(f"GA模型测试集准确率: {ga_accuracy:.4f}")

    # 6. 训练Sklearn的逻辑回归模型
    print("\n>>> 训练Sklearn逻辑回归模型...")
    sklearn_lr = LogisticRegression(random_state=42, max_iter=1000)
    sklearn_lr.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_lr.predict(X_test_scaled)
    sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
    print(f"Sklearn模型测试集准确率: {sklearn_accuracy:.4f}")

    # 获取Sklearn模型的参数
    sklearn_params = np.append(sklearn_lr.coef_[0], sklearn_lr.intercept_[0])
    print(f"Sklearn模型参数: {sklearn_params}")

    # 7.1 适应度历史
    plot_fitness_history(ga, "fitness_history.png")

    # 7.2 参数进化
    plot_parameter_evolution(ga, "parameter_evolution.png")

    # 7.3 10个独立的决策边界图
    # 生成所有可能的特征对组合
    n_features = X_train_scaled.shape[1]
    feature_pairs = list(combinations(range(n_features), 2))

    # 生成10个决策边界图
    for idx, (f1, f2) in enumerate(feature_pairs, 1):
        filename = f"decision_boundaries_{idx}.png"
        plot_decision_boundary_single(X_train_scaled, y_train, best_params_ga, sklearn_params, f1,
                                      f2, filename)

    with open("code-plots/parameters_comparison.txt", "w") as f:
        f.write("=" * 50 + "\n")
        f.write("GA vs Sklearn 参数对比\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"GA准确率: {ga_accuracy:.4f}\n")
        f.write(f"Sklearn准确率: {sklearn_accuracy:.4f}\n")
        f.write(f"准确率差异: {abs(ga_accuracy - sklearn_accuracy):.4f}\n\n")

        f.write("遗传算法参数配置:\n")
        f.write(f"  n_population={ga.n_pop}, n_generations={ga.n_gen}\n")
        f.write(f"  crossover_rate={ga.cr}, mutation_rate={ga.mr}\n")
        f.write(f"  n_elite={ga.n_elite}, tournament_size={ga.tournament_size}\n\n")

        f.write("参数对比:\n")
        f.write(f"{'参数':<10} {'GA值':<15} {'Sklearn值':<15} {'差异':<15}\n")
        f.write("-" * 55 + "\n")

        param_names = [f'w{i + 1}' for i in range(len(best_params_ga) - 1)] + ['bias']
        for i, (name, ga_val, sk_val) in enumerate(
                zip(param_names, best_params_ga, sklearn_params)):
            diff = abs(ga_val - sk_val)
            f.write(f"{name:<10} {ga_val:<15.6f} {sk_val:<15.6f} {diff:<15.6f}\n")


if __name__ == "__main__":
    main()
